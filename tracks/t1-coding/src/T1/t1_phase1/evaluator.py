"""Host-only orchestration; generated programs execute only in Docker."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

from .baseline import EVAL_PAUSE_CAP, summarize_checker
from .docker_runner import build_agent_command, build_checker_command, run_container
from .house import HouseConfig, HouseError, eval_throttle
from .staging import stage_unit, _relative

MAX_OUTPUT=64*1024*1024
RESERVED={'reward.json','pytest_report.json','reward.txt'}
LOCAL_AGENT_FAILURES={'agent_input_error','agent_budget_exhausted','agent_resource_limit'}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def verify_dataset(root: Path, manifest: dict, commit: str):
    if manifest.get('source_commit')!=commit:
        raise ValueError('Dataset source commit mismatch')
    expected=set()
    for record in manifest['files']:
        rel=_relative(record['local_path']); expected.add(str(rel))
        path=root/Path(rel)
        if any((root/Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1)):
            raise ValueError('Dataset symlink rejected')
        if not path.is_file(): raise ValueError('Dataset file is missing')
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=record['sha256']: raise ValueError('Dataset hash mismatch')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if not p.is_dir()}
    if actual!=expected: raise ValueError('Dataset contains unmanifested files')


def copy_for_grading(source: Path, destination: Path) -> dict:
    """Only regular output artifacts cross into a fresh trusted verifier run."""
    if destination.exists(): raise ValueError('Grading destination must be fresh')
    files=[]; total=0
    for path in sorted(source.rglob('*')):
        mode=path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError('Output contains links or special files')
        if path.is_dir(): continue
        if path.stat().st_nlink!=1: raise ValueError('Hardlinked output rejected')
        total+=path.stat().st_size
        if total>MAX_OUTPUT: raise ValueError('Output exceeds local size limit')
        rel=path.relative_to(source)
        if any(p in RESERVED or p.startswith('_agent_') for p in rel.parts): continue
        files.append((path,rel))
    destination.mkdir(parents=True,mode=0o777); destination.chmod(0o777)
    hashes={}
    for path,relative in files:
        target=destination/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target); target.chmod(0o666)
        with target.open('rb') as f: hashes[str(relative)]=hashlib.file_digest(f,'sha256').hexdigest()
    for path in destination.rglob('*'):
        if path.is_dir(): path.chmod(0o777)
    return {'files':hashes,'input_bytes':total}


def build_summary(rows: list[dict]) -> dict:
    counts=dict(Counter(row['status'] for row in rows))
    attempted=sum(row.get('model_calls',0)>0 for row in rows)
    passed=counts.get('passed',0)
    # A completed local agent failure is measured even when no model request was sent.
    measured=attempted or any(row['status'] in LOCAL_AGENT_FAILURES for row in rows)
    unavailable=any(row['status'].startswith(('not_run','checker_','infrastructure_')) for row in rows)
    return {'roster_size':len(rows),'model_attempted_units':attempted,'observed_passes':passed,
            'status_counts':counts,'local_pass_rate':passed/len(rows) if rows and measured and not unavailable else None,
            'partial_roster_lower_bound':passed/len(rows) if rows and measured else None,
            'official_score':None,'scope':'local public-unit pipeline; not an official score'}


def agent_outcome(agent: dict, process: dict) -> str:
    """A valid measured failure stays an agent result whether solve exits 0 (it always does once the status is
    written) or 1 (images before B3)."""
    status=agent.get('status')
    calls=agent.get('model_calls',0)
    measured={'generation_failed','execution_failed','execution_timeout','output_limit_exceeded'}
    # Official g2 scans the output before any check runs: an input canary there scores 0 whether the
    # program exited 0, crashed, timed out or overran the output limit.
    canary=type(calls) is int and calls>=1 and bool(agent.get('output_canary_files',0))
    if process.get('timed_out'):
        if canary:
            return 'contamination_canary'
        return 'execution_timeout' if calls else 'infrastructure_agent_timeout'
    if (status in LOCAL_AGENT_FAILURES and type(calls) is int and 0<=calls<=25
            and process.get('returncode') in (0,1)):
        return status
    if status in measured and type(calls) is int and calls>=1:
        return 'contamination_canary' if canary and status!='generation_failed' else status
    if isinstance(status,str) and status.startswith(('not_run','infrastructure_')): return status
    if process.get('returncode')!=0 or status!='executed_ungraded' or type(calls) is not int or calls<1:
        return 'infrastructure_agent_error'
    # Official g2 scans the output before any check runs; an input canary there scores 0.
    if agent.get('output_canary_files',0):
        return 'contamination_canary'
    return 'executed_ungraded'


def read_status(path: Path) -> dict:
    """Never block on a participant-created FIFO/device or follow its symlinks."""
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>20000:
            raise ValueError('Invalid controller status file')
        value=json.loads(stream.read(20001))
    if (not isinstance(value,dict) or type(value.get('model_calls')) is not int
            or not 0<=value.get('model_calls')<=25):  # organizer allowance: 25 requests per unit
        raise ValueError('Invalid controller status schema')
    if not isinstance(value.get('status'),str): raise ValueError('Invalid controller status')
    return value


def preflight(context: str, image: str, platform: str) -> dict:
    config_keys=('MODEL_ENDPOINT','MODEL_NAME','MODEL_TOKEN')
    result={'model_env_present':{k:bool(os.environ.get(k)) for k in config_keys},
            'house_config_valid':False,'docker_ready':False,'image_ready':False,
            'context':context,'platform':platform,'image':image,
            'profile':'local-4cpu-8g; bridge for House; not platform-equivalent'}
    try: HouseConfig.from_env(); result['house_config_valid']=True
    except HouseError: pass
    try:
        info=subprocess.run(['docker','--context',context,'info','--format','{{json .}}'],
            capture_output=True,text=True,timeout=15,check=False)
        if info.returncode==0:
            data=json.loads(info.stdout)
            result['docker_ready']=True
            result['docker_resources']={k:data.get(k) for k in ('NCPU','MemTotal','Architecture','ServerVersion')}
            image_result=subprocess.run(['docker','--context',context,'image','inspect',image],
                capture_output=True,text=True,timeout=10,check=False)
            if image_result.returncode==0:
                metadata=json.loads(image_result.stdout)[0]
                result['image_id']=metadata['Id']; result['image_arch']=metadata['Architecture']
                result['image_ready']=platform.split('/')[-1]==metadata['Architecture']
    except (OSError,subprocess.TimeoutExpired,ValueError,KeyError,IndexError): pass
    return result


def prepare(root: Path, participant_root: Path) -> dict:
    suite=json.loads((root/'selection/suite.json').read_text())
    manifest=json.loads((root/'evidence/dataset_manifest.json').read_text())
    verify_dataset(root/'datasets',manifest,suite['source_commit'])
    if participant_root.exists(): raise ValueError('Participant root must be fresh')
    result={}
    for unit in suite['units']:
        result[unit['id']]=stage_unit(root/'datasets'/unit['id'],participant_root/unit['id'])
    return {'source_commit':suite['source_commit'],'units':result}


def evaluate(root: Path, run_dir: Path, *, split='dev', context='colima',
             image='t1-phase1:b0-amd64', platform='linux/amd64', unit_timeout=420,
             total_budget=7200, checker_timeout=600, integration_evidence: Path | None=None) -> dict:
    if run_dir.exists(): raise ValueError('Run directory already exists; refusing to overwrite evidence')
    if min(unit_timeout,total_budget,checker_timeout)<=0: raise ValueError('Budgets must be positive')
    suite=json.loads((root/'selection/suite.json').read_text())
    units=[x for x in suite['units'] if split=='all' or x['split']==split]
    run_dir.mkdir(parents=True)
    readiness=preflight(context,image,platform)
    # Resolve once: a mutable tag must not select a different image after receipt validation.
    image_id=readiness.get('image_id')
    write_json(run_dir/'preflight.json',readiness)
    participant=run_dir/'participant'
    staged=prepare(root,participant)
    write_json(run_dir/'staging.json',staged)
    rows=[{'id':u['id'],'split':u['split'],'model_calls':0,'status':'not_run',
           'official_score':None} for u in units]
    missing=[]
    if not readiness['house_config_valid']: missing.append('house')
    if not readiness['docker_ready']: missing.append('docker')
    elif not readiness['image_ready']: missing.append('image')
    verified=False
    if integration_evidence is not None:
        try:
            fixture=json.loads(integration_evidence.read_text())
            verified=(fixture.get('status')=='passed' and fixture.get('fixture_only') is True
                      and fixture.get('image_id')==image_id
                      and fixture.get('platform')==platform and bool(image_id))
        except (OSError,ValueError,TypeError): pass
    if not verified: missing.append('container_validation')
    started=time.monotonic()
    for unit,row in zip(units,rows):
        if missing:
            row.update(status='not_run_missing_'+'_'.join(missing),missing_prerequisites=missing)
            continue
        remaining=total_budget-(time.monotonic()-started)
        if remaining<60:
            row['status']='not_run_global_deadline'; continue
        # Keep the global deadline separate from this unit's setup/execution/grading time.
        unit_started=time.monotonic()
        unit_dir=run_dir/'units'/unit['id']; out=unit_dir/'agent-output'; report=unit_dir/'verifier-log'
        for path in (out,report): path.mkdir(parents=True); path.chmod(0o777)
        budget=max(1,min(unit_timeout,unit['agent_timeout_sec'],int(remaining)-20))
        cmd=build_agent_command(image_id,participant/unit['id'],out,context=context,
            platform=platform,timeout_sec=budget,model_env=True)
        # T1_EVAL_THROTTLE (forwarded with the model variables) lets solve sleep up to EVAL_PAUSE_CAP s of throttled
        # waits that it does not charge to its budget; the container ceiling grows by the same amount.
        process=run_container(cmd,timeout_sec=budget+10+(EVAL_PAUSE_CAP if eval_throttle() else 0))
        # Logs can include task canaries; preserve only status and byte counts here.
        safe_process={k:v for k,v in process.items() if k not in ('stdout','stderr')}
        safe_process.update(stdout_bytes=len(process['stdout'].encode()),stderr_bytes=len(process['stderr'].encode()))
        write_json(unit_dir/'agent-process.json',safe_process)
        try:
            path=out/'_agent_status.json'
            agent=read_status(path)
            row['model_calls']=agent.get('model_calls',0)
            row['usage']=agent.get('usage',{})
            row['agent_status']=agent.get('status')
            write_json(unit_dir/'agent-status.json',agent)
        except (OSError,ValueError,TypeError):
            agent={'status':'missing_status'}
        row['status']=agent_outcome(agent,process)
        if row['status']=='executed_ungraded':
            grade=unit_dir/'grading-output'
            command=None
            try:
                copied=copy_for_grading(out,grade)
                write_json(unit_dir/'deliverables.json',copied)
            except (ValueError,OSError): row['status']='invalid_output_tree'
            else:
                try:
                    command=build_checker_command(image_id,root/'datasets'/unit['id'],grade,report,
                        context=context,platform=platform)
                except (ValueError,OSError):
                    # The harness cannot present the unit's checks: not the agent's failure.
                    row['status']='checker_error'
            if command is not None:
                try:
                    remaining=total_budget-(time.monotonic()-started)
                    if remaining<=0:
                        row['status']='checker_not_run_global_deadline'
                    else:
                        checker=run_container(command,timeout_sec=min(checker_timeout,remaining))
                        write_json(unit_dir/'checker-process.json',{k:v for k,v in checker.items() if k not in ('stdout','stderr')})
                        row.update(summarize_checker(grade,checker))
                except (ValueError,OSError): row['status']='invalid_output_tree'
        finished=time.monotonic()
        row['duration_sec']=round(finished-unit_started,3)
        row['run_elapsed_sec']=round(finished-started,3)
        write_json(run_dir/'results.json',{'units':rows,'summary':build_summary(rows)})
    result={'source_commit':suite['source_commit'],'selection_version':suite['selection_version'],
            'baseline':'B0_one_generation','split':split,'unit_budget_sec':unit_timeout,
            'agent_image_requested':image,'agent_image_id':image_id,
            'total_budget_sec':total_budget,'units':rows,'summary':build_summary(rows)}
    write_json(run_dir/'results.json',result)
    return result
