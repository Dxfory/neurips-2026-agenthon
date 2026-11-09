"""Opt-in worker diagnostics, never a trusted scorer or an answer cache."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid

MAX_REPORT_BYTES = 262144
KINDS = frozenset(('contract', 'close', 'balance', 'bounds', 'convergence', 'estimate',
                   'overflow', 'duplicate', 'skip', 'invalid_id', 'report_write', 'coverage'))
STATUSES = ('passed', 'failed', 'skipped', 'unverified')
USAGE_KINDS = ('checks','semantics','contracts','verification','resources','workflow',
               'documents','evidence','cache')
USAGE_FIELDS = ('runtime_usage_state',) + tuple('runtime_use_'+k for k in USAGE_KINDS)
FIELDS = ('runtime_state', 'runtime_checks', 'requirement_state', 'requirement_checks') + tuple('runtime_' + s for s in STATUSES) + USAGE_FIELDS
_ACTIVE = ContextVar('t1_runtime', default=None)

PROMPT = '''RUNTIME TOOLS (task requirements take precedence):
from qfruntime import Checks, StageCache, probe, EvidenceLedger, availability_filter, asof_vintages
Use Checks() for one defensible TASK-relevant check, or record why none applies. Signatures:
c.contract(name, values, shape=(None, 3), actual_unit=unit, expected_unit=required_unit,
           finite=True, lower=None, upper=None, choices=None, method=None, expected_method=None)
c.close(name, actual, independent_reference, atol=..., rtol=...,
        computed_method=..., reference_method=...)
c.balance(name, parts, total, atol=..., rtol=...); c.bounds(name, values, lower=..., upper=..., atol=0)
c.convergence(name, coarse, fine, atol=..., rtol=...)
c.estimate(name, estimate, reference, standard_error, z=...); c.skip(name, reason)
Choose tolerances from TASK and justified numerical precision before computing. Never relax checks,
drop failed checks, or substitute a cached answer during repairs. Input contract failures raise;
numerical failures are reported without preventing deliverables. Distinct method labels alone do
not establish independence: compute the reference by a separate formula or algorithm.
Lifecycle:
c = Checks()
try:
    ...  # compute TASK outputs and run the selected check, or c.skip with the reason
finally:
    c.finish()

Shared semantics: load_task_spec() from qfruntime reads T1_TASK_SPEC when supplied. It extracts only
explicit definitions; .unresolved needs your interpretation of TASK. Build QuantitySpec(name, unit=...,
shape=..., axes=..., method=..., time_basis=..., timezone=..., atol=..., rtol=..., period=...,
source=SourceRef(source_id,start,end,exact_quote)); unknown fields stay None. TaskSpec.from_dict(payload,
sources={source_id: original_text}) verifies source spans/content, not your interpretation. Use the same
QuantitySpec throughout parsing, contracts, execution and verification; conflicting definitions fail.
enforce_contract(c,name,value,expected=quantity,observed=actual_quantity) validates declared meanings;
convert_unit(value,from_unit,to_unit) explicitly converts fraction/percent/bps, never currency or method.
verify_quantity(c,name,quantity=quantity,actual=value,operation=...,inputs=...) selects supported
diagnostics using quantity.method/atol/rtol: integral inputs={function,lower,upper}, derivative={function,x},
linear_system={matrix,rhs}, sum/cashflows={parts}, probabilities={}. Unsupported methods or unconverged
references stay unverified. Residuals and sums are consistency checks, not independent solutions.

Checks automatically writes a controller-private scratch report via T1_RUNTIME_REPORT; keep the
default path. This report is allowed but is not a deliverable. Empty or skipped checks remain unverified.
For expensive work, probe(compute, sample_size=..., target_size=..., time_budget=...,
memory_budget=None, growth_power=1, fixed_seconds=0) runs compute(sample_size) once and reports an
estimated cost under the stated scaling assumption; memory is unverified. Preserve TASK precision,
sample counts and RNG state for the full run. T1_RUNTIME_BUDGET_SEC is the current run budget.
measure_resources(compute) measures wall/CPU time and process-lifetime peak RSS, not incremental memory.
run_batches(draw(count,rng),max_samples=...,batch_size=...,required_min_samples=...,absolute_error=...,
seed=...,budget=ResourceBudget(wall_seconds=...,rss_bytes=...)) preserves required sample counts.
Inspect status/reason: exhausted budgets are incomplete; normal-SE estimates do not guarantee coverage.
Optional per-solve cache: StageCache(os.environ['T1_RUNTIME_CACHE'],
namespace=os.environ['T1_RUNTIME_NAMESPACE']).run(name, version=..., inputs=JSON_inputs,
parameters=JSON_parameters, dependencies=[upstream_names], compute=lambda: ..., validate=lambda result: ...).
Include actual input content or its content hash, every parameter and code version. Run dependencies
in the current process first; each hit is revalidated. Cache JSON intermediates, never final answers.
For declared dependency graphs use Stage(name,compute,validate,version,dependencies=(),inputs=...,
parameters=...,semantic_spec=task_spec), where compute(inputs,parameters,dependencies) returns JSON.
Workflow(stages,cache=cache).run(targets=None) orders dependencies and blocks failed descendants.
read_document(root,relative_path,format=None,limits=DocumentLimits()) reads bounded text/HTML/CSV/JSON.
extract_quantity(document,quantity,selectors={role:field_name},as_of=...) retains all candidates and
exact provenance; resolve ambiguous/unverified results explicitly. No PDF/OCR or automatic unit guessing.
EvidenceLedger().add(key, value, unit=..., period=..., source_text=..., source_id=..., start=..., end=...)
retains source hashes, a short exact span and conflicting facts; .report() exposes them. A quote does
not itself verify the interpretation. Offsets are Unicode characters, quote length at most 500.
availability_filter(records, as_of=..., available_field='available_at', event_field='event_at',
revision_field='revision_at') and asof_vintages(records, as_of=..., key_fields=[...],
available_field='available_at', revision_field='revision_at') require timezone-aware timestamps.
Availability and revision must be known by as_of; event time alone is insufficient. Explicitly use
revision_field=None only for an unrevised stream. Do not add other helper reports to deliverables
unless TASK asks.
'''


PLAN_CHECK = ('Use one existing D line for validation: define its quantity/unit/method/time meaning from TASK, then '
              'select one executable Checks check grounded in TASK or '
              'the computation; give its quantity, check call and reference/tolerance basis, or explain why no '
              'applicable check is justified. Keep the existing line and word limits; do not invent a reference.')


def enabled():
    return os.environ.get('T1_RUNTIME') == '1'


def requirement_prompt():
    if os.environ.get('T1_REQUIREMENTS') != '1':
        return ''
    return '''\nREQUIREMENT COVERAGE (experimental):
from qfruntime import load_requirement_plan, RequirementRun
p = load_requirement_plan() reads the complete current instruction; inspect p.source and p.blocks
(id/start/end/quote). Interpret current TASK; do not treat each block as an inferred requirement.
p.add(id, source_ids=[block_ids], inputs=[input_labels], method=explicit_method,
      stages=[stage_names], checks=[Checks_record_ids], artifacts=[contracts])
Use p.context([block_ids],reason=...) only for actual context. Unassigned blocks remain unverified.
Contracts: {"path":"relative.json","format":"json","schema":{"type":"object",
"required":["value"],"properties":{"value":{"type":"number"}}}};
array schema uses items (every element checked); CSV uses format="csv", columns=[...],
types={column:"number"|"integer"|"boolean"|"string"}, optional rows/ordered.
Freeze the plan before execution: r=RequirementRun(p); result=r.stage(name,compute_callback).
Compute TASK outputs; execute the declared Checks validations; r.finish(c,output_root); c.finish().
Do not put diagnostic receipts in output_root. Covered means declared steps/checks/schema only,
not official acceptance. Missing steps/files fail; absent definitions or skipped checks stay unknown.
Do not mark unresolved requirements as context merely to complete coverage. No added model request
is required by this protocol; use the existing generation budget. Ordinary task requirements prevail.
'''


def available():
    try:
        return importlib.util.find_spec('qfruntime') is not None
    except (ImportError, ValueError):
        return False


def allowed(instruction):
    text = re.sub(r'[*`]', '', instruction)
    pattern = (r'(?im)^\s*(?:[-+]\s*)?(?:(?:you\s+)?must\s+)?(?:use|using)\s+'
               r'(?:only\s+(?:the\s+)?(?:standard\s+(?:Python\s+)?librar(?:y|ies)|'
               r'(?:Python\s+)?standard\s+library)|'
               r'(?:the\s+)?(?:Python\s+)?standard\s+library\s+only)\b')
    return re.search(pattern, text) is None


class Session:
    def __init__(self, instruction=None):
        self.temp = tempfile.TemporaryDirectory(prefix='t1-runtime-', dir='/tmp', ignore_cleanup_errors=True)
        self.root = Path(self.temp.name).resolve()
        self.namespace = uuid.uuid4().hex
        self.spec_path = None
        self.requirement_path = None
        self.semantic = {'semantic_state':'not_supplied'}
        if instruction is not None:
            try:
                from qfruntime.requirements import RequirementPlan
                plan = RequirementPlan(instruction)
                path = self.root/'requirement-source.json'
                path.write_text(json.dumps({'source':plan.source,'source_id':plan.source_id,
                                            'sha256':plan.source_sha256},ensure_ascii=False))
                path.chmod(0o400)
                self.requirement_path = path
            except (ImportError, ValueError, TypeError, OSError, OverflowError):
                self.requirement_path = None
            try:
                from qfruntime.semantics import extract_task_spec
                spec = extract_task_spec(instruction)
                spec_path = self.root/'task-spec.json'
                spec_path.write_text(json.dumps({'spec':spec.to_dict(),
                    'sources':{'instruction':instruction}},ensure_ascii=False))
                spec_path.chmod(0o400)
                self.spec_path = spec_path
                self.semantic = {'semantic_state':'partial' if spec.unresolved else 'explicit',
                    'semantic_quantities':len(spec.quantities),'semantic_unresolved':len(spec.unresolved),
                    'semantic_sha256':spec.fingerprint}
                coverage = spec.unresolved_coverage
                self.semantic.update(
                    semantic_source_index_state='partial' if coverage is not None else 'complete',
                    semantic_unresolved_total=coverage['total'] if coverage is not None else len(spec.unresolved),
                    semantic_unresolved_omitted=coverage['omitted'] if coverage is not None else 0)
            except (ImportError, ValueError, TypeError, OSError, OverflowError):
                self.spec_path = None
                self.semantic = {'semantic_state':'unresolved'}
        self.token = _ACTIVE.set(self)

    def close(self):
        _ACTIVE.reset(self.token)
        self.temp.cleanup()


def start(instruction=None):
    return Session(instruction)


@contextmanager
def session(instruction=None):
    active = start(instruction)
    try:
        yield active
    finally:
        active.close()


def active():
    return _ACTIVE.get() is not None


def semantic_context():
    current = _ACTIVE.get()
    return dict(current.semantic) if current is not None else {}


def worker_fields(report_path, budget):
    current = _ACTIVE.get()
    if current is None:
        return {}
    fields = {'T1_RUNTIME_REPORT': str(report_path), 'T1_RUNTIME_CACHE': str(current.root),
            'T1_RUNTIME_NAMESPACE': current.namespace, 'T1_RUNTIME_BUDGET_SEC': str(float(budget)),
            'T1_RUNTIME_USAGE': str(Path(report_path).with_name('runtime-usage.json'))}
    if current.spec_path is not None:
        fields['T1_TASK_SPEC'] = str(current.spec_path)
    if current.requirement_path is not None:
        fields['T1_REQUIREMENT_SOURCE'] = str(current.requirement_path)
    return fields


def read_usage(path):
    """Usage is bounded, untrusted call evidence; never an acceptance signal."""
    try:
        path = Path(path)
        if path.resolve().parent != path.absolute().parent:
            return {'runtime_usage_state':'invalid'}
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4096:
                return {'runtime_usage_state':'invalid'}
            raw = stream.read(4097)
        data = json.loads(raw,parse_constant=_reject,object_pairs_hook=_pairs)
        if (type(data) is not dict or set(data) != {'schema_version','calls'} or
                type(data['schema_version']) is not int or data['schema_version'] != 1 or
                type(data['calls']) is not dict or set(data['calls'])-set(USAGE_KINDS) or
                any(type(v) is not int or not 0 <= v <= 1000000 for v in data['calls'].values())):
            raise ValueError('invalid usage')
        return {'runtime_usage_state':'valid', **{'runtime_use_'+k:data['calls'].get(k,0) for k in USAGE_KINDS}}
    except FileNotFoundError:
        return {'runtime_usage_state':'missing'}
    except (OSError,ValueError,TypeError,RecursionError):
        return {'runtime_usage_state':'invalid'}


def _reject(*args):
    raise ValueError('invalid report')


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            _reject()
        out[key] = value
    return out


def _bounded(value, depth=0):
    if depth > 8:
        _reject()
    if value is None or type(value) is bool:
        return
    if type(value) in (int, float):
        if not math.isfinite(value):
            _reject()
    elif type(value) is str:
        if len(value) > 512:
            _reject()
    elif type(value) in (list, dict):
        if len(value) > 128:
            _reject()
        if type(value) is dict:
            for key in value:
                _bounded(key, depth + 1)
            value = value.values()
        for child in value:
            _bounded(child, depth + 1)
    else:
        _reject()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_report(path, canaries=()):
    """Read only a bounded plain file; export enums, finite numbers and hashes."""
    result = {'runtime_state': 'invalid', 'runtime_checks': [],
              'requirement_state':'unverified', 'requirement_checks':0,
              **{'runtime_' + s: 0 for s in STATUSES}}
    try:
        path = Path(os.path.abspath(path))
        directory_fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parts[1:-1]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                os.close(directory_fd)
                directory_fd = next_fd
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_REPORT_BYTES:
                _reject()
            raw = stream.read(MAX_REPORT_BYTES + 1)
        if len(raw) > MAX_REPORT_BYTES:
            _reject()
        text = raw.decode('utf-8')
        if any(str(value).lower() in text.lower() for value in canaries if value):
            _reject()
        data = json.loads(text, parse_constant=_reject, object_pairs_hook=_pairs)
        _bounded(data)
        if (type(data) is not dict or set(data) != {'schema_version', 'checks'}
                or type(data['schema_version']) is not int or data['schema_version'] != 1
                or type(data['checks']) is not list or len(data['checks']) > 128):
            _reject()
        rows, ids, coverage = [], set(), []
        for row in data['checks']:
            if type(row) is not dict or set(row) != {
                    'id', 'kind', 'status', 'parameters', 'difference', 'tolerance', 'evidence_type'}:
                _reject()
            if (type(row['id']) is not str or not row['id'] or len(row['id']) > 96
                    or row['id'] in ids or row['kind'] not in KINDS or row['status'] not in STATUSES
                    or type(row['parameters']) is not dict or type(row['evidence_type']) is not str):
                _reject()
            for key in ('difference', 'tolerance'):
                value = row[key]
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                    _reject()
            if row['kind'] in ('overflow', 'duplicate', 'invalid_id', 'report_write') and row['status'] != 'failed':
                _reject()
            if row['kind'] == 'skip' and row['status'] not in ('skipped', 'failed', 'unverified'):
                _reject()
            if row['kind'] == 'coverage' and row['evidence_type'] != 'requirement_coverage':
                _reject()
            if row['kind'] in ('coverage','skip') and row['evidence_type'] == 'requirement_coverage':
                coverage.append(row)
            if row['kind'] in ('close', 'balance', 'bounds', 'convergence', 'estimate') and row['status'] == 'passed':
                if row['difference'] is None or row['tolerance'] is None or row['difference'] > row['tolerance']:
                    _reject()
            ids.add(row['id'])
            rows.append({'id': _hash(row['id']), 'kind': row['kind'], 'status': row['status'],
                         'spec': _hash([row['kind'], row['parameters'], row['evidence_type']]),
                         'difference': row['difference'], 'tolerance': row['tolerance']})
        result.update(runtime_state='valid' if rows else 'empty', runtime_checks=rows)
        requirements = [row for row in coverage if row['parameters'].get('phase')=='requirement']
        summaries = [row for row in coverage if row['parameters'].get('phase')=='summary']
        sealed = False
        if len(summaries) == 1 and coverage[-1] is summaries[0]:
            summary = summaries[0]['parameters']
            expected = summary.get('expected_requirements')
            count = summary.get('expected_records')
            plan_hash = summary.get('plan_fingerprint')
            sealed = (type(expected) is int and expected == len(requirements) and expected > 0
                      and type(count) is int and count == len(coverage)-1
                      and type(plan_hash) is str and re.fullmatch(r'[0-9a-f]{64}',plan_hash) is not None
                      and all(row['parameters'].get('plan_fingerprint')==plan_hash for row in coverage)
                      and all(row['parameters'].get('phase') in ('requirement','constraint','summary')
                              for row in coverage))
        result.update(requirement_checks=len(requirements), requirement_state=(
            'missing' if not coverage else 'incomplete' if any(row['status']=='failed' for row in rows)
            else 'covered' if sealed and all(row['status']=='passed' for row in coverage) else 'unverified'))
        for row in rows:
            result['runtime_' + row['status']] += 1
    except FileNotFoundError:
        result['runtime_state'] = 'missing'
        result['requirement_state'] = 'missing'
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        pass
    return result


def flags(status):
    return {('runtime', row['id']): ('selfval',
            f"Runtime check #{index} ({row['kind']}) failed: difference={row['difference']}, "
            f"tolerance={row['tolerance']}. Keep its identity, specification and tolerance; fix the computation.")
            for index, row in enumerate(status.get('runtime_checks', []), 1) if row['status'] == 'failed'}


def regression(before, after):
    """Conservative repair guard; self-reported checks still cannot certify truth."""
    old = before.get('runtime_checks', [])
    if not old:
        return ''
    if after.get('runtime_state') != 'valid':
        return 'the runtime report disappeared or became invalid'
    current = {row['id']: row for row in after['runtime_checks']}
    for row in old:
        new = current.get(row['id'])
        if new is None or new['spec'] != row['spec']:
            return 'a runtime check was removed or its specification changed'
        if row['status'] == 'passed' and new['status'] != 'passed':
            return 'a previously passing runtime check regressed'
        if row['status'] == 'failed':
            if new['status'] in ('skipped', 'unverified'):
                return 'a failed runtime check was skipped'
            if row['tolerance'] is not None and (new['tolerance'] is None or new['tolerance'] > row['tolerance']):
                return 'a failed runtime check tolerance increased'
    if any(row['status'] == 'failed' and row['id'] not in {r['id'] for r in old} for row in current.values()):
        return 'a new runtime check failed'
    return ''
