"""Opt-in program launcher recording bounded helper calls, not correctness.

Only fixed capability counters are written. Importing a helper is not a call.
Counts are untrusted diagnostics; abrupt process termination can leave a lower
bound (the first call is flushed immediately). No arguments or code are retained.
"""
from contextlib import contextmanager
import functools
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

_TARGETS = {
    'checks': {'Checks.contract':'checks','Checks.close':'checks','Checks.balance':'checks',
               'Checks.bounds':'checks','Checks.convergence':'checks','Checks.estimate':'checks'},
    'semantics': {'load_task_spec':'semantics','extract_task_spec':'semantics','convert_unit':'semantics'},
    'contracts': {'enforce_contract':'contracts','enforce_task':'contracts'},
    'verification': {'verify':'verification'},
    'resources': {'measure_resources':'resources','run_batches':'resources'},
    'workflow': {'Workflow.run':'workflow'},
    'documents': {'read_document':'documents','extract_quantity':'documents'},
    'evidence': {'EvidenceLedger.add':'evidence','availability_filter':'evidence','asof_vintages':'evidence'},
    'stages': {'StageCache.run':'cache','probe':'resources'},
    'requirements': {'load_requirement_plan':'semantics','RequirementRun.stage':'workflow',
                     'RequirementRun.finish':'contracts'},
    'artifacts': {'validate_artifact':'contracts'},
}


@contextmanager
def record_usage(path):
    """Instrument public helper entry points during this process's program run."""
    calls = {name:0 for name in ('checks','semantics','contracts','verification','resources',
                                 'workflow','documents','evidence','cache')}
    originals = []
    path = Path(path).absolute()

    def publish():
        # A broken diagnostic sink must not change the program's exit semantics.
        try:
            if path.parent.resolve() != path.parent or path.is_symlink():
                return
            fd, temp = tempfile.mkstemp(prefix='.usage-', dir=path.parent)
            try:
                with os.fdopen(fd,'w') as stream:
                    json.dump({'schema_version':1,'calls':calls},stream,allow_nan=False)
                os.replace(temp,path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        except OSError:
            pass

    def wrap(function, capability):
        @functools.wraps(function)
        def counted(*args, **kwargs):
            calls[capability] = min(1000000,calls[capability]+1)
            if calls[capability] == 1:
                publish()
            return function(*args, **kwargs)
        return counted

    package = importlib.import_module('qfruntime')
    try:
        for module_name, targets in _TARGETS.items():
            module = importlib.import_module('qfruntime.'+module_name)
            for name, capability in targets.items():
                parts = name.split('.')
                owner = module if len(parts) == 1 else getattr(module,parts[0])
                original = getattr(owner,parts[-1])
                counted = wrap(original,capability)
                originals.append((owner,parts[-1],original))
                setattr(owner,parts[-1],counted)
                if len(parts) == 1 and getattr(package,name,None) is original:
                    originals.append((package,name,original)); setattr(package,name,counted)
        publish()
        yield
    finally:
        publish()
        for owner,name,original in reversed(originals):
            setattr(owner,name,original)


def main():
    if len(sys.argv) != 2 or not os.environ.get('T1_RUNTIME_USAGE'):
        raise ValueError('runner requires a program and private usage path')
    program = sys.argv[1]
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = [program]
        sys.path[0:1] = [str(Path(program).resolve().parent)]
        with record_usage(os.environ['T1_RUNTIME_USAGE']):
            runpy.run_path(program,run_name='__main__')
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


if __name__ == '__main__':
    main()
