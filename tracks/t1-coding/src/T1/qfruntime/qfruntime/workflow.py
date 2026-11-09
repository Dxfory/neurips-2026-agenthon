"""Deterministic, serial DAG scheduling over validated StageCache results.

Inputs are finite JSON; represent file identities with content hashes. Semantic
specifications are explicit JSON (or expose to_dict), never inferred. This does
not select task algorithms, relax validators, or turn partial work into a pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from .stages import StageCache, _json_bytes

__all__ = ['Stage', 'Workflow']


def _snapshot(value):
    return json.loads(_json_bytes(value))


def _bound_compute(compute, inputs, parameters, dependencies):
    # Capture only function and finite JSON so StageCache can fingerprint them.
    def bound():
        return compute(inputs, parameters, dependencies)
    return bound


@dataclass(frozen=True)
class Stage:
    """One declared computation, with mandatory validator and explicit version.

    compute(inputs, parameters, dependencies) returns finite JSON. Dependencies
    map names to detached upstream results. validate(result) must return literal
    True without mutation, including on each cache hit. Opaque callable state
    disables reuse conservatively under StageCache's fingerprint rules.

    inputs/parameters/semantic_spec are snapshotted at each run. Mutating their
    declared contents between runs invalidates dependent stages. semantic_spec
    may be a TaskSpec/QuantitySpec with to_dict(), or explicit finite JSON.
    """
    name: str
    compute: object
    validate: object
    version: str
    dependencies: tuple = ()
    inputs: object = None
    parameters: object = None
    semantic_spec: object = None

    def __post_init__(self):
        StageCache._entry_name(self.name)
        if not isinstance(self.version, str) or not self.version or len(self.version) > 4096:
            raise ValueError('version must be a nonempty string of at most 4096 characters')
        if not callable(self.compute) or not callable(self.validate):
            raise ValueError('compute and validate must be callable')
        if not isinstance(self.dependencies, (tuple, list)):
            raise ValueError('dependencies must be a list or tuple of stage names')
        for name in self.dependencies:
            StageCache._entry_name(name)
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError('dependencies must be unique')
        object.__setattr__(self, 'dependencies', tuple(self.dependencies))


class Workflow:
    """Validate a DAG before work, then run selected targets and all ancestors.

    Independent branches continue after a failure. Failed/blocked stages expose
    no result and block their descendants, even if a prior run cached success.
    Successful stages are revalidated through StageCache every run. Signature
    changes propagate through dependencies, while unrelated stages can reuse.
    status=completed means all requested computations validated, not task pass.
    Serial use only; shared cache directories need external coordination.
    """

    def __init__(self, stages, *, cache):
        if not isinstance(cache, StageCache):
            raise ValueError('cache must be StageCache')
        self.cache = cache
        self._stages = {}
        for stage in stages:
            if not isinstance(stage, Stage):
                raise ValueError('stages must contain Stage instances')
            if stage.name in self._stages:
                raise ValueError('stage names must be unique')
            self._stages[stage.name] = stage
        for stage in self._stages.values():
            if any(name not in self._stages for name in stage.dependencies):
                raise ValueError('stage has a missing dependency')
        pending = dict(self._stages)
        self._order = []
        visited = set()
        while pending:
            ready = [name for name, stage in pending.items() if all(dep in visited for dep in stage.dependencies)]
            if not ready:
                raise ValueError('stage dependencies contain a cycle')
            for name in ready:
                self._order.append(name)
                visited.add(name)
                del pending[name]

    def run(self, targets=None):
        if targets is None:
            selected = set(self._stages)
        else:
            if not isinstance(targets, (list, tuple)):
                raise ValueError('targets must be a list or tuple of stage names')
            for name in targets:
                StageCache._entry_name(name)
                if name not in self._stages:
                    raise ValueError('unknown workflow target')
            if len(set(targets)) != len(targets):
                raise ValueError('targets must be unique')
            selected, pending = set(), list(targets)
            while pending:
                name = pending.pop()
                if name not in selected:
                    selected.add(name)
                    pending.extend(self._stages[name].dependencies)
        order = [name for name in self._order if name in selected]
        results, states = {}, {}
        for name in order:
            stage = self._stages[name]
            blocked = [dep for dep in stage.dependencies if states[dep]['status'] != 'completed']
            if blocked:
                states[name] = {'status': 'blocked', 'blocked_by': blocked}
                continue
            try:
                inputs = _snapshot(stage.inputs)
                parameters = _snapshot(stage.parameters)
                semantic = stage.semantic_spec
                if callable(getattr(semantic, 'to_dict', None)):
                    semantic = semantic.to_dict()
                semantic = _snapshot(semantic)
                dependencies = _snapshot({dep: results[dep] for dep in stage.dependencies})
                result = self.cache.run(name, version=stage.version, inputs=inputs,
                                        parameters={'parameters': parameters, 'semantic_spec': semantic},
                                        dependencies=list(stage.dependencies),
                                        compute=_bound_compute(stage.compute, inputs, parameters, dependencies),
                                        validate=stage.validate)
                results[name] = result
                diagnostic = self.cache.diagnostics()['stages'][name]
                states[name] = {'status': 'completed', 'cache': diagnostic['status']}
            except Exception as exc:
                states[name] = {'status': 'failed', 'error_type': type(exc).__name__}
        return {'status': 'completed' if all(value['status'] == 'completed' for value in states.values()) else 'incomplete',
                'order': order, 'results': results, 'stages': states}
