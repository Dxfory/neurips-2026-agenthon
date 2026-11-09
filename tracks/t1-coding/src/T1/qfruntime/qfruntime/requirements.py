"""Source accounting and declared requirement coverage for the current solve.

Text blocks are not automatically interpreted as requirements. The caller must
map them or explicitly explain context; unassigned blocks remain unknown. A
covered declaration means its named steps ran, checks passed and artifacts met
their declared structure. It does not prove interpretation, algorithm choice,
check independence or that numerical checks describe the emitted bytes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .artifacts import validate_artifact
from .checks import Checks
from .semantics import SourceRef, TaskSpec

__all__ = ['RequirementPlan', 'RequirementRun', 'load_requirement_plan']
MAX_SOURCE_BYTES = 1024 * 1024


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _label(value):
    if type(value) is not str or not value.strip() or len(value) > 96:
        raise ValueError('invalid requirement label')
    return value


def _names(values):
    if not isinstance(values, (list, tuple)) or len(values) > 128:
        raise ValueError('invalid requirement bindings')
    names = tuple(_label(value) for value in values)
    if len(names) != len(set(names)):
        raise ValueError('duplicate requirement bindings')
    return names


class RequirementPlan:
    """A source-bound plan; mappings are supplied at runtime, never by task ID.

    ``blocks`` splits paragraphs into exact spans of at most 500 characters for
    SourceRef compatibility. All nonwhitespace source characters are retained.
    Context classification is a caller interpretation, not verified relevance.
    An empty input list explicitly means no external input (e.g. constants).
    Semantic definitions can be attached as a source-bound TaskSpec; names alone
    are never treated as numerical or semantic checks.
    """
    def __init__(self, source, *, source_id='instruction', semantics=None):
        if type(source) is not str or not source.strip() or len(source.encode()) > MAX_SOURCE_BYTES:
            raise ValueError('invalid or oversized requirement source')
        self._source, self.source_id = source, _label(source_id)
        self.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self._blocks = []
        spans, offset = [], 0
        for separator in re.finditer(r'\n[ \t]*\n', source):
            spans.append((offset, separator.start()))
            offset = separator.end()
        spans.append((offset, len(source)))
        for left, right in spans:
            while left < right and source[left].isspace():
                left += 1
            while right > left and source[right-1].isspace():
                right -= 1
            for start in range(left, right, 500):
                end = min(start + 500, right)
                if not source[start:end].strip():
                    continue
                ref = SourceRef(source_id, start, end, source[start:end])
                self._blocks.append({'id':f'source_{len(self._blocks)+1:04d}', **ref.to_dict()})
        if not self._blocks or len(self._blocks) > 4096:
            raise ValueError('requirement source block limit')
        self._requirements, self._context = {}, {}
        if semantics is not None:
            if not isinstance(semantics, TaskSpec):
                raise ValueError('invalid requirement semantics')
            semantics = TaskSpec.from_dict(semantics.to_dict(), sources={source_id:source})
        self._semantics = semantics

    @property
    def source(self):
        return self._source

    @property
    def blocks(self):
        return copy.deepcopy(self._blocks)

    @property
    def unassigned(self):
        assigned = set(self._context)
        for requirement in self._requirements.values():
            assigned.update(requirement['source_ids'])
        return tuple(b['id'] for b in self._blocks if b['id'] not in assigned)

    @property
    def fingerprint(self):
        return _hash({'source_sha256':self.source_sha256,'source_id':self.source_id,
                      'requirements':self._requirements,'context':self._context,
                      'semantics':self._semantics.fingerprint if self._semantics else None})

    def _source_ids(self, values):
        values = _names(values)
        if not values or set(values) - {b['id'] for b in self._blocks}:
            raise ValueError('unknown or missing source block')
        return values

    def context(self, source_ids, *, reason):
        """Explicitly label source blocks as context, with a bounded rationale."""
        ids, reason = self._source_ids(source_ids), _label(reason)
        used = {s for r in self._requirements.values() for s in r['source_ids']}
        if set(ids) & (used | set(self._context)):
            raise ValueError('source block already assigned')
        self._context.update({key:reason for key in ids})

    def add(self, requirement_id, *, source_ids, inputs, method, stages, checks,
            artifacts, quantities=()):
        """Bind one requirement to its input labels, steps, checks and artifacts.

        Inputs are declared provenance labels, not automatic data validation.
        Missing method/stage/check/artifact bindings stay unverified at finish.
        Multiple requirements may share a source paragraph and computed stage.
        """
        requirement_id = _label(requirement_id)
        if requirement_id in self._requirements or len(self._requirements) >= 96:
            raise ValueError('duplicate or excessive requirements')
        ids = self._source_ids(source_ids)
        if set(ids) & set(self._context):
            raise ValueError('source block classified as context')
        if not isinstance(artifacts, (list, tuple)) or len(artifacts) > 32:
            raise ValueError('invalid artifact bindings')
        raw = json.dumps(artifacts, allow_nan=False)
        if len(raw.encode()) > 65536 or any(type(a) is not dict for a in artifacts):
            raise ValueError('invalid or excessive artifact contracts')
        self._requirements[requirement_id] = {
            'source_ids':ids,'inputs':_names(inputs),'method':_label(method) if method is not None else None,
            'stages':_names(stages),'checks':_names(checks),'quantities':_names(quantities),
            'artifacts':json.loads(raw),
        }


class RequirementRun:
    """One in-process attempt; each declared stage can run at most once.

    Wrap actual computations (including Workflow.run) in stage(name, callback).
    A stage receipt proves callback completion only. Artifacts are inspected at
    finish, independently of the caller's claimed completion. Keep this object
    local to the run; it is not a cross-attempt cache or a trusted scorer.
    """
    def __init__(self, plan):
        if not isinstance(plan, RequirementPlan):
            raise TypeError('RequirementPlan required')
        self._plan, self._fingerprint = plan, plan.fingerprint
        self._requirements = copy.deepcopy(plan._requirements)
        self._stages, self._finished = {}, False

    def stage(self, name, compute):
        name = _label(name)
        expected = {s for r in self._requirements.values() for s in r['stages']}
        if self._finished or self._plan.fingerprint != self._fingerprint:
            raise ValueError('requirement plan changed or attempt finished')
        if name not in expected or name in self._stages or not callable(compute):
            raise ValueError('unknown, duplicate or invalid requirement stage')
        self._stages[name] = 'failed'
        value = compute()
        self._stages[name] = 'completed'
        return value

    def finish(self, checks, output_root):
        if not isinstance(checks, Checks):
            raise TypeError('Checks required')
        if self._finished or self._plan.fingerprint != self._fingerprint:
            raise ValueError('requirement plan changed or attempt finished')
        self._finished = True
        rows = {r['id']:r for r in checks.report()['checks']}
        results, artifacts = [], []
        for name, binding in self._requirements.items():
            reasons, failed, unknown = [], False, False
            if not all(binding[key] for key in ('method','stages','checks','artifacts')):
                reasons.append('incomplete_binding'); unknown = True
            for stage in binding['stages']:
                state = self._stages.get(stage)
                if state != 'completed':
                    reasons.append('stage_missing' if state is None else 'stage_failed'); failed = True
            for check in binding['checks']:
                row = rows.get(check)
                if row is None or row['status'] == 'failed':
                    reasons.append('check_missing' if row is None else 'check_failed'); failed = True
                elif row['status'] != 'passed' or row['evidence_type'] == 'requirement_coverage':
                    reasons.append('check_unverified'); unknown = True
            if binding['quantities']:
                semantic = self._plan._semantics
                defined = {q.name:q for q in semantic.quantities} if semantic else {}
                if set(binding['quantities']) - set(defined):
                    reasons.append('semantic_definition_missing'); unknown = True
                for quantity in binding['quantities']:
                    definition = defined.get(quantity)
                    if definition is None:
                        continue
                    if (any(getattr(definition, field) is None for field in
                            ('unit','shape','axes','method','time_basis','source')) or
                            not any(row['kind']=='contract' and row['status']=='passed' and
                                    row['parameters'].get('expected_spec')==definition.fingerprint
                                    for row in rows.values())):
                        reasons.append('semantic_check_missing'); unknown = True
            for contract in binding['artifacts']:
                result = validate_artifact(output_root, contract)
                artifacts.append({'requirement':name, **result})
                if result['status'] == 'failed':
                    reasons.append('artifact_failed'); failed = True
                elif result['status'] != 'passed':
                    reasons.append('artifact_unverified'); unknown = True
            state = 'incomplete' if failed else ('unverified' if unknown else 'covered')
            results.append({'id':name,'status':state,'reasons':sorted(set(reasons))})
            self._record(checks, name, state)
        unresolved = len(self._plan.unassigned)
        semantic_unknown = bool(self._plan._semantics and self._plan._semantics.unresolved)
        diagnostic_failure = any(row['status']=='failed' for row in rows.values())
        constraint_state = ('incomplete' if diagnostic_failure else 'unverified'
                            if unresolved or semantic_unknown or not results else 'covered')
        self._record(checks, 'global', constraint_state, phase='constraint')
        state = ('incomplete' if any(r['status']=='incomplete' for r in results) else
                 'unverified' if unresolved or semantic_unknown or not results or any(r['status']=='unverified' for r in results)
                 else 'covered')
        if diagnostic_failure or any(r['kind'] in ('overflow','duplicate','report_write','invalid_id') for r in checks.report()['checks']):
            state = 'incomplete'
        # Written last: an interrupted per-requirement loop is never a complete
        # coverage receipt. Counts are constant across valid repair attempts.
        self._record(checks, 'finish', state, phase='summary')
        return {'status':state,'plan_fingerprint':self._fingerprint,'source_sha256':self._plan.source_sha256,
                'unassigned_blocks':unresolved,'requirements':results,'artifacts':artifacts}

    def _record(self, checks, name, state, *, phase='requirement'):
        record_id = 'requirement.' + _hash([phase,name])
        parameters = {'plan_fingerprint':self._fingerprint,
                      'requirement_fingerprint':_hash(name),'phase':phase}
        if phase == 'summary':
            parameters.update(expected_requirements=len(self._requirements),
                              expected_records=len(self._requirements)+1)
        checks._run(record_id, 'skip' if state == 'unverified' else 'coverage',
                    parameters,
                    lambda:(state == 'covered', None, None), 'requirement_coverage')


def load_requirement_plan(path=None):
    """Read a bounded controller context even if narrow semantic extraction failed."""
    selected = path if path is not None else os.environ.get('T1_REQUIREMENT_SOURCE')
    if not selected:
        raise ValueError('no requirement source context')
    selected = Path(os.path.abspath(selected))
    fd = os.open(selected.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in selected.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        source = os.open(selected.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(source, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 6*MAX_SOURCE_BYTES:
            raise ValueError('invalid requirement context file')
        raw = stream.read(6*MAX_SOURCE_BYTES+1)
    if len(raw) > 6*MAX_SOURCE_BYTES:
        raise ValueError('requirement context too large')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate requirement context key')
            result[key] = value
        return result
    payload = json.loads(raw, object_pairs_hook=pairs)
    if type(payload) is not dict or set(payload) != {'source','source_id','sha256'}:
        raise ValueError('invalid requirement source context')
    plan = RequirementPlan(payload['source'], source_id=payload['source_id'])
    if plan.source_sha256 != payload['sha256']:
        raise ValueError('requirement source hash mismatch')
    return plan
