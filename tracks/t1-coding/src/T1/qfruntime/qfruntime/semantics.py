"""Shared, explicit task meanings. Source attachment is not semantic entailment.

The small extractor recognises explicit declarations, not arbitrary prose. Every
other nonempty line remains unresolved; its bounded index explicitly counts omitted
entries. No unit/method/time convention is guessed.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
import re
import stat
from types import MappingProxyType
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = ['SourceRef','QuantitySpec','TaskSpec','canonical_unit','convert_unit',
           'extract_task_spec','load_task_spec']

_UNITS = {
    'fraction': 'fraction', 'decimal': 'fraction', 'ratio': 'fraction',
    'percent': 'percent', 'percentage': 'percent', '%': 'percent',
    'basis point': 'basis_point', 'basis points': 'basis_point',
    'basis_point': 'basis_point', 'bps': 'basis_point', 'bp': 'basis_point',
    'percentage point': 'percentage_point', 'percentage points': 'percentage_point',
    'percentage_point': 'percentage_point',
}
_SCALES = {'fraction': 1., 'percent': .01, 'basis_point': .0001}
_TIME_BASES = {'static', 'event', 'available', 'revision', 'decision', 'period_end'}
_MAX_TEXT = 1048576


def _label(value, name, maximum=128):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(name + ' must be a bounded nonempty string')
    if any(ord(c) < 32 for c in value):
        raise ValueError(name + ' contains control characters')
    return value.strip()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def canonical_unit(unit):
    """Normalise known spelling aliases only; unknown units remain exact labels."""
    unit = _label(unit, 'unit')
    return _UNITS.get(unit.lower(), unit)


def convert_unit(values, from_unit, to_unit):
    """Explicit ratio/percent/bps conversion; never infer currency or rate bases.

    Percentage-point differences remain distinct from percent levels; converting
    between these meanings requires a caller-defined, separately justified step.
    """
    source, target = canonical_unit(from_unit), canonical_unit(to_unit)
    if source == target:
        factor = 1.
    elif source in _SCALES and target in _SCALES:
        factor = _SCALES[source] / _SCALES[target]
    else:
        raise ValueError('unsupported unit conversion')
    count = 0

    def convert(value, depth=0):
        nonlocal count
        if depth > 32:
            raise ValueError('too many dimensions')
        if isinstance(value, (list, tuple)):
            converted = [convert(v, depth+1) for v in value]
            shapes = [v[1] for v in converted]
            if shapes and any(s != shapes[0] for s in shapes):
                raise ValueError('ragged values')
            return [v[0] for v in converted], (len(value),)+(shapes[0] if shapes else ())
        count += 1
        if count > 1000000 or type(value) not in (int, float):
            raise ValueError('invalid or too many numeric values')
        try:
            number = float(value)
            if not math.isfinite(number) or (type(value) is int and int(number) != value):
                raise ValueError('nonfinite or lossy numeric value')
            out = number * factor
            if not math.isfinite(out):
                raise ValueError('conversion overflow')
            if number != 0 and out == 0:
                raise ValueError('conversion underflow')
        except OverflowError as exc:
            raise ValueError('conversion overflow') from exc
        return out, ()
    return convert(values)[0]


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    start: int
    end: int
    quote: str

    def __post_init__(self):
        _label(self.source_id, 'source_id')
        if (type(self.start) is not int or type(self.end) is not int or
                self.start < 0 or self.end <= self.start or
                not isinstance(self.quote, str) or not 0 < len(self.quote) <= 500 or
                len(self.quote) != self.end-self.start):
            raise ValueError('invalid source span')

    def verify(self, sources):
        text = sources.get(self.source_id)
        if not isinstance(text, str) or text[self.start:self.end] != self.quote:
            raise ValueError('source quote does not match')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class QuantitySpec:
    name: str
    unit: str | None = None
    shape: tuple | None = None
    axes: tuple | None = None
    method: str | None = None
    time_basis: str | None = None
    timezone: str | None = None
    atol: float | None = None
    rtol: float | None = None
    period: str | None = None
    source: SourceRef | None = None

    def __post_init__(self):
        _label(self.name, 'quantity name')
        if self.unit is not None:
            object.__setattr__(self, 'unit', canonical_unit(self.unit))
        for name in ('method', 'period'):
            if getattr(self, name) is not None:
                _label(getattr(self, name), name)
        if self.shape is not None:
            if (not isinstance(self.shape, (list, tuple)) or len(self.shape) > 16 or
                    any(x is not None and (type(x) is not int or x < 0) for x in self.shape)):
                raise ValueError('invalid shape')
            object.__setattr__(self, 'shape', tuple(self.shape))
        if self.axes is not None:
            if not isinstance(self.axes, (list, tuple)) or self.shape is None:
                raise ValueError('axes require an explicit shape')
            axes = tuple(_label(x, 'axis') for x in self.axes)
            if len(axes) != len(self.shape) or len(set(axes)) != len(axes):
                raise ValueError('axes must uniquely identify each dimension')
            object.__setattr__(self, 'axes', axes)
        if self.time_basis is not None and (type(self.time_basis) is not str or self.time_basis not in _TIME_BASES):
            raise ValueError('unsupported time basis')
        if self.timezone is not None:
            _label(self.timezone, 'timezone')
            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError('unknown timezone') from exc
        for name in ('atol', 'rtol'):
            value = getattr(self, name)
            if value is not None:
                try:
                    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                        raise ValueError('invalid tolerance')
                except OverflowError as exc:
                    raise ValueError('invalid tolerance') from exc
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise ValueError('source must be SourceRef')

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @property
    def fingerprint(self):
        return _hash(self.to_dict())


@dataclass(frozen=True)
class TaskSpec:
    quantities: tuple = ()
    unresolved: tuple = ()
    schema_version: int = 1
    source_hashes: tuple = ()
    unresolved_coverage: dict | None = None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported semantic schema')
        if not isinstance(self.quantities, (tuple, list)) or len(self.quantities) > 128:
            raise ValueError('invalid quantities')
        if any(not isinstance(q, QuantitySpec) for q in self.quantities):
            raise ValueError('quantities must be QuantitySpec')
        if len({q.name for q in self.quantities}) != len(self.quantities):
            raise ValueError('duplicate quantity definitions')
        if not isinstance(self.unresolved, (tuple, list)) or len(self.unresolved) > 128:
            raise ValueError('invalid unresolved requirements')
        for text in self.unresolved:
            _label(text, 'unresolved requirement', 512)
        coverage = self.unresolved_coverage
        if coverage is not None:
            if (not isinstance(coverage, (dict, MappingProxyType)) or
                    set(coverage) != {'indexed', 'total', 'omitted'} or
                    any(type(value) is not int or not 0 <= value <= _MAX_TEXT
                        for value in coverage.values()) or
                    coverage['indexed'] != len(self.unresolved) or not self.unresolved or
                    coverage['omitted'] < 1 or
                    coverage['total'] != coverage['indexed'] + coverage['omitted']):
                raise ValueError('invalid unresolved coverage')
            object.__setattr__(self, 'unresolved_coverage', MappingProxyType(dict(coverage)))
        if not isinstance(self.source_hashes, (tuple, list)) or len(self.source_hashes) > 128:
            raise ValueError('invalid source hashes')
        hashes = []
        for pair in self.source_hashes:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('invalid source hash pair')
            name, digest = pair
            _label(name, 'source_id')
            if not isinstance(digest, str) or re.fullmatch('[a-f0-9]{64}', digest) is None:
                raise ValueError('invalid source hash')
            hashes.append((name, digest))
        if len({x[0] for x in hashes}) != len(hashes):
            raise ValueError('duplicate source hashes')
        object.__setattr__(self, 'quantities', tuple(self.quantities))
        object.__setattr__(self, 'unresolved', tuple(self.unresolved))
        object.__setattr__(self, 'source_hashes', tuple(sorted(hashes)))

    def get(self, name):
        for quantity in self.quantities:
            if quantity.name == name:
                return quantity
        raise KeyError(name)

    def to_dict(self):
        result = {'schema_version': self.schema_version,
                  'quantities': [q.to_dict() for q in self.quantities],
                  'unresolved': list(self.unresolved),
                  'source_hashes': [list(x) for x in self.source_hashes]}
        # Preserve the existing payload and fingerprint when the whole index fits.
        if self.unresolved_coverage is not None:
            result['unresolved_coverage'] = dict(self.unresolved_coverage)
        return result

    @property
    def fingerprint(self):
        return _hash(self.to_dict())

    @classmethod
    def from_dict(cls, payload, *, sources=None):
        if (not isinstance(payload, dict) or set(payload)-{'schema_version','quantities',
                'unresolved','source_hashes','unresolved_coverage'} or not {'schema_version','quantities'} <= set(payload)):
            raise ValueError('invalid semantic payload')
        if not isinstance(payload['quantities'], list) or len(payload['quantities']) > 128:
            raise ValueError('invalid quantities')
        quantities = []
        allowed = {f.name for f in fields(QuantitySpec)}
        for record in payload['quantities']:
            if not isinstance(record, dict) or set(record)-allowed or 'name' not in record:
                raise ValueError('invalid quantity fields')
            record = dict(record)
            if record.get('source') is not None:
                ref = record['source']
                if not isinstance(ref, dict) or set(ref) != {'source_id','start','end','quote'}:
                    raise ValueError('invalid source fields')
                record['source'] = SourceRef(**ref)
                if sources is None:
                    raise ValueError('source text is required to verify source references')
                record['source'].verify(sources)
            quantities.append(QuantitySpec(**record))
        hashes = payload.get('source_hashes', ())
        if sources is not None:
            if not isinstance(sources, dict) or len(sources) > 128:
                raise ValueError('invalid sources')
            verified = {}
            for name, text in sources.items():
                _label(name, 'source_id')
                if not isinstance(text, str) or len(text.encode('utf-8')) > _MAX_TEXT:
                    raise ValueError('source too large')
                verified[name] = hashlib.sha256(text.encode('utf-8')).hexdigest()
            # Existing hashes cannot silently rebind to different source text.
            prior = cls((), source_hashes=hashes).source_hashes
            if any(verified.get(name) != digest for name, digest in prior):
                raise ValueError('source content hash changed')
            hashes = tuple(verified.items())
        elif hashes:
            raise ValueError('source text is required to verify source hashes')
        return cls(tuple(quantities), payload.get('unresolved', ()),
                   payload['schema_version'], hashes,
                   unresolved_coverage=payload.get('unresolved_coverage'))

    def merge(self, other):
        if not isinstance(other, TaskSpec):
            raise ValueError('can only merge task specs')
        if self.unresolved_coverage is not None or other.unresolved_coverage is not None:
            if self.fingerprint == other.fingerprint:
                return self
            raise ValueError('cannot merge incomplete unresolved source indexes')
        quantities = {q.name:q for q in self.quantities}
        for q in other.quantities:
            if q.name in quantities and quantities[q.name] != q:
                raise ValueError('conflicting quantity definition: '+q.name)
            quantities[q.name] = q
        hashes = dict(self.source_hashes)
        for name, digest in other.source_hashes:
            if name in hashes and hashes[name] != digest:
                raise ValueError('conflicting source versions')
            hashes[name] = digest
        return TaskSpec(tuple(quantities.values()),
                        tuple(dict.fromkeys(self.unresolved+other.unresolved)),
                        source_hashes=tuple(hashes.items()))


def extract_task_spec(text, *, source_id='instruction'):
    """Extract explicit quantity declarations and simple unit statements only.

    Supported grammar: ``quantity NAME: unit=percent; shape=[null,2];
    axes=["row","asset"]; method=...; time_basis=available; atol=...; rtol=...``.
    Also recognises ``NAME are/is expressed/measured/reported in UNIT.``.
    Unsupported prose remains unknown. ``unresolved`` indexes the first 128 lines;
    ``unresolved_coverage`` records indexed/total/omitted counts when more exist.
    These are source-index counts, not interpreted or completed requirements.
    The complete source hash is retained; the controller persists the source text.
    """
    if not isinstance(text, str) or len(text.encode('utf-8')) > _MAX_TEXT:
        raise ValueError('instruction too large')
    _label(source_id, 'source_id')
    quantities, unresolved, offset, unknown_count = [], [], 0, 0
    allowed = {f.name for f in fields(QuantitySpec)}-{'name','source'}
    for line in text.splitlines(keepends=True):
        content = line.strip()
        start = offset + len(line)-len(line.lstrip())
        offset += len(line)
        if not content:
            continue
        explicit = re.fullmatch(r'quantity\s+([^:]+):\s*(.+)', content, re.I)
        natural = re.fullmatch(r'([\w-]+)\s+(?:is|are)\s+(?:expressed|measured|reported)\s+in\s+([^.;]+)\.?', content, re.I)
        record = None
        if explicit:
            record = {'name': explicit.group(1).strip()}
            for part in explicit.group(2).split(';'):
                key, sep, value = part.strip().partition('=')
                if not sep or key not in allowed or key in record:
                    raise ValueError('invalid explicit quantity declaration')
                if key in ('shape','axes','atol','rtol'):
                    try:
                        value = json.loads(value, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite')))
                    except (ValueError, TypeError) as exc:
                        raise ValueError('invalid declaration value') from exc
                record[key] = value.strip() if isinstance(value,str) else value
        elif natural:
            record = {'name': natural.group(1), 'unit': natural.group(2).strip()}
        if record is None:
            unknown_count += 1
            if len(unresolved) < 128:
                unresolved.append('source line at '+str(start)+': '+content[:450])
            continue
        record['source'] = SourceRef(source_id,start,start+len(content),content)
        quantities.append(QuantitySpec(**record))
    coverage = ({'indexed': len(unresolved), 'total': unknown_count,
                 'omitted': unknown_count - len(unresolved)}
                if unknown_count > len(unresolved) else None)
    return TaskSpec(tuple(quantities), tuple(unresolved), source_hashes=(
        (source_id, hashlib.sha256(text.encode('utf-8')).hexdigest()),),
        unresolved_coverage=coverage)


def load_task_spec(path=None):
    """Load the controller-private, bounded context and verify all source bindings."""
    from pathlib import Path
    selected = path if path is not None else os.environ.get('T1_TASK_SPEC')
    if not selected:
        raise ValueError('no task semantic context provided')
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
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 2*_MAX_TEXT:
            raise ValueError('invalid semantic context file')
        raw = stream.read(2*_MAX_TEXT+1)
    if len(raw) > 2*_MAX_TEXT:
        raise ValueError('semantic context too large')
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result:
                raise ValueError('duplicate semantic key')
            result[key] = value
        return result
    payload = json.loads(raw, object_pairs_hook=pairs)
    if type(payload) is not dict or set(payload) != {'spec','sources'}:
        raise ValueError('invalid semantic context')
    return TaskSpec.from_dict(payload['spec'], sources=payload['sources'])
