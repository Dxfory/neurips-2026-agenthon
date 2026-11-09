"""Validate one closed, current-input normalization plan and execute its fixed op.

Callers bind the exact source hash and input schemas outside the model response.
Source ID membership establishes provenance identity, not correct interpretation.
This module does no file access, dynamic dispatch, unit conversion or rounding.
Importing it and parsing legacy numeric plans require only the standard library.
The separate literal panel codec/adapters lazily use pandas, NumPy and qfkit.
"""
import json


_MAX_BYTES = 8192
_KEYS = ('schema', 'op', 'input', 'date_col', 'value_cols', 'numeric_errors',
         'missing', 'refs', 'bind')


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value


def _names(values, label, *, containers=(list,)):
    if type(values) not in containers or not values:
        raise ValueError(f'{label} must be a nonempty collection')
    names = [_text(value, label) for value in values]
    if len(set(names)) != len(names):
        raise ValueError(f'{label} must be distinct')
    return names


def _basename(value):
    value = _text(value, 'input')
    if (value != value.strip() or value.startswith('.') or not value.endswith('.csv')
            or not value[:-4].strip() or any(char in value for char in '/\\:')
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError('input must be a plain CSV basename without paths')
    try:
        if len(value.encode('utf-8')) > 255:
            raise ValueError('input basename too long')
    except UnicodeError as error:
        raise ValueError('input must be valid UTF-8') from error
    return value


def _bounded(text):
    if type(text) is not str:
        raise ValueError('plan must be JSON text')
    try:
        if not text.strip() or len(text.encode('utf-8')) > _MAX_BYTES:
            raise ValueError('plan must contain at most 8192 UTF-8 bytes')
    except UnicodeError as error:
        raise ValueError('plan must be valid UTF-8') from error


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _number(value):
    # There are no numeric fields in this schema, including finite numbers.
    raise ValueError('numeric literals are not allowed in this plan schema')


def _validate(plan):
    if type(plan) is not dict or set(plan) != set(_KEYS):
        raise ValueError('plan must have exactly the numeric-input-plan-v1 fields')
    for key, expected in (('schema', 'numeric-input-plan-v1'),
                          ('op', 'numeric_time_frame'), ('bind', 'normalized_input')):
        if type(plan[key]) is not str or plan[key] != expected:
            raise ValueError(f'unsupported {key}')
    _basename(plan['input'])
    date_col = _text(plan['date_col'], 'date_col')
    values = _names(plan['value_cols'], 'value_cols')
    if date_col in values:
        raise ValueError('date_col must not be a value column')
    for key, choices in (('numeric_errors', ('raise', 'coerce')),
                         ('missing', ('raise', 'drop'))):
        if type(plan[key]) is not str or plan[key] not in choices:
            raise ValueError(f'unsupported {key}')
    refs = _names(plan['refs'], 'refs')
    result = {key: plan[key] for key in _KEYS}
    result['value_cols'] = values
    result['refs'] = refs
    _bounded(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
    return result


def _check_columns(plan, columns):
    names = _names(columns, 'input columns', containers=(list, tuple))
    if plan['date_col'] not in names or any(name not in names for name in plan['value_cols']):
        raise ValueError('date and value columns must exist in the selected input')


def parse_plan(raw_text, *, columns_by_input, source_ids):
    """Parse a bounded JSON plan and validate its columns and source ID membership.

    ``columns_by_input`` maps allowed CSV basenames to unique string column names.
    ``source_ids`` comes from the caller's verified RequirementPlan blocks. Neither
    membership nor this closed schema proves the selected parameters match the text.
    Returned dictionaries/lists contain only JSON data; no code is evaluated.
    """
    _bounded(raw_text)
    try:
        raw = json.loads(raw_text, object_pairs_hook=_object, parse_int=_number,
                         parse_float=_number, parse_constant=_number)
    except (ValueError, RecursionError) as error:
        raise ValueError('invalid numeric input plan JSON') from error
    plan = _validate(raw)
    if type(columns_by_input) is not dict or not columns_by_input:
        raise ValueError('columns_by_input must be a nonempty input allowlist')
    for name, columns in columns_by_input.items():
        _basename(name)
        _names(columns, 'input columns', containers=(list, tuple))
    if plan['input'] not in columns_by_input:
        raise ValueError('unknown input')
    _check_columns(plan, columns_by_input[plan['input']])
    allowed_refs = _names(source_ids, 'source_ids', containers=(list, tuple, set, frozenset))
    if not set(plan['refs']).issubset(allowed_refs):
        raise ValueError('unknown source ref')
    return plan


def normalize_frame(frame, plan):
    """Execute the fixed numeric adapter, preserving its result and row provenance.

    Revalidate the plan schema and actual frame columns on every call. The caller
    is responsible for binding this frame to the parsed input name and for retaining
    the parse-time source allowlist/hash; this signature cannot recheck that source.
    Positions are zero-based original row positions in output order, never index
    labels. Dropped rows may change the observation calendar. No frame is mutated.
    """
    plan = _validate(plan)
    import pandas as pd
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('frame must be a pandas DataFrame')
    _check_columns(plan, list(frame.columns))
    from qfkit.input_tools import numeric_time_frame
    result = numeric_time_frame(frame, date_col=plan['date_col'],
                                value_cols=plan['value_cols'],
                                numeric_errors=plan['numeric_errors'], missing=plan['missing'])
    # The adapter has already checked every source date and its uniqueness. Use
    # those dates to recover positions after its joint row filter and stable sort.
    source_dates = pd.DatetimeIndex([pd.Timestamp(value) for value in frame[plan['date_col']]])
    positions = source_dates.get_indexer(result.index).tolist()
    if any(position < 0 for position in positions):
        raise ValueError('normalized dates do not match original source rows')
    receipt = {
        'schema': 'numeric-input-receipt-v1',
        'input': plan['input'], 'bind': plan['bind'], 'refs': list(plan['refs']),
        'actual_operation': 'qfkit.input_tools.numeric_time_frame',
        'numeric_errors': plan['numeric_errors'], 'missing': plan['missing'],
        'rows_before': len(frame), 'rows_after': len(result),
        'rows_dropped': len(frame) - len(result),
        'retained_source_positions': positions,
        'output_schema': {
            'columns': [{'name': name, 'dtype': str(result[name].dtype)} for name in result.columns],
            'index': {'name': result.index.name, 'type': type(result.index).__name__,
                      'dtype': str(result.index.dtype), 'unique': bool(result.index.is_unique),
                      'ascending': bool(result.index.is_monotonic_increasing),
                      'timezone': None},
        },
        'unit_conversion_applied': False, 'rounding_applied': False,
        'semantic_verification': 'not_performed',
    }
    return result, receipt


# Independent literal panel contract. The legacy nine-field schema above stays closed.
PANEL_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_PANEL_MANIFEST_BYTES = 32768
_PANEL_KEYS = ('schema', 'op', 'input', 'time_col', 'entity_col', 'value_cols',
               'times', 'entities', 'numeric_errors', 'missing', 'bind', 'refs')
_PANEL_BINDING_KEYS = ('schema', 'plan', 'source_id', 'task_sha256', 'source_sha256',
                       'semantic_sha256', 'columns', 'adapter_version')
_PANEL_MANIFEST_KEYS = ('schema', 'binding', 'binding_sha256', 'rows_before', 'shape',
                        'axes', 'payload_bytes', 'payload_sha256',
                        'semantic_verification', 'source_dataframe_link')


def _panel_version():
    import qfkit
    if qfkit.__version__ != '0.6.2':
        raise ValueError('panel adapter version drift')


def _panel_dump(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(',', ':'))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('panel data must be finite JSON') from error


def _panel_bytes(text, limit):
    if type(text) is not str:
        raise ValueError('panel JSON must be text')
    try:
        raw = text.encode('utf-8')
    except UnicodeError as error:
        raise ValueError('panel text must be UTF-8') from error
    if not raw.strip() or len(raw) > limit:
        raise ValueError('panel text exceeds byte limit')
    return raw


def _panel_json(text, limit):
    _panel_bytes(text, limit)
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=_number)
    except (ValueError, RecursionError) as error:
        raise ValueError('invalid panel JSON') from error


def _panel_sha(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def _panel_digest(value):
    import re
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected a lowercase SHA256 digest')
    return value


def encode_panel_label(value):
    """Encode one literal label without merging string, integer or float identities.

    Timestamps retain nanoseconds, wall time and timezone name; unsupported or
    non-roundtrippable timezone objects are rejected. No expressions are evaluated.
    """
    _panel_version()
    from qfkit.input_tools import _label_key
    key = _label_key(value)
    kind = key[0]
    if kind == 'timestamp':
        token = {'kind': kind, 'value': value.isoformat(),
                 'timezone': None if value.tz is None else str(value.tz)}
        decode_panel_label(token)
    else:
        scalar = key[1]
        token = {'kind': kind, 'value': (str(scalar) if kind == 'integer' else
                                        scalar.hex() if kind == 'float' else scalar)}
    _panel_bytes(_panel_dump(token), _MAX_BYTES)
    return token


def decode_panel_label(token):
    """Decode the closed, canonical typed-literal label representation."""
    _panel_version()
    import math
    import re
    if type(token) is not dict or type(token.get('kind')) is not str:
        raise ValueError('invalid typed panel label')
    kind = token['kind']
    keys = {'kind', 'value', 'timezone'} if kind == 'timestamp' else {'kind', 'value'}
    if set(token) != keys or type(token['value']) is not str:
        raise ValueError('invalid typed panel label fields')
    _panel_bytes(_panel_dump(token), _MAX_BYTES)
    value = token['value']
    if kind == 'string':
        return value
    if kind == 'integer':
        if re.fullmatch('0|-?[1-9][0-9]*', value) is None:
            raise ValueError('noncanonical integer label')
        try:
            return int(value)
        except ValueError as error:
            raise ValueError('integer label too long') from error
    if kind == 'float':
        try:
            out = float.fromhex(value)
        except (ValueError, OverflowError) as error:
            raise ValueError('invalid float label') from error
        if not math.isfinite(out) or out.hex() != value:
            raise ValueError('noncanonical float label')
        return out
    if kind == 'timestamp':
        import pandas as pd
        zone = token['timezone']
        if zone is not None and (type(zone) is not str or not zone):
            raise ValueError('invalid timestamp timezone')
        try:
            out = pd.Timestamp(value)
            if pd.isna(out) or (out.tz is None) != (zone is None):
                raise ValueError('timestamp timezone mismatch')
            if zone is not None:
                out = out.tz_convert(zone)
            if out.isoformat() != value or (None if out.tz is None else str(out.tz)) != zone:
                raise ValueError('noncanonical timestamp label')
        except Exception as error:
            raise ValueError('invalid or unsupported timestamp label') from error
        return out
    raise ValueError('unsupported panel label kind')


def _panel_axis(tokens):
    from qfkit.input_tools import _label_key
    if type(tokens) is not list or not tokens:
        raise ValueError('axis must be a nonempty list of typed literals')
    values = tuple(decode_panel_label(token) for token in tokens)
    keys = [_label_key(value) for value in values]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate typed axis member')
    return values


def _panel_plan(plan):
    from .semantics import SourceRef
    if type(plan) is not dict or set(plan) != set(_PANEL_KEYS):
        raise ValueError('expected exactly the panel-input-plan-v1 fields')
    # Bound and detach the complete nested plan before any adapter allocation.
    plan = _panel_json(_panel_dump(plan), _MAX_BYTES)
    for key, expected in (('schema', 'panel-input-plan-v1'),
                          ('op', 'labelled_panel'), ('bind', 'panel_input')):
        if plan[key] != expected:
            raise ValueError('unsupported panel ' + key)
    _basename(plan['input'])
    time = _text(plan['time_col'], 'time_col')
    entity = _text(plan['entity_col'], 'entity_col')
    values = _names(plan['value_cols'], 'value_cols')
    if time == entity or time in values or entity in values:
        raise ValueError('label columns must be distinct and outside value_cols')
    for key, choices in (('numeric_errors', ('raise', 'coerce')), ('missing', ('raise', 'nan'))):
        if type(plan[key]) is not str or plan[key] not in choices:
            raise ValueError('unsupported panel ' + key)
    _panel_axis(plan['times'])
    _panel_axis(plan['entities'])
    refs = plan['refs']
    if type(refs) is not list or not refs:
        raise ValueError('panel refs must be nonempty source spans')
    seen = set()
    for ref in refs:
        if type(ref) is not dict or set(ref) != {'source_id', 'start', 'end', 'quote'}:
            raise ValueError('invalid panel source span fields')
        SourceRef(**ref)
        key = _panel_dump(ref)
        if key in seen:
            raise ValueError('duplicate source span')
        seen.add(key)
    return plan


def _panel_columns(plan, columns):
    names = _names(columns, 'panel input columns', containers=(list, tuple))
    required = [plan['time_col'], plan['entity_col']] + plan['value_cols']
    if any(name not in names for name in required):
        raise ValueError('panel columns not present in selected input')
    return names


def _panel_binding(binding):
    _panel_version()
    if type(binding) is not dict or set(binding) != set(_PANEL_BINDING_KEYS):
        raise ValueError('invalid panel binding fields')
    binding = _panel_json(_panel_dump(binding), _PANEL_MANIFEST_BYTES)
    if binding['schema'] != 'panel-binding-v1' or binding['adapter_version'] != '0.6.2':
        raise ValueError('unsupported panel binding version')
    plan = _panel_plan(binding['plan'])
    _text(binding['source_id'], 'source_id')
    if any(ref['source_id'] != binding['source_id'] for ref in plan['refs']):
        raise ValueError('source span identity mismatch')
    for field in ('task_sha256', 'source_sha256', 'semantic_sha256'):
        _panel_digest(binding[field])
    _panel_columns(plan, binding['columns'])
    return binding


def parse_panel_plan(raw_text, *, columns_by_input, source_text, source_sha256,
                     semantic_sha256, source_id='instruction'):
    """Bind one CSV's literal axes/policies to caller-provided identities.

    Spans are checked against this exact TASK text. Source/semantic hashes are
    caller assertions, not evidence that a CSV was read or an interpretation is
    correct. The returned binding must be retained outside the artifact as the
    loader's expected binding. This function performs no file access.
    """
    _panel_version()
    from .semantics import SourceRef
    plan = _panel_plan(_panel_json(raw_text, _MAX_BYTES))
    source = _panel_bytes(source_text, 1048576)
    _text(source_id, 'source_id')
    _panel_digest(source_sha256)
    _panel_digest(semantic_sha256)
    if type(columns_by_input) is not dict or set(columns_by_input) != {plan['input']}:
        raise ValueError('panel context must describe exactly the selected CSV')
    columns = _panel_columns(plan, columns_by_input[plan['input']])
    for ref in plan['refs']:
        if ref['source_id'] != source_id:
            raise ValueError('unknown panel source ref')
        SourceRef(**ref).verify({source_id: source_text})
    return _panel_binding({'schema': 'panel-binding-v1', 'plan': plan,
                           'source_id': source_id, 'task_sha256': _panel_sha(source),
                           'source_sha256': source_sha256, 'semantic_sha256': semantic_sha256,
                           'columns': columns, 'adapter_version': '0.6.2'})


def _panel_layout(plan):
    shape = (len(plan['times']), len(plan['entities']), len(plan['value_cols']))
    cells = shape[0] * shape[1]
    values = cells * shape[2]
    size = values * 8 + cells * 8 + values
    if size > PANEL_MAX_PAYLOAD_BYTES:
        raise ValueError('panel payload exceeds byte limit')
    return shape, values, cells, size


def _panel_arrays(panel, plan, rows):
    import numpy as np
    from qfkit.input_tools import _label_key
    fields = {'values', 'time_labels', 'entity_labels', 'feature_labels',
              'source_positions', 'finite_mask'}
    if type(panel) is not dict or set(panel) != fields:
        raise ValueError('invalid panel result fields')
    shape, count, cells, size = _panel_layout(plan)
    if type(rows) is not int or not 0 <= rows <= cells:
        raise ValueError('invalid source row count')
    values, positions, mask = (panel[key] for key in ('values', 'source_positions', 'finite_mask'))
    if (type(values) is not np.ndarray or values.dtype != np.dtype('float64') or values.shape != shape
            or type(positions) is not np.ndarray or positions.dtype.kind != 'i' or positions.shape != shape[:2]
            or type(mask) is not np.ndarray or mask.dtype != np.dtype('bool') or mask.shape != shape):
        raise ValueError('invalid panel array shape or dtype')
    for field, tokens in (('time_labels', plan['times']), ('entity_labels', plan['entities'])):
        labels = panel[field]
        if type(labels) is not tuple or [_label_key(x) for x in labels] != [_label_key(x) for x in _panel_axis(tokens)]:
            raise ValueError('panel actual axis differs from plan')
    if type(panel['feature_labels']) is not tuple or panel['feature_labels'] != tuple(plan['value_cols']):
        raise ValueError('panel feature order differs from plan')
    if np.isinf(values).any() or not np.array_equal(mask, np.isfinite(values)):
        raise ValueError('invalid panel values or finite mask')
    if ((positions < -1) | (positions >= rows)).any():
        raise ValueError('invalid panel source positions')
    observed = positions[positions >= 0]
    if not np.array_equal(np.sort(observed), np.arange(rows)):
        raise ValueError('source positions must cover each supplied row exactly once')
    if not np.isnan(values[positions == -1]).all():
        raise ValueError('structural missing positions must contain only NaN')
    if plan['missing'] == 'raise' and ((positions == -1).any() or not mask.all()):
        raise ValueError('missing values violate the plan')
    return values, positions, mask


def panel_artifact(frame, binding):
    """Adapt a caller-supplied frame to a bounded manifest and fixed numeric bytes.

    This is not a CSV reader. Row positions refer to this frame, not file lines.
    CSV-to-frame linkage remains caller_asserted; semantics remain unverified.
    The manifest digest must be retained independently of this returned packet.
    """
    import numpy as np
    import pandas as pd
    from qfkit.input_tools import labelled_panel
    binding = _panel_binding(binding)
    plan = binding['plan']
    shape, count, cells, size = _panel_layout(plan)
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != binding['columns']:
        raise ValueError('actual frame column identity/order drift')
    if len(frame) > cells:
        raise ValueError('too many source rows for explicit panel axes')
    panel = labelled_panel(frame, time_col=plan['time_col'], entity_col=plan['entity_col'],
                           value_cols=plan['value_cols'], times=_panel_axis(plan['times']),
                           entities=_panel_axis(plan['entities']), numeric_errors=plan['numeric_errors'],
                           missing=plan['missing'])
    values, positions, mask = _panel_arrays(panel, plan, len(frame))
    payload = (values.astype('<f8', copy=False).tobytes(order='C') +
               positions.astype('<i8', copy=False).tobytes(order='C') +
               mask.astype('u1').tobytes(order='C'))
    manifest = {'schema': 'panel-artifact-v1', 'binding': binding,
                'binding_sha256': _panel_sha(_panel_dump(binding).encode('utf-8')),
                'rows_before': len(frame), 'shape': list(shape),
                'axes': {'time': plan['times'], 'entity': plan['entities'], 'feature': plan['value_cols']},
                'payload_bytes': size, 'payload_sha256': _panel_sha(payload),
                'semantic_verification': 'unverified', 'source_dataframe_link': 'caller_asserted'}
    text = _panel_dump(manifest)
    _panel_bytes(text, _PANEL_MANIFEST_BYTES)
    return text, payload


def load_panel_artifact(manifest_text, payload, *, expected_binding, expected_manifest_sha256):
    """Load only the exact packet independently expected by the caller.

    Fixed layout: little-endian C-order float64 values, int64 source positions,
    then uint8 mask bytes (0/1). No headers, object arrays, pickle, files or code.
    Hash identity and structural validity do not prove the numeric values came
    from a particular CSV, nor that downstream calculations use this result.
    """
    _panel_version()
    import numpy as np
    raw = _panel_bytes(manifest_text, _PANEL_MANIFEST_BYTES)
    if _panel_sha(raw) != _panel_digest(expected_manifest_sha256):
        raise ValueError('panel manifest identity drift')
    expected = _panel_binding(expected_binding)
    manifest = _panel_json(manifest_text, _PANEL_MANIFEST_BYTES)
    if type(manifest) is not dict or set(manifest) != set(_PANEL_MANIFEST_KEYS):
        raise ValueError('invalid panel manifest fields')
    if (manifest['schema'] != 'panel-artifact-v1'
            or manifest['semantic_verification'] != 'unverified'
            or manifest['source_dataframe_link'] != 'caller_asserted'):
        raise ValueError('invalid panel manifest claims')
    binding = _panel_binding(manifest['binding'])
    if binding != expected or manifest['binding_sha256'] != _panel_sha(_panel_dump(expected).encode('utf-8')):
        raise ValueError('panel binding identity drift')
    plan = binding['plan']
    shape, count, cells, size = _panel_layout(plan)
    if (type(manifest['shape']) is not list or len(manifest['shape']) != 3
            or any(type(n) is not int for n in manifest['shape']) or manifest['shape'] != list(shape)
            or manifest['axes'] != {'time': plan['times'], 'entity': plan['entities'], 'feature': plan['value_cols']}):
        raise ValueError('panel actual axes or shape drift')
    if (type(manifest['payload_bytes']) is not int or manifest['payload_bytes'] != size
            or type(payload) is not bytes or len(payload) != size
            or _panel_sha(payload) != _panel_digest(manifest['payload_sha256'])):
        raise ValueError('invalid panel payload identity or length')
    values = np.frombuffer(payload, dtype='<f8', count=count).reshape(shape).astype(np.float64, copy=True)
    positions = np.frombuffer(payload, dtype='<i8', count=cells, offset=count*8).reshape(shape[:2]).copy()
    mask_bytes = np.frombuffer(payload, dtype='u1', count=count, offset=count*8+cells*8).reshape(shape)
    if (mask_bytes > 1).any():
        raise ValueError('invalid boolean mask byte')
    panel = {'values': values, 'source_positions': positions, 'finite_mask': mask_bytes.astype(bool),
             'time_labels': _panel_axis(plan['times']), 'entity_labels': _panel_axis(plan['entities']),
             'feature_labels': tuple(plan['value_cols'])}
    _panel_arrays(panel, plan, manifest['rows_before'])
    return panel, manifest


def require_same_panel_axes(left, right):
    """Require equal typed time/entity members in equal order, independently of shape.

    Feature sets and source row positions may differ across input files. This
    check is alignment only, not a claim about the meanings or units of values.
    """
    _panel_version()
    import numpy as np
    from qfkit.input_tools import _label_key
    keys = []
    for panel in (left, right):
        if (type(panel) is not dict or type(panel.get('values')) is not np.ndarray
                or panel['values'].ndim != 3):
            raise ValueError('expected a panel with a 3D values array')
        axes = []
        for index, field in enumerate(('time_labels', 'entity_labels')):
            labels = panel.get(field)
            if type(labels) is not tuple or len(labels) != panel['values'].shape[index] or not labels:
                raise ValueError('panel labels do not match dimensions')
            axis = tuple(_label_key(label) for label in labels)
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate panel axis member')
            axes.append(axis)
        keys.append(axes)
    if keys[0] != keys[1]:
        raise ValueError('panel time/entity axes do not align')
