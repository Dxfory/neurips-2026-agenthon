"""Explicit in-memory date, numeric-row, lagged-sample and labelled-panel adapters.

No file access, unit conversion, calendar resampling, estimation or task routing.
Policies come from the caller's specification, never from column names.
"""
from numbers import Number
from datetime import date as _date, timedelta as _timedelta
import re

import numpy as np
import pandas as pd

__all__ = ['numeric_time_frame', 'date_row_position', 'lagged_sample', 'labelled_panel']


def _timestamp(value):
    if isinstance(value, Number):
        raise ValueError('absolute date required; no numeric epoch or relative date inference')
    if isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[T ].+)?', value.strip()) is None:
        raise ValueError('date strings require complete ISO YYYY-MM-DD dates with optional time')
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid date') from error
    if pd.isna(stamp) or stamp.tzinfo is not None:
        raise ValueError('dates must be present and timezone-naive; convert timezones explicitly first')
    return stamp


def _date_index(values):
    try:
        index = pd.DatetimeIndex([_timestamp(value) for value in values])
    except (TypeError, OverflowError) as error:
        raise ValueError('1-D absolute dates required') from error
    if not index.is_unique:
        raise ValueError('duplicate dates; aggregate or resolve duplicates explicitly first')
    return index


def _missing_policy(missing):
    if missing not in ('raise', 'drop'):
        raise ValueError("missing must be 'raise' or 'drop'")


def _reject_temporal_values(values):
    array = np.asarray(values)
    if array.dtype.kind in 'Mm' or (array.dtype.kind == 'O' and any(
            isinstance(value, (_date, _timedelta, np.datetime64, np.timedelta64))
            for value in array.flat)):
        raise ValueError('numeric values cannot be dates or durations; convert units explicitly first')


def numeric_time_frame(frame, *, date_col, value_cols, numeric_errors, missing):
    """Build a sorted numeric frame on a unique date index with explicit row policy.

    Parameters
    ----------
    frame : pandas DataFrame; never modified in place.
    date_col : date column name. All dates must be valid, unique, timezone-naive
        complete ISO date strings or datetime objects, even on rows whose values are dropped.
        Numeric epochs, NaT and relative dates are rejected. Time of day is retained.
    value_cols : nonempty sequence of distinct numeric column names, excluding date_col.
        Only these columns appear in the result, in the requested order.
    numeric_errors : 'raise' or 'coerce', passed to pandas.to_numeric per value column.
        Coerce converts nonnumeric values to NaN. Date/duration value types always
        raise; no implicit epoch units, percent or currency parsing.
    missing : 'raise' or 'drop'. After conversion, test all selected columns jointly
        for finite values. Drop removes an entire row; raise rejects any such row.
        Dropping defines a NEW valid-observation row calendar. Use it only when the
        task calls for offsets on valid rows, never to silently bridge calendar gaps.

    Returns
    -------
    DataFrame : float columns, ascending DatetimeIndex named date_col. Dates identify
        source rows; original index labels and unselected columns are not retained.
        An empty result is allowed. No interpolation, fill, rescaling or daily flooring.

    Example
    -------
    >>> raw = pd.DataFrame({'day':['2024-01-02','2024-01-01'], 'x':['bad','7']})
    >>> numeric_time_frame(raw, date_col='day', value_cols=['x'], numeric_errors='coerce', missing='drop')['x'].tolist()
    [7.0]
    """
    _missing_policy(missing)
    if numeric_errors not in ('raise', 'coerce'):
        raise ValueError("numeric_errors must be 'raise' or 'coerce'")
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('DataFrame with unique columns required')
    if isinstance(value_cols, (str, bytes)):
        raise ValueError('value_cols must be a sequence of column names')
    columns = list(value_cols)
    if (not columns or len(set(columns)) != len(columns) or date_col in columns
            or date_col not in frame or any(column not in frame for column in columns)):
        raise ValueError('distinct existing date and value columns required')
    index = _date_index(frame[date_col])
    for column in columns:
        _reject_temporal_values(frame[column])
    converted = frame[columns].apply(pd.to_numeric, errors=numeric_errors)
    if any(pd.api.types.is_complex_dtype(dtype) for dtype in converted.dtypes):
        raise ValueError('real numeric values required')
    values = converted.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values).all(axis=1)
    if missing == 'raise' and not valid.all():
        raise ValueError('nonfinite numeric rows')
    result = pd.DataFrame(values[valid], index=index[valid], columns=columns)
    result.index.name = date_col
    return result.sort_index(kind='mergesort')


def date_row_position(index, date, *, direction):
    """Find an exact, next or previous date without mixing NumPy and pandas scalars.

    Parameters
    ----------
    index : nonempty ascending sequence of unique, timezone-naive absolute dates.
        No sorting or deduplication is performed. Intraday timestamps are preserved.
    date : timezone-naive complete ISO date string or datetime object, not a numeric epoch.
    direction : 'next' (first index >= date), 'previous' (last index <= date),
        or 'exact'. All three include equality. No calendar-day interpolation.

    Returns
    -------
    int : zero-based row position. Raises IndexError if no match in that direction;
        raises ValueError for an invalid calendar, date or policy. Never clips.

    Example
    -------
    >>> date_row_position(['2024-01-01','2024-01-03'], '2024-01-02', direction='next')
    1
    """
    if direction not in ('next', 'previous', 'exact'):
        raise ValueError("direction must be 'next', 'previous' or 'exact'")
    dates = _date_index(index)
    if not len(dates) or not dates.is_monotonic_increasing:
        raise ValueError('nonempty ascending unique date index required')
    stamp = _timestamp(date)
    position = int(dates.searchsorted(stamp, side='right' if direction == 'previous' else 'left'))
    if direction == 'previous':
        position -= 1
    if position < 0 or position >= len(dates) or (direction == 'exact' and dates[position] != stamp):
        raise IndexError('date has no matching row in the requested direction')
    return position


def lagged_sample(values, *, lags, start, stop, missing):
    """Pair response rows with explicit prior row positions before filtering missing data.

    Parameters
    ----------
    values : 1-D real numeric sequence on the caller's chosen row calendar. No sorting,
        differencing, unit conversion or pre-compression is performed. Pandas labels
        are ignored; all positions refer to this supplied sequence.
    lags : nonempty sequence of distinct positive integers, in output column order.
    start, stop : integer RESPONSE bounds, half-open [start, stop).
        Requires max(lags) <= start < stop <= len(values); no clipping or implicit
        extra loss of the first response row. A leading NaN from differencing can
        be retained in values while the caller selects valid sample bounds.
    missing : 'raise' or 'drop'. Form all (response, lagged predictors) first, then
        jointly reject/drop nonfinite pairs. Never pairs across a missing row by
        deleting it before shifting. Dropping may yield an empty sample.

    Returns
    -------
    dict : y shape (n,), X shape (n, len(lags)), positions shape (n,) with retained
        response positions. X[r,k] = values[positions[r]-lags[k]]. No intercept
        column; feed arrays to ols(y, X, add_const=True) if an intercept is required.

    Example
    -------
    >>> lagged_sample([2,4,8,16], lags=[1], start=1, stop=4, missing='raise')['X'].ravel().tolist()
    [2.0, 4.0, 8.0]
    """
    _missing_policy(missing)
    _reject_temporal_values(values)
    raw = np.asarray(values)
    if raw.ndim != 1 or np.iscomplexobj(raw):
        raise ValueError('1-D real numeric sequence required')
    values = np.asarray(values, dtype=float)
    lag_list = list(lags)
    def integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
    if (not lag_list or any(not integer(lag) or lag <= 0 for lag in lag_list)
            or len(set(lag_list)) != len(lag_list)):
        raise ValueError('distinct positive integer lags required')
    if not integer(start) or not integer(stop) or not max(lag_list) <= start < stop <= len(values):
        raise ValueError('response bounds must satisfy max(lags) <= start < stop <= len(values)')
    positions = np.arange(start, stop, dtype=int)
    y = values[positions]
    X = values[positions[:, None] - np.asarray(lag_list, dtype=int)[None, :]]
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if missing == 'raise' and not valid.all():
        raise ValueError('nonfinite response or lagged predictor')
    return {'y': y[valid], 'X': X[valid], 'positions': positions[valid]}


def _label_key(value):
    """A deliberately finite scalar domain with no string/date/numeric coercion."""
    if isinstance(value, (_date, _timedelta, np.datetime64, np.timedelta64)) and not isinstance(value, pd.Timestamp):
        raise ValueError('temporal axis labels require pandas Timestamp; durations are not supported')
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('boolean axis labels are not supported')
    if isinstance(value, str):
        return ('string', str(value))
    if isinstance(value, (int, np.integer)):
        return ('integer', int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or (isinstance(value, np.floating) and value.dtype.itemsize > 8):
            raise ValueError('finite float axis labels of at most 64 bits required')
        return ('float', float(value))
    if isinstance(value, pd.Timestamp) and not pd.isna(value):
        # Same instant in differently named zones is not the same declared label.
        # No timezone conversion, calendar flooring or string parsing occurs.
        return ('timestamp', value.isoformat(), str(value.tz))
    raise ValueError('axis labels require strings, integers, finite floats or pandas Timestamp')


def _ordered_labels(values):
    if (not isinstance(values, (list, tuple, range, np.ndarray, pd.Index, pd.Series))
            or isinstance(values, np.ndarray) and values.ndim != 1):
        raise ValueError('an explicit ordered one-dimensional label sequence is required')
    labels = tuple(values)
    if not labels:
        raise ValueError('axis labels cannot be empty')
    mapping = {}
    for position, label in enumerate(labels):
        key = _label_key(label)
        if key in mapping:
            raise ValueError('duplicate requested axis label')
        mapping[key] = position
    return labels, mapping


def _numeric_column(series, numeric_errors):
    original_missing = []
    for value in series:
        if isinstance(value, (bool, np.bool_, complex, np.complexfloating, _date, _timedelta,
                              np.datetime64, np.timedelta64)):
            raise ValueError('real numeric values required; no booleans, complex or temporal values')
        absent = pd.isna(value)
        if not isinstance(absent, (bool, np.bool_)):
            raise ValueError('scalar numeric values required')
        original_missing.append(bool(absent))
    converted = pd.to_numeric(series, errors=numeric_errors)
    if pd.api.types.is_complex_dtype(converted.dtype):
        raise ValueError('real numeric values required')
    numbers = converted.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numbers).any():
        raise ValueError('infinite values are not supported')
    # pandas may turn empty text into NaN even under errors='raise'; that is not
    # an observed missing scalar. Only the caller's coerce policy may accept it.
    if numeric_errors == 'raise' and (np.isnan(numbers) & ~np.asarray(original_missing, dtype=bool)).any():
        raise ValueError('invalid numeric value; select coerce explicitly to retain it as NaN')
    return numbers


def labelled_panel(frame, *, time_col, entity_col, value_cols, times, entities,
                   numeric_errors, missing):
    """Map long labelled observations to explicit time, entity and feature axes.

    Parameters
    ----------
    frame : pandas DataFrame; long observations with unique string column names.
        Never mutated. Axis values are the scalars already present in this frame;
        pandas construction may have coerced their types, which is not reversed.
    time_col, entity_col : distinct existing nonempty string column names.
    value_cols : ordered nonempty list, tuple, pandas Index or 1-D NumPy array.
        Distinct existing string column names, excluding both label columns.
    times, entities : explicit ordered nonempty label sequences.
        List, tuple, range, 1-D NumPy array, pandas Index or Series. Their order
        defines the axes; never sorted or inferred from the observations.
        Supported labels: strings, non-bool Python/NumPy integers, finite
        Python/NumPy floats up to 64 bits, pandas Timestamp. Integer 1, float
        1.0 and string '1' are distinct; scalars in one numeric family match by
        value. Timestamp identity includes local timestamp and timezone name;
        equal instants in different zones remain distinct. Bool, missing labels,
        compound labels, dates/datetimes outside Timestamp and durations raise.
    numeric_errors : 'raise' or 'coerce'; required.
        Numeric values convert to float64 with its ordinary precision limits;
        no units, rescaling or explicit rounding. Raise rejects invalid text
        including empty strings; coerce retains invalid text as NaN. Observed
        None/pd.NA/NaN may remain missing under either policy. Actual bool,
        complex, temporal and infinite numeric values always raise.
    missing : 'raise' or 'nan'; required.
        Raise rejects any NaN or absent grid row; nan preserves the entire
        requested grid. Coerce plus raise therefore rejects newly coerced NaN.

    Returns
    -------
    dict : values float64 (T,N,L), time_labels/entity_labels/feature_labels tuples,
        source_positions integer (T,N), finite_mask bool (T,N,L). Positions are
        zero-based row positions in THIS supplied frame, independent of its
        index labels. Only structurally absent rows have position -1; an observed
        or coerced NaN retains its actual row position. Arrays are independent.

    Notes
    -----
    Duplicate typed (time,entity) pairs and observations outside the requested
    axes raise. No aggregation, dropping, sorting, zero filling, implicit lag,
    date parsing, calendar normalization, financial estimation or file I/O occurs.
    Align multiple panels using identical explicit axes; define the sample
    universe, time shifts, units and estimator inputs before calling. Successful
    conversion alone does not prove those caller-supplied semantics are correct.

    Example
    -------
    >>> rows = pd.DataFrame({'day':['late','early'], 'id':['x','x'], 'v':[20.,10.]})
    >>> result = labelled_panel(rows, time_col='day', entity_col='id', value_cols=['v'],
    ...                         times=['early','late'], entities=['x'],
    ...                         numeric_errors='raise', missing='raise')
    >>> result['values'][:, :, 0].tolist()
    [[10.0], [20.0]]
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('a long DataFrame with unique columns is required')
    if numeric_errors not in ('raise', 'coerce') or missing not in ('raise', 'nan'):
        raise ValueError('numeric_errors is raise/coerce and missing is raise/nan')
    if not isinstance(value_cols, (list, tuple, np.ndarray, pd.Index)) or (
            isinstance(value_cols, np.ndarray) and value_cols.ndim != 1):
        raise ValueError('value_cols must be an ordered one-dimensional column sequence')
    columns = tuple(value_cols)
    selected = (time_col, entity_col) + columns
    if (not columns or any(not isinstance(col, str) or not col for col in selected)
            or len(set(selected)) != len(selected) or any(col not in frame.columns for col in selected)):
        raise ValueError('distinct existing string column names required')
    time_labels, time_map = _ordered_labels(times)
    entity_labels, entity_map = _ordered_labels(entities)
    positions = np.full((len(time_labels), len(entity_labels)), -1, dtype=np.intp)
    for position, (time, entity) in enumerate(zip(frame[time_col], frame[entity_col])):
        time_key, entity_key = _label_key(time), _label_key(entity)
        if time_key not in time_map or entity_key not in entity_map:
            raise ValueError('observed axis label is outside the requested axes')
        i, j = time_map[time_key], entity_map[entity_key]
        if positions[i, j] != -1:
            raise ValueError('duplicate observed time/entity pair')
        positions[i, j] = position
    numeric = np.column_stack([_numeric_column(frame[col], numeric_errors) for col in columns])
    values = np.full((*positions.shape, len(columns)), np.nan, dtype=float)
    present = positions >= 0
    values[present] = numeric[positions[present]]
    finite = np.isfinite(values)
    if missing == 'raise' and not finite.all():
        raise ValueError('missing grid rows or numeric values')
    return {'values': values, 'time_labels': time_labels, 'entity_labels': entity_labels,
            'feature_labels': columns, 'source_positions': positions, 'finite_mask': finite}
