"""Explicit calendar-string parsing and row filtering, without task or I/O policy."""
from datetime import datetime
import re

import numpy as np
import pandas as pd

__all__ = ['parse_calendar_dates', 'filter_rows']


def _formats(formats):
    if not isinstance(formats, (list, tuple)) or not formats:
        raise ValueError('formats must be a nonempty list or tuple of explicit calendar formats')
    for pattern in formats:
        if not isinstance(pattern, str):
            raise ValueError('each date format must be a string')
        directives = re.findall(r'%[YymbBd]', pattern)
        if ('%' in re.sub(r'%[YymbBd]', '', pattern)
                or sum(d in ('%Y', '%y') for d in directives) != 1
                or sum(d in ('%m', '%b', '%B') for d in directives) != 1
                or directives.count('%d') != 1):
            raise ValueError('formats require exactly one year, month and day; no time or timezone directives')
    return tuple(dict.fromkeys(formats))


def parse_calendar_dates(values, *, formats, missing='raise'):
    """Parse mixed calendar date strings without dropping, sorting or deduplicating.

    Parameters
    ----------
    values : one-dimensional pandas Series, Index, list, tuple or numpy array.
        Nonmissing items must be strings. Whitespace around a string is ignored;
        blank strings remain invalid, not missing. Numeric epochs, timestamps,
        time-of-day and timezone conversions are not inferred.
    formats : nonempty list/tuple of explicit strptime formats, each containing
        one year (%Y or %y), month (%m, %b or %B), and day (%d). Only calendar
        directives are supported. Month names follow the current locale; %y uses
        Python's 1969-2068 window. Choose formats from the input specification.
        Every matching format is considered: different resulting dates raise.
    missing : 'raise' (default) or 'preserve'. Preserve retains actual null values
        (None, NaN, pandas NA/NaT) as NaT. It never permits malformed strings.

    Returns
    -------
    (pandas.Series, dict) : datetime64[ns] dates and a count report. The report's keys:
        rows_input, rows_output, rows_parsed, rows_missing, rows_invalid, rows_ambiguous,
        missing_positions (source positions of preserved missing values), and missing,
        invalid, ambiguous (the same three counts; invalid and ambiguous are always 0 in a
        returned report, since an invalid or ambiguous date raises ValueError). Series inputs
        retain their index and name, including duplicate labels. Other inputs use
        positional indices. Dates outside pandas' nanosecond range are invalid.
        No input is mutated, filtered, sorted or assigned a calendar convention.

    Raises
    ------
    ValueError : bad policy/shape, malformed or ambiguous dates, or disallowed
        missing values. Data errors report separate missing/invalid/ambiguous
        counts plus up to five source positions per category, without echoing values.

    Example
    -------
    >>> dates, report = parse_calendar_dates(['2024-03-15', '03/16/2024'], formats=['%Y-%m-%d', '%m/%d/%Y'])
    >>> dates.dt.strftime('%Y-%m-%d').tolist()
    ['2024-03-15', '2024-03-16']
    """
    formats = _formats(formats)
    if missing not in ('raise', 'preserve'):
        raise ValueError("missing must be 'raise' or 'preserve'; parsing never drops rows")
    if (not isinstance(values, (pd.Series, pd.Index, list, tuple, np.ndarray))
            or isinstance(values, np.ndarray) and values.ndim != 1):
        raise ValueError('values must be a one-dimensional ordered sequence or Series')
    if any(isinstance(value, (list, tuple, dict, np.ndarray, pd.Series, pd.Index)) for value in values):
        raise ValueError('values must be one-dimensional scalar calendar strings')
    source = values.copy(deep=True) if isinstance(values, pd.Series) else pd.Series(values, dtype=object)
    parsed = []
    positions = {'missing': [], 'invalid': [], 'ambiguous': []}
    samples = {'missing': [], 'invalid': [], 'ambiguous': []}
    for position, value in enumerate(source):
        stamp, issue = pd.NaT, None
        if pd.isna(value):
            issue = 'missing'
        elif not isinstance(value, str):
            issue = 'invalid'
        else:
            candidates = set()
            for pattern in formats:
                try:
                    candidates.add(datetime.strptime(value.strip(), pattern).date())
                except (ValueError, OverflowError):
                    continue
            if not candidates:
                issue = 'invalid'
            elif len(candidates) > 1:
                issue = 'ambiguous'
            else:
                try:
                    stamp = pd.Timestamp(candidates.pop())
                    _ = stamp.value  # Enforce the return dtype's nanosecond range.
                except (ValueError, OverflowError):
                    issue = 'invalid'
        if issue:
            positions[issue].append(position)
            if len(samples[issue]) < 5:
                samples[issue].append(f'position={position}')
        parsed.append(stamp)
    counts = {kind: len(items) for kind, items in positions.items()}
    if counts['invalid'] or counts['ambiguous'] or missing == 'raise' and counts['missing']:
        detail = '; '.join(f'{kind}: ' + ', '.join(items) for kind, items in samples.items() if items)
        raise ValueError('calendar date parsing failed: '
                         + ', '.join(f'{kind}={count}' for kind, count in counts.items()) + '; ' + detail)
    result = pd.Series(parsed, index=source.index.copy(), name=source.name, dtype='datetime64[ns]')
    report = {'rows_input': len(source), 'rows_output': len(result),
              'rows_parsed': len(source) - counts['missing'], 'rows_missing': counts['missing'],
              'rows_invalid': counts['invalid'], 'rows_ambiguous': counts['ambiguous'],
              'missing_positions': positions['missing'],
              'missing': counts['missing'], 'invalid': counts['invalid'], 'ambiguous': counts['ambiguous']}
    return result, report


def filter_rows(frame, keep, *, reason):
    """Apply one explicit row mask and report positions in the current input frame.

    Parameters
    ----------
    frame : pandas DataFrame; copied, never mutated or sorted.
    keep : one-dimensional boolean sequence with exactly one value per input row.
        A Series must have exactly the same index, including order and duplicates.
        No missing mask values, integer truth conversion or index alignment occurs.
    reason : nonblank description of the caller's filtering rule. The function
        does not infer whether the rule is appropriate or combine earlier steps.

    Returns
    -------
    (DataFrame, dict) : retained rows in source order and reason, rows_input,
        rows_kept, rows_dropped, kept_source_positions, dropped_source_positions.
        Positions are zero-based in THIS call's frame, not index labels. Compose
        successive reports or retain an explicit row id for original-file identity.

    Example
    -------
    >>> kept, report = filter_rows(pd.DataFrame({'x': [4, 9]}), [True, False], reason='declared upper bound')
    >>> report['dropped_source_positions']
    [1]
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('frame must be a pandas DataFrame')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('reason must describe the filtering rule')
    if isinstance(keep, pd.Series) and not keep.index.equals(frame.index):
        raise ValueError('keep Series index must exactly match frame index and row order')
    if not isinstance(keep, (pd.Series, pd.Index, list, tuple, np.ndarray)):
        raise ValueError('keep must be a one-dimensional boolean sequence')
    raw = np.asarray(keep)
    if raw.ndim != 1 or len(raw) != len(frame):
        raise ValueError('keep must have one boolean per input row')
    if any(not isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError('keep must contain only booleans, with no missing values')
    mask = np.asarray(raw, dtype=bool)
    retained, dropped = np.flatnonzero(mask).tolist(), np.flatnonzero(~mask).tolist()
    result = frame.iloc[retained].copy(deep=True)
    return result, {'reason': reason, 'rows_input': len(frame), 'rows_kept': len(retained),
                    'rows_dropped': len(dropped), 'kept_source_positions': retained,
                    'dropped_source_positions': dropped}
