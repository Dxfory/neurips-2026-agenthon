"""Host output report (final boost): what the delivered files hold, sent back to the model to check against the task.

After the program ran cleanly and its required files exist, the host reads the delivered files and builds an OUTPUT
REPORT: per file a bounded summary (CSV/TSV header, row count, first and last rows, numeric column ranges; a JSON
outline; a parquet's shape, dtypes and ranges; a text file's head; else the size). The model compares it with every
requirement of the TASK text and fixes what is wrong. Alongside the report go generic, high-precision violation lines:

    1 non-finite   NaN or +-inf in a required file (CSV: nan, inf or an empty cell in an otherwise numeric column;
                   JSON: NaN/Infinity, a 'NaN' string, a null in a numeric list or in a numeric record field)
    2 constant     a numeric column or JSON numeric list of >= 5 values that are all 0 or all identical
    3 self-check   a boolean field named like a self-check (valid, ok, passed, converged, ...) that is False
    4 probability  a probability, p-value, hit rate or coverage rate outside [0, 1]
    5 negative     a negative volatility, sigma, standard deviation or variance
    6 empty        a required table without data rows, an empty required JSON, a required parquet with 0 rows,
                   a required file of 0 bytes
    7 placeholder  placeholder text (placeholder, TODO, TBD, not implemented, ...) in a delivered string value

Names are read as words (split at non-letters, digits and camelCase): 'lookahead' holds no 'ok', 'invalid' no
'valid', 'swap_value' no 'p value'. Rule 1 leaves NaN/null/empty values to the report's statistics when the TASK
mentions NaN, null, None, empty, blank or missing values, and inf values when it mentions inf or infinity. Rule 2 skips
names that suggest an id, count, date, event count or indicator (and, for a non-zero constant, an echoed parameter such
as a weight, a strike or a one-letter symbol like K1); rule 5 skips names that say a value may be signed (skew,
change, return, residual, covariance, ...); rule 4 skips error, residual, log and percent names; rule 3 skips negated
or configuration names, a False beside a p-value (a test result), and data facts ('satisfied', 'consistent', 'match'
count only beside check, valid, test, ok or pass; 'within' only before 'tolerance'). Numbers are shown exactly: a float
as its shortest round-trip text (repr), a CSV cell as its own text, at most NUMBER_CHARS characters.
The module never edits a file and never raises: a file it cannot read is reported as '(unreadable: <ExcType>)'.
"""
from __future__ import annotations

import csv
import io
import json
import math
import numbers
import re
from collections.abc import Mapping
from functools import lru_cache

REPORT_HEADER = ('OUTPUT REPORT (computed by the host from the files your program delivered, not by your program): '
                 'check every requirement of the TASK against it.')
LIMIT = 6000
TRUNCATED = '...(truncated)'
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.pdf')
TABLE_SUFFIXES = {'.csv': ',', '.tsv': '\t'}
PARQUET_SUFFIXES = ('.parquet', '.pq')
TEXT_SUFFIXES = ('.txt', '.md', '.log')
LINE_CHARS = 200  # a shown table row or outline line
TEXT_CHARS = 400  # head of a text file
STRING_CHARS = 60  # a JSON string value or key in the outline
NUMERIC_COLUMNS = 25  # numeric columns summarized per table
SHOWN_ROWS = 3  # first data rows shown (then the last one)
JSON_DEPTH = 4  # nesting levels of keys in the outline
JSON_KEYS = 30  # keys shown per dict
LIST_ITEMS = 3  # items shown per list
OUTLINE_LINES = 300
MAX_VIOLATIONS = 25
VIOLATION_CHARS = 160
PER_KIND = 4  # violation lines per file and rule before the rest is folded into a count
MIN_VALUES = 5  # rule 2: values needed before a constant series is judged
STAT_CELLS = 2_000_000  # table cells read for column statistics per file
FILE_BYTES = 32 * 1024 * 1024  # a larger file is reported by size only
TOTAL_BYTES = 96 * 1024 * 1024  # bytes parsed per report
WALK_NODES = 2_000_000  # JSON values visited for violations
WALK_DEPTH = 100
MIN_SECTION = 120  # chars a file section keeps when the report is cut
TOLERANCE = 1e-9  # rule 4 bounds
NAME_CHARS = 60  # a column name or path in a violation line
NUMBER_CHARS = 24  # a shown number: a float's repr, an integer, or a CSV cell's own text
HUGE = 1e15  # blocking (thin's outputs_fix): a finite number past this in absolute value is not a plausible result
# blocking: names whose numbers may pass HUGE (ids, counts, dates and times such as nanosecond timestamps)
HUGE_OK_WORDS = frozenset('ns nanos nanosecond nanoseconds us micros ms epoch ts'.split())

KINDS = {1: 'non-finite', 2: 'constant-value', 3: 'self-check', 4: 'probability-range', 5: 'negative-volatility',
         6: 'empty', 7: 'placeholder'}
PRIORITY = {6: 0, 1: 1, 3: 2, 4: 3, 5: 4, 7: 5, 2: 6}  # most certain first, for the MAX_VIOLATIONS cut
MISSING_TOKENS = frozenset(('none', 'null', 'na', 'n/a', '#n/a', 'nat', '<na>'))  # with '' and float nan / inf
NAN_STRINGS = frozenset(('nan', 'NaN', '-nan', 'inf', '-inf', '+inf', 'Inf', '-Inf', 'Infinity', '-Infinity',
                         '+Infinity', 'infinity', '-infinity'))  # a JSON string holding a non-finite number

ID_WORDS = frozenset('''id ids idx index indices count counts cnt n num number nobs obs flag flags year years yr date
    dates datetime time timestamp day days month months quarter week weeks period periods rank ranks bucket buckets
    label labels seed step iter version'''.split())
ID_PREFIXES = ('num', 'idx', 'index', 'count', 'date', 'flag', 'label', 'bucket', 'rank')
ID_SUFFIXES = ('id', 'ids', 'idx')
ZERO_OK_WORDS = frozenset('''is has indicator dummy mask breach breaches exceed exceeds exceedance exceedances
    violation violations hit hits exception exceptions error errors err diff difference residual residuals missing
    outlier outliers anomaly anomalies fail failed failure failures gap gaps skip skipped dropped warning warnings
    rate rates rf r q dividend dividends yield cost costs fee fees commission slippage tax spread lag lags threshold
    cap floor barrier drift mu shift offset adj adjustment borrow funding carry default defaults dof df trades removed
    '''.split())
SAME_OK_WORDS = ZERO_OK_WORDS | frozenset('''weight weights w window horizon alpha confidence conf level tenor maturity
    expiry expiration strike notional quantile param params parameter parameters lookback freq frequency qty quantity
    size lot shares multiplier spot vol volatility sigma principal face coupon par target limit leverage nominal base
    benchmark initial init'''.split())
CHECK_WORDS = frozenset('''valid validated validates ok pass passed passes converged converges success
    successful succeeded'''.split())
FACT_WORDS = frozenset('satisfied consistent match matches matched'.split())  # also data facts ('feller_..._satisfied')
ROW_FACT_WORDS = FACT_WORDS - {'match', 'matches', 'matched'}  # per row a match is data, whatever its name
CHECK_CONTEXT = frozenset('check checks checked valid test tests ok pass passed sanity self selfcheck'.split())
TOLERANCE_WORDS = ('tolerance', 'tolerances', 'tol')  # 'within' is a self-check only before one (not per row)
NOT_CHECK_WORDS = frozenset('''not no non without fail failed failure failures error errors invalid use uses used
    using enable enabled enforce enforced require required requires allow allowed allows should must force forced
    expect expected'''.split())
TEST_WORDS = frozenset('pvalue pvalues pval statistic stat stats tstat zstat lr chi2 critical'.split())
SELF_WORDS = frozenset('check checks sanity self selfcheck'.split())  # a False named so is never a test result
TEST_KEYS = 6  # a test-result dict is small
PROB_WORDS = frozenset('prob probs probability probabilities pvalue pvalues pval pvals'.split())
PROB_NOT_WORDS = frozenset('''log ln logit logprob odds ratio pct percent percentage bps bp weighted z score density pdf
    likelihood err errs error errors relerr abserr residual residuals resid diff diffs difference differences change
    changes chg'''.split())
VOL_PARTS = ('volatility', 'vol_', '_vol', 'sigma', 'std', 'stdev', 'variance')
VOL_WORDS = frozenset('vol vols volatility volatilities sigma sigmas std stdev stddev variance variances'.split())
VOL_NOT_PARTS = ('skew', 'kurt', 'change', 'diff', 'delta', 'return', 'pnl', 'covar', 'volume', 'resid', 'standardi',
                 'zscore', 'log')
VOL_NOT_WORDS = frozenset('''cov z score stat statistic tstat ln spread premium error err bias beta corr correlation
    slope coef coefficient signal excess surprise innovation innovations shock shocks ret rets chg move moves gap
    loading elasticity sensitivity adjusted adj scaled normalized norm weighted momentum regime factor trend carry
    alpha'''.split())

_WORD = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+')
_PLACEHOLDER = re.compile(r'(?<![A-Za-z0-9])(placeholders?|todo|tbd|not\s+implemented|to\s+be\s+implemented|'
                          r'dummy\s+values?)(?![A-Za-z0-9])', re.IGNORECASE)
_INDEX = re.compile(r'\[\d+\]')
_MISSING = object()
# rule 1: TASK words (whole words, '_' a separator, any case) that allow missing or infinite values in the outputs
_ALLOWS = (('missing', re.compile(r'(?<![A-Za-z0-9])(?:nan|nans|null|nulls|none|empty|blank|blanks|n/a|na|missing'
                                  r'[ \t_-]+(?:values?|data|entries|observations|cells|fields))(?![A-Za-z0-9])',
                                  re.IGNORECASE)),
           ('inf', re.compile(r'(?<![A-Za-z0-9])(?:inf|infs|infinity|infinities)(?![A-Za-z0-9])|∞',
                              re.IGNORECASE)))


# ---- names ------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _words(name):
    """Lower-case words of a key or column name (split at non-letters, digits and camelCase humps)."""
    return tuple(word.lower() for word in _WORD.findall(name))


def _pair(words, first, seconds):
    return any(a == first and b in seconds for a, b in zip(words, words[1:]))


@lru_cache(maxsize=4096)
def _constant_judged(name, zero):
    """Rule 2 judges this name: no id, count, date or indicator word; for a non-zero value no echoed parameter
    either (a weight, strike, rate, ... or a one-letter symbol)."""
    words = _words(name)
    if any(w in ID_WORDS or w.startswith(ID_PREFIXES) or w.endswith(ID_SUFFIXES) for w in words):
        return False
    if any(w in (ZERO_OK_WORDS if zero else SAME_OK_WORDS) for w in words):
        return False
    letters = [w for w in words if not w.isdigit()]  # K1, T_2: a one-letter symbol with an index
    return zero or not (len(letters) == 1 and len(letters[0]) == 1)


@lru_cache(maxsize=4096)
def _check_name(name, row):
    """Rule 3: the name says a self-check held. 'satisfied', 'consistent' and 'match' also name data facts (a Feller
    condition, a parity): they count only beside check, valid, test, ok or pass; 'within' only before 'tolerance' and
    outside rows; per row a match is data."""
    words = _words(name)
    found = set(words)
    if found & NOT_CHECK_WORDS:
        return False
    if found & CHECK_WORDS:
        return True
    if found & (ROW_FACT_WORDS if row else FACT_WORDS) and found & CHECK_CONTEXT:
        return True
    return not row and _pair(words, 'within', TOLERANCE_WORDS)


def _test_block(mapping):
    """A small dict that reports a statistical test (a p-value or statistic beside the boolean, at most TEST_KEYS
    keys): its False is a result, not a failed self-check."""
    if len(mapping) > TEST_KEYS:
        return False
    for key in mapping:
        words = _words(str(key))
        if TEST_WORDS & set(words) or _pair(words, 'p', ('value', 'values', 'val')):
            return True
    return False


@lru_cache(maxsize=4096)
def _prob_name(name):
    """Rule 4: the name's words say the values are probabilities (prob, probability, pvalue, p value, hit rate, coverage
    rate; never a substring: 'swap_value' holds no 'p value') and not log, percent, odds, density, error or difference
    values."""
    words = _words(name)
    found = set(words)
    hit = (bool(PROB_WORDS & found) or _pair(words, 'p', ('value', 'values', 'val', 'vals'))
           or _pair(words, 'hit', ('rate',)) or _pair(words, 'coverage', ('rate',)))
    return hit and not PROB_NOT_WORDS & found


@lru_cache(maxsize=4096)
def _vol_name(name):
    """Rule 5: the name says the values are a volatility, sigma, standard deviation or variance (never signed)."""
    low, words = name.lower(), _words(name)
    hit = (any(part in low for part in VOL_PARTS) or bool(VOL_WORDS & set(words))
           or _pair(words, 'standard', ('deviation', 'deviations')))
    return hit and not any(part in low for part in VOL_NOT_PARTS) and not VOL_NOT_WORDS & set(words)


# ---- values -----------------------------------------------------------------------------------------------------

def _isnum(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _num(value):
    """A number shown exactly: an integer in full, a float as its shortest round-trip text (repr: 4.0 stays 4.0, no
    rounding hidden), at most NUMBER_CHARS chars."""
    try:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, numbers.Integral):
            return _clip(str(int(value)), NUMBER_CHARS)
        value = float(value)
        if math.isnan(value):
            return 'NaN'
        if math.isinf(value):
            return 'inf' if value > 0 else '-inf'
        return _clip(repr(value), NUMBER_CHARS)
    except (TypeError, ValueError, OverflowError):
        return '?'


def _shown(value):
    """A cell or value as the report shows it: a CSV cell's own text, else _num (at most NUMBER_CHARS chars)."""
    return _clip(value, NUMBER_CHARS) if isinstance(value, str) else _num(value)


def _allowances(instruction):
    """Rule 1 kinds the TASK allows in the outputs: 'missing' (it mentions NaN, null, None, empty, blank or missing
    values) and 'inf' (it mentions inf or infinity)."""
    text = instruction if isinstance(instruction, str) else ''
    allow = {kind for kind, pattern in _ALLOWS if pattern.search(text)}
    try:  # 'no_rows': the TASK lets a table have no data rows (stub_check reads that the same way)
        from . import stub_check
        if stub_check._empty_table_allowed(text):
            allow.add('no_rows')
    except Exception:
        pass
    return frozenset(allow)


def _label(allow, blank):
    """'NaN/inf/<blank>' without the kinds the TASK allows."""
    missing = 'missing' not in allow
    parts = (['NaN'] if missing else []) + (['inf'] if 'inf' not in allow else []) + ([blank] if missing else [])
    return '/'.join(parts)


def _clip(text, size):
    text = str(text)
    return text if len(text) <= size else text[:max(size - 3, 0)] + '...'


def _scalar(value):
    """A JSON scalar as the outline shows it."""
    if value is None:
        return 'null'
    if isinstance(value, bool) or _isnum(value):
        return _num(value)
    if isinstance(value, str):
        return json.dumps(value if len(value) <= STRING_CHARS else value[:STRING_CHARS] + '...', ensure_ascii=False)
    return type(value).__name__


def _is_scalar(value):
    return not isinstance(value, (dict, list))


def _flat(items):
    return all(_is_scalar(item) for item in items[:LIST_ITEMS])


def _summary(value):
    """One-line view of a JSON value: a scalar, '{n keys}', or 'list[n]' with its first items when they are scalars."""
    if isinstance(value, dict):
        return f'{{{len(value)} key{"" if len(value) == 1 else "s"}}}'
    if isinstance(value, list):
        if value and _flat(value):
            more = ', ...' if len(value) > LIST_ITEMS else ''
            return f'list[{len(value)}] [{", ".join(_scalar(item) for item in value[:LIST_ITEMS])}{more}]'
        return f'list[{len(value)}]'
    return _scalar(value)


def _outline(value, depth, indent, out):
    """Append the outline of a dict's or list's contents: keys down to JSON_DEPTH levels, at most JSON_KEYS keys per
    dict and LIST_ITEMS items per list (a list of scalars is already shown inline by _summary)."""
    if depth >= JSON_DEPTH or len(out) >= OUTLINE_LINES:
        return
    if isinstance(value, dict):
        for count, (key, item) in enumerate(value.items()):
            if count == JSON_KEYS:
                out.append(f'{indent}... (+{len(value) - JSON_KEYS} more keys)')
                return
            if len(out) >= OUTLINE_LINES:
                return
            out.append(f'{indent}{_clip(key, STRING_CHARS)}: {_summary(item)}')
            _outline(item, depth + 1, indent + '  ', out)
    elif isinstance(value, list) and not _flat(value):
        for index, item in enumerate(value[:LIST_ITEMS]):
            if len(out) >= OUTLINE_LINES:
                return
            out.append(f'{indent}[{index}] {_summary(item)}')
            _outline(item, depth + 1, indent + '  ', out)


# ---- tables -----------------------------------------------------------------------------------------------------

def _cell(text):
    """A CSV cell read as _MISSING, a bool, a float or (else) its stripped text."""
    value = text.strip()
    if not value:
        return _MISSING
    low = value.lower()
    if low in ('true', 'false'):
        return low == 'true'
    if low in MISSING_TOKENS:
        return _MISSING
    try:
        return float(value)
    except (ValueError, OverflowError):
        return value


class _Column:
    """Running statistics of one table column (rows numbered from 1)."""

    def __init__(self, name):
        self.name = name
        self.numbers = self.missing = self.inf = self.parsed = self.text = 0
        self.true = self.false = 0
        self.first_missing = self.first_inf = self.first_false = None
        self.low = self.high = self.first = None
        self.low_shown = self.high_shown = self.first_shown = None  # the cell text (CSV) or value behind them
        self.same = True
        self.negative = self.outside = self.placeholders = 0
        self.first_negative = self.first_outside = self.first_placeholder = None
        self.huge, self.first_huge = 0, None  # blocking only: |value| > HUGE

    @property
    def numeric(self):
        return self.parsed > 0 and not self.text and not (self.true or self.false)

    @property
    def boolean(self):
        return bool(self.true or self.false) and not self.text and not self.parsed

    def bad(self, allow):
        """(cells, first row) of the NaN/empty and inf cells rule 1 judges: the kinds the TASK allows aside."""
        kinds = [(count, first) for kind, count, first in (('missing', self.missing, self.first_missing),
                                                          ('inf', self.inf, self.first_inf))
                 if count and kind not in allow]
        return sum(count for count, _ in kinds), min((first for _, first in kinds), default=None)

    def add(self, row, value):
        """Count one cell: a CSV string or a parquet value."""
        shown = value
        if isinstance(value, str):
            shown = value.strip()
            value = _cell(value)
            if isinstance(value, str):
                self.text += 1
                match = _PLACEHOLDER.search(value)
                if match:
                    self.placeholders += 1
                    if self.first_placeholder is None:
                        self.first_placeholder = (row, match.group(0))
                return
        elif value is None or type(value).__name__ in ('NAType', 'NaTType'):
            value = _MISSING
        elif isinstance(value, bool) or getattr(getattr(value, 'dtype', None), 'kind', '') == 'b':
            value = bool(value)
        elif not isinstance(value, float):
            value = _float(value)
            if value is None:
                self.text += 1
                return
        if value is _MISSING:
            self.missing += 1
            self.first_missing = self.first_missing or row
        elif isinstance(value, bool):
            if value:
                self.true += 1
            else:
                self.false += 1
                self.first_false = self.first_false or row
        else:
            self._number(row, value, shown)

    def _number(self, row, value, shown):
        self.parsed += 1
        if not math.isfinite(value):
            if math.isinf(value):
                self.inf += 1
                self.first_inf = self.first_inf or row
            else:
                self.missing += 1
                self.first_missing = self.first_missing or row
            return
        self.numbers += 1
        if self.first is None:
            self.first, self.first_shown = value, shown
        elif value != self.first:
            self.same = False
        if self.low is None or value < self.low:
            self.low, self.low_shown = value, shown
        if self.high is None or value > self.high:
            self.high, self.high_shown = value, shown
        if value < 0:
            self.negative += 1
            self.first_negative = self.first_negative or (row, shown)
        if value < -TOLERANCE or value > 1 + TOLERANCE:
            self.outside += 1
            self.first_outside = self.first_outside or (row, shown)
        if abs(value) > HUGE:
            self.huge += 1
            self.first_huge = self.first_huge or (row, shown)


def _column_lines(columns):
    """Report lines of the numeric (at most NUMERIC_COLUMNS) and boolean columns."""
    lines = []
    numeric = [column for column in columns if column.numeric]
    if numeric:
        lines.append('numeric columns (n = finite values):')
        for column in numeric[:NUMERIC_COLUMNS]:
            inf = f' inf={column.inf}' if column.inf else ''
            span = (f'min={_shown(column.low_shown)} max={_shown(column.high_shown)}' if column.numbers
                    else 'min=- max=-')
            lines.append(_clip(f'  {column.name}: n={column.numbers} nan={column.missing}{inf} {span}', LINE_CHARS))
        if len(numeric) > NUMERIC_COLUMNS:
            lines.append(f'  (+{len(numeric) - NUMERIC_COLUMNS} more numeric columns)')
    booleans = [column for column in columns if column.boolean]
    if booleans:
        shown = ', '.join(f'{c.name} (true={c.true} false={c.false})' for c in booleans[:NUMERIC_COLUMNS])
        lines.append(_clip(f'boolean columns: {shown}', LINE_CHARS))
    return lines


def _column_findings(name, columns, required, partial, unit, allow=frozenset()):
    """[(rule, line)] of a table's columns; rule 2 needs every row read (not partial); rule 1 leaves the kinds in
    `allow` (the TASK allows them) to the statistics."""
    found = []
    if required:
        bad = [(column, *column.bad(allow)) for column in columns if column.numeric]
        bad = [item for item in bad if item[1]]
        if bad:
            first, _, row = min(bad, key=lambda item: item[2])
            listed = ', '.join(repr(_clip(c.name, 24)) for c, _, _ in bad[:3]) + (f' +{len(bad) - 3}' if len(bad) > 3
                                                                                    else '')
            found.append((1, f'{name}: {sum(count for _, count, _ in bad)} {_label(allow, "empty")} numeric cells in '
                             f'{listed} (first: column {_clip(first.name, 40)!r}, {unit} {row})'))
    for column in columns:
        label = repr(_clip(column.name, NAME_CHARS))
        if column.boolean and column.false and _check_name(column.name, True):
            found.append((3, f'{name}: self-check column {label} is False in {column.false} of '
                             f'{column.true + column.false} rows (first: {unit} {column.first_false})'))
        if not column.numeric:
            continue
        if column.outside and _prob_name(column.name):
            row, value = column.first_outside
            found.append((4, f'{name}: column {label} has {column.outside} value(s) outside [0, 1] '
                             f'(first: {unit} {row}, {_shown(value)})'))
        if column.negative and _vol_name(column.name):
            row, value = column.first_negative
            found.append((5, f'{name}: column {label} has {column.negative} negative value(s) '
                             f'(first: {unit} {row}, {_shown(value)})'))
        if not partial and column.numbers >= MIN_VALUES and column.same:
            zero = column.first == 0
            if _constant_judged(column.name, zero):
                what = '0' if zero else f'identical ({_shown(column.first_shown)})'
                found.append((2, f'{name}: numeric column {label} has all {column.numbers} values {what}'))
    hits = [column for column in columns if column.placeholders]
    if hits:
        first = min(hits, key=lambda column: column.first_placeholder[0])
        row, text = first.first_placeholder
        found.append((7, f'{name}: placeholder text {text!r} in {sum(c.placeholders for c in hits)} cell(s) '
                         f'(first: column {_clip(first.name, 40)!r}, {unit} {row})'))
    return found


def _join(row, delimiter):
    out = io.StringIO()
    csv.writer(out, delimiter=delimiter, lineterminator='').writerow(row)
    return out.getvalue().replace('\r', '\\r').replace('\n', '\\n')


def _table(name, data, delimiter, required, allow=frozenset(), huge=None):
    """(lines, findings) of a CSV/TSV file: header, data rows, first and last rows, column statistics."""
    text = data.decode('utf-8-sig', errors='replace')
    if not text.strip():
        return ['(blank file)'], [(6, f'{name}: required table is blank')] if required else []
    header, columns, head, last = None, [], [], None
    rows = ragged = cells = 0
    partial = False
    for row in csv.reader(io.StringIO(text, newline=''), delimiter=delimiter):
        if not any(cell.strip() for cell in row):
            continue
        if header is None:
            header = row
            columns = [_Column(cell.strip() or f'column {j + 1}') for j, cell in enumerate(row)]
            continue
        rows += 1
        if len(head) < SHOWN_ROWS:
            head.append(row)
        last = row
        if len(row) != len(header):
            ragged += 1
        if partial or cells + len(columns) > STAT_CELLS:
            partial = True
            continue
        cells += len(columns)
        for j, column in enumerate(columns):
            column.add(rows, row[j] if j < len(row) else '')
    lines = [_clip(f'header: {_join(header, delimiter)}', LINE_CHARS), f'data rows: {rows}']
    lines += [_clip(f'row {i}: {_join(row, delimiter)}', LINE_CHARS) for i, row in enumerate(head, 1)]
    if rows > SHOWN_ROWS:
        if rows > SHOWN_ROWS + 1:
            lines.append('...')
        lines.append(_clip(f'row {rows}: {_join(last, delimiter)}', LINE_CHARS))
    if ragged:
        lines.append(f'note: {ragged} data row(s) have a cell count other than the header\'s {len(header)}')
    if partial:
        lines.append(f'(column statistics over the first {cells // max(len(columns), 1)} data rows)')
    lines += _column_lines(columns)
    found = _column_findings(name, columns, required, partial, 'data row', allow)
    if required and not rows and 'no_rows' not in allow:
        found.append((6, f'{name}: required table has a header but no data rows'))
    if huge is not None:
        huge += _huge_lines(name, columns, 'data row')
    return lines, found


def _parquet_over_bound(data):
    """clock_fix: whether stub_check.parquet_bound (the one bound for both readers) shows the file over the bound; a
    file without readable metadata is read as before (its reader fails on the footer and says so)."""
    try:
        from .stub_check import parquet_bound
        return parquet_bound(data) is False
    except Exception:
        return True


def _parquet(name, data, required, allow=frozenset(), huge=None):
    """(lines, findings) of a parquet file: shape, dtypes, first rows (floats shown exactly), column statistics."""
    try:
        import pandas
    except Exception as exc:  # optional dependency
        return [f'(parquet not summarized: pandas unavailable, {type(exc).__name__})'], []
    frame = pandas.read_parquet(io.BytesIO(data))
    rows, width = frame.shape
    lines = [f'shape: {rows} rows x {width} columns',
             _clip('columns: ' + ', '.join(f'{c}:{t}' for c, t in zip(frame.columns, frame.dtypes)), 2 * LINE_CHARS)]
    if rows and width:
        lines.append('first rows:')
        shown = frame.head(SHOWN_ROWS).to_string(max_colwidth=40, float_format=_num)
        lines += [_clip(f'  {line}', LINE_CHARS) for line in shown.splitlines()]
    read = rows if rows * max(width, 1) <= STAT_CELLS else STAT_CELLS // max(width, 1)
    columns = []
    for j in range(width):
        column = _Column(str(frame.columns[j]))
        for i, value in enumerate(frame.iloc[:read, j].tolist(), 1):
            column.add(i, value)
        columns.append(column)
    if read < rows:
        lines.append(f'(column statistics over the first {read} rows)')
    lines += _column_lines(columns)
    found = _column_findings(name, columns, required, read < rows, 'row', allow)
    if required and not rows and 'no_rows' not in allow:
        found.append((6, f'{name}: required parquet has 0 rows'))
    if huge is not None:
        huge += _huge_lines(name, columns, 'row')
    return lines, found


# ---- JSON -------------------------------------------------------------------------------------------------------

class _Walk:
    """Rule hits in a parsed JSON value (paths like a.b[2].c)."""

    def __init__(self):
        self.nodes = self.placeholders = 0
        self.first_placeholder = None
        self.nonfinite = {'missing': 0, 'inf': 0}
        self.first_nonfinite = {}  # kind ('missing': NaN or null, 'inf') -> first path, in the order found
        self.groups = {}  # (rule, path with [] for indexes) -> [first path, count, value]
        self.constant = []
        self.huge, self.first_huge = 0, None  # blocking only: |value| > HUGE under a name that is no id or time

    def bad(self, path, kind='missing'):
        self.nonfinite[kind] += 1
        self.first_nonfinite.setdefault(kind, path or '(top level)')

    def note(self, rule, path, detail=''):
        key = (rule, _INDEX.sub('[]', path))
        if key in self.groups:
            self.groups[key][1] += 1
        else:
            self.groups[key] = [path or '(top level)', 1, detail]

    def visit(self, value, path='', name='', row=False, depth=0):
        self.nodes += 1
        if self.nodes > WALK_NODES or depth > WALK_DEPTH:
            return
        if isinstance(value, dict):
            test = None
            for key, item in value.items():
                key = str(key)
                where = f'{path}.{key}' if path else key
                if item is False and _check_name(key, row):
                    test = _test_block(value) if test is None else test
                    if not test or SELF_WORDS & set(_words(key)):
                        self.note(3, where)
                else:
                    self.visit(item, where, key, row, depth + 1)
        elif isinstance(value, list):
            self.series(value, path, name)
            for index, item in enumerate(value):
                where = f'{path}[{index}]'
                if item is False and _check_name(name, True):
                    self.note(3, where)
                else:
                    self.visit(item, where, name, True, depth + 1)
        elif _isnum(value):
            number = _float(value)
            if number is None:
                return
            if not math.isfinite(number):
                self.bad(path, 'inf' if math.isinf(number) else 'missing')
                return
            if (number < -TOLERANCE or number > 1 + TOLERANCE) and _prob_name(name):
                self.note(4, path, _num(value))
            if number < 0 and _vol_name(name):
                self.note(5, path, _num(value))
            if abs(number) > HUGE and not _huge_ok(name):
                self.huge += 1
                self.first_huge = self.first_huge or (path or '(top level)', _num(value))
        elif isinstance(value, str):
            if value.strip() in NAN_STRINGS:
                self.bad(path, 'missing' if 'nan' in value.lower() else 'inf')
            match = _PLACEHOLDER.search(value)
            if match:
                self.placeholders += 1
                if self.first_placeholder is None:
                    self.first_placeholder = (path or '(top level)', match.group(0))

    def series(self, items, path, name):
        """A numeric list (numbers and nulls): its nulls count as NaN, and >= MIN_VALUES finite values all equal make
        a constant series. A list of records: a null in an otherwise numeric field counts as NaN."""
        numbers = [_float(item) for item in items if _isnum(item)]
        if numbers and len(numbers) + sum(item is None for item in items) == len(items):
            for index, item in enumerate(items):
                if item is None:
                    self.bad(f'{path}[{index}]')
            finite = [(x, item) for x, item in zip(numbers, (item for item in items if _isnum(item)))
                      if x is not None and math.isfinite(x)]
            if len(finite) >= MIN_VALUES and all(x == finite[0][0] for x, _ in finite):
                zero = finite[0][0] == 0
                if _constant_judged(name, zero):
                    self.constant.append((path or '(top level)', len(finite), finite[0][1], zero))
        elif items and all(isinstance(item, dict) for item in items):
            keys = dict.fromkeys(key for item in items for key in item)
            if len(keys) * len(items) > WALK_NODES:
                return
            for key in keys:
                values = [item[key] for item in items if key in item]
                if (any(v is None for v in values) and any(_isnum(v) for v in values)
                        and all(v is None or _isnum(v) for v in values)):
                    for index, item in enumerate(items):
                        if key in item and item[key] is None:
                            self.bad(f'{path}[{index}].{key}')


def _json(name, data, required, allow=frozenset(), huge=None):
    """(lines, findings) of a JSON file: an outline, its NaN/null/inf counts and the rule hits (rule 1 leaves the
    kinds in `allow`, which the TASK allows, to the counts)."""
    value = json.loads(data.decode('utf-8-sig'))
    lines = [f'top level: {_summary(value)}']
    _outline(value, 0, '  ', lines)
    if len(lines) > OUTLINE_LINES:
        lines = lines[:OUTLINE_LINES] + ['  ...']
    walk = _Walk()
    walk.visit(value)
    if walk.first_nonfinite:
        lines.append('non-finite numeric values: ' + ', '.join(
            f'{"NaN/null" if kind == "missing" else "inf"}={walk.nonfinite[kind]} (first at {_clip(path, NAME_CHARS)})'
            for kind, path in walk.first_nonfinite.items()))
    found = []
    judged = [kind for kind in walk.first_nonfinite if kind not in allow]  # in the order found
    if required and judged:
        found.append((1, f'{name}: {sum(walk.nonfinite[kind] for kind in judged)} {_label(allow, "null")} numeric '
                         f'value(s) (first at {_clip(walk.first_nonfinite[judged[0]], NAME_CHARS)})'))
    for (rule, _), (path, count, detail) in walk.groups.items():
        path, more = _clip(path, NAME_CHARS), f' (+{count - 1} more like it)' if count > 1 else ''
        if rule == 3:
            found.append((3, f'{name}: self-check field {path} is False{more}'))
        elif rule == 4:
            found.append((4, f'{name}: {path} = {detail} is outside [0, 1]{more}'))
        else:
            found.append((5, f'{name}: {path} = {detail} is negative{more}'))
    for path, count, number, zero in walk.constant:
        what = '0' if zero else f'identical ({_num(number)})'  # number: the list's first finite item as written
        found.append((2, f'{name}: numeric list {_clip(path, NAME_CHARS)} has all {count} values {what}'))
    if walk.placeholders:
        path, text = walk.first_placeholder
        found.append((7, f'{name}: placeholder text {text!r} in {walk.placeholders} string value(s) '
                         f'(first at {_clip(path, NAME_CHARS)})'))
    if required and (value is None or (isinstance(value, (dict, list)) and not value)):
        found.append((6, f'{name}: required JSON is empty (top level {json.dumps(value)})'))
    if huge is not None and walk.huge:
        path, number = walk.first_huge
        huge.append(f'{name}: {walk.huge} numeric value(s) with |value| > {HUGE:.0e} '
                    f'(first at {_clip(path, NAME_CHARS)} = {number})')
    return [_clip(line, LINE_CHARS) for line in lines], found


def _text(name, data, required, suffix):
    """(lines, findings) of a text file: its head; placeholders only in a required .txt/.md deliverable."""
    text = data.decode('utf-8', errors='replace')
    more = '...' if len(text) > TEXT_CHARS else ''
    lines = [f'{len(text)} chars, first {min(len(text), TEXT_CHARS)}:', text[:TEXT_CHARS] + more]
    found = []
    if required and suffix != '.log':
        if not text.strip():
            found.append((6, f'{name}: required file is blank'))
        match = _PLACEHOLDER.search(text)
        if match:
            found.append((7, f'{name}: placeholder text {match.group(0)!r} at char {match.start()}'))
    return lines, found


# ---- files and report -------------------------------------------------------------------------------------------

def _norm(path):
    path = str(path).replace('\\', '/')
    while path.startswith('./'):
        path = path[2:]
    return path.lstrip('/')


def _skipped(path):
    return path.lower().endswith(IMAGE_SUFFIXES) or any(part.startswith('_') for part in path.split('/'))


def _suffix(path):
    base = path.rsplit('/', 1)[-1].lower()
    return '.' + base.rsplit('.', 1)[-1] if '.' in base else ''


def _plan(expected_files, files):
    """[(path, data or None when not delivered, required)]: expected files in their order, then the others sorted;
    images and '_' paths left out. Without expected files every file is required."""
    if isinstance(expected_files, (str, bytes)):
        expected_files = [expected_files]
    try:
        expected = [_norm(name) for name in (expected_files or ())]
    except TypeError:
        expected = []
    entries = {}
    for key, data in (files.items() if isinstance(files, Mapping) else ()):
        path = _norm(key)
        if path and not _skipped(path):
            entries.setdefault(path, data)
    plan, used = [], set()
    for want in dict.fromkeys(expected):
        if not want or _skipped(want):
            continue
        match = want if want in entries and want not in used else None
        if match is None:
            base = want.rsplit('/', 1)[-1]
            same = [path for path in entries if path not in used and path.rsplit('/', 1)[-1] == base]
            match = same[0] if len(same) == 1 else None
        if match is None:
            plan.append((want, None, True))
        else:
            used.add(match)
            plan.append((match, entries[match], True))
    plan += [(path, entries[path], not expected) for path in sorted(entries) if path not in used]
    return plan, bool(expected)


def _file(name, data, required, tag, budget, allow=frozenset(), huge=None, bound_parquet=False):
    """(section text, [(rule, line)]) of one delivered file; budget is the shared [bytes left to parse]; allow holds
    the rule 1 kinds the TASK allows; `huge` (a list, blocking only) gets the file's lines on numbers past HUGE;
    bound_parquet (clock_fix): a parquet file beyond the bound its metadata shows (stub_check.parquet_bound) is
    reported by size only, without a finding."""
    if data is None:
        return f'== {name} ({tag}) ==\n(not delivered)', []
    lines, found, size = [], [], '?'
    try:
        if isinstance(data, str):
            data = data.encode('utf-8', errors='replace')
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(type(data).__name__)
        data = bytes(data)
        size, suffix = len(data), _suffix(name)
        if not data:
            lines = ['(empty file)']
            found = [(6, f'{name}: required file is empty (0 bytes)')] if required else []
        elif size > FILE_BYTES or size > budget[0]:
            lines = ['(not summarized: too large)']
        else:
            budget[0] -= size
            if suffix in TABLE_SUFFIXES:
                lines, found = _table(name, data, TABLE_SUFFIXES[suffix], required, allow, huge)
            elif suffix == '.json':
                lines, found = _json(name, data, required, allow, huge)
            elif suffix in PARQUET_SUFFIXES and bound_parquet and _parquet_over_bound(data):
                lines = ['(not summarized: too large once decoded)']
            elif suffix in PARQUET_SUFFIXES:
                lines, found = _parquet(name, data, required, allow, huge)
            elif suffix in TEXT_SUFFIXES:
                lines, found = _text(name, data, required, suffix)
            else:
                lines = ['(not summarized: size only)']
    except Exception as exc:  # a file we cannot read is reported, never raised
        lines, found = [f'(unreadable: {type(exc).__name__})'], []
    return '\n'.join([f'== {name} ({tag}, {size} bytes) =='] + lines), found


def _cut(text, size):
    """text within size chars, ending with TRUNCATED when cut (at a line end when one is near)."""
    if len(text) <= size:
        return text
    if size <= len(TRUNCATED):
        return TRUNCATED[:max(size, 0)]
    room = size - len(TRUNCATED) - 1
    head = text[:room]
    end = head.rfind('\n')
    if end >= room // 2:
        head = head[:end]
    return f'{head}\n{TRUNCATED}' if head else TRUNCATED


def _fit(sections, budget):
    """The sections joined by newlines within budget chars: each long section is cut to a fair share (shorter ones
    give their unused share to the longer), sections past what fits at MIN_SECTION chars are dropped and counted, and
    a cut report ends with TRUNCATED."""
    joined = '\n'.join(sections)
    if len(joined) <= budget:
        return joined
    keep, tail = len(sections), ''
    while keep > 1 and keep * (MIN_SECTION + 1) + len(tail) > budget:
        keep -= 1
        tail = f'({len(sections) - keep} more files not shown){TRUNCATED}'
    avail = budget - (keep - 1) - (len(tail) if tail else len(TRUNCATED)) - 1
    shares, left = {}, avail
    for rank, index in enumerate(sorted(range(keep), key=lambda i: len(sections[i]))):
        shares[index] = min(len(sections[index]), max(left, 0) // (keep - rank))
        left -= shares[index]
    parts = [_cut(sections[index], shares[index]) for index in range(keep)]
    if tail:
        parts.append(tail)
    elif not parts[-1].endswith(TRUNCATED):
        parts.append(TRUNCATED)
    return '\n'.join(parts)


def _violations(found):
    """Violation lines: most certain rules first, then by file; per file and rule at most PER_KIND lines (the rest
    folded into a count); clipped, deduplicated, at most MAX_VIOLATIONS."""
    groups = {}
    for index, name, rule, line in found:
        groups.setdefault((PRIORITY.get(rule, 9), index, rule, name), []).append(line)
    lines = []
    for (_, _, rule, name), group in sorted(groups.items(), key=lambda item: item[0][:2]):
        if len(group) > PER_KIND:
            group = group[:PER_KIND - 1] + [f'{name}: +{len(group) - PER_KIND + 1} more {KINDS[rule]} findings']
        lines += group
    out = []
    for line in lines:
        line = _clip(line, VIOLATION_CHARS)
        if line not in out:
            out.append(line)
        if len(out) == MAX_VIOLATIONS:
            break
    return out


def _huge_ok(name):
    """Whether a column or key name may hold numbers past HUGE: an id, count, date or time (a nanosecond timestamp)."""
    return bool(set(_words(str(name))) & (ID_WORDS | HUGE_OK_WORDS))


def _huge_lines(name, columns, unit):
    """blocking's lines on a table's numeric columns that hold numbers past HUGE (id and time names aside)."""
    hits = [column for column in columns if column.numeric and column.huge and not _huge_ok(column.name)]
    if not hits:
        return []
    first = min(hits, key=lambda column: column.first_huge[0])
    listed = ', '.join(repr(_clip(c.name, 24)) for c in hits[:3]) + (f' +{len(hits) - 3}' if len(hits) > 3 else '')
    row, value = first.first_huge
    return [f'{name}: {sum(c.huge for c in hits)} numeric cells with |value| > {HUGE:.0e} in {listed} '
            f'(first: column {_clip(first.name, 40)!r}, {unit} {row}, {_shown(value)})']


def blocking(instruction: str, expected_files, files: dict) -> list:
    """thin's outputs_fix: the lines that keep a clean run from passing: rule 1 of each required file (NaN or inf,
    or an empty cell in an otherwise numeric column; the kinds the TASK allows aside, as in build) and the numbers
    of a required file past HUGE in absolute value (a table's numeric column, a JSON number; names of ids, counts,
    dates and times aside). Each line names the file, the column or path and the count. Never raises."""
    try:
        plan, has_expected = _plan(expected_files, files)
        budget, allow, lines = [TOTAL_BYTES], _allowances(instruction), []
        for name, data, required in plan:
            if data is None or not required:
                continue
            huge = []
            _, hits = _file(name, data, required, 'required', budget, allow, huge, bound_parquet=True)
            lines += [_clip(line, VIOLATION_CHARS) for rule, line in hits if rule == 1]
            lines += [_clip(line, VIOLATION_CHARS) for line in huge]
        return lines[:MAX_VIOLATIONS]
    except Exception:
        return []


def build(instruction: str, expected_files, files: dict, program: str = '', limit: int = LIMIT, *,
          bound_parquet=False) -> dict:
    """{'report': str, 'violations': [str]} for the delivered `files` ({relative path: bytes}). `expected_files` names
    the required files (empty: every file is required). The report starts with REPORT_HEADER and has at most `limit`
    chars; the checks read the files, and `instruction` (the TASK text) only for the values rule 1 allows (`program` is
    kept for the caller). `bound_parquet` (clock_fix): a parquet file is decoded only within the bound its metadata
    shows. Never raises."""
    try:
        limit = int(limit)
    except (TypeError, ValueError, OverflowError):
        limit = LIMIT
    try:
        plan, has_expected = _plan(expected_files, files)
        budget, sections, found, allow = [TOTAL_BYTES], [], [], _allowances(instruction)
        for index, (name, data, required) in enumerate(plan):
            tag = ('required' if required else 'extra') if has_expected else 'delivered'
            section, hits = _file(name, data, required, tag, budget, allow, bound_parquet=bound_parquet)
            sections.append(section)
            found += [(index, name, rule, line) for rule, line in hits]
        body = _fit(sections, limit - len(REPORT_HEADER) - 1) if sections else '(no delivered files to report)'
        return {'report': _cut(f'{REPORT_HEADER}\n{body}', limit), 'violations': _violations(found)}
    except Exception as exc:  # never raise into the solver loop
        return {'report': _cut(f'{REPORT_HEADER}\n(report unavailable: {type(exc).__name__})', max(limit, 0)),
                'violations': []}
