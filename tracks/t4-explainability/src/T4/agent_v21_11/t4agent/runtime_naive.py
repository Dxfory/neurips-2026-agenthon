"""Corpus-only fallback with no externally calibrated or selected forecast method.

Use the current task's parsed series: last observation for next-value targets,
zero for changes (including changes of a ratio's numerator). The central normal
90% interval is theoretical: z(.95) times the observed step RMS, with the existing
calendar-to-series horizon conversion. It is not measured coverage or a backtest.

Only parsing and quantity/horizon semantics are shared with ``stats``; its fitted
scale tables, method choices and small-sample corrections are never consulted.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal
from statistics import NormalDist

from .corpus import Corpus
from .stats import Series, StatForecast, _horizon_steps, _is_change, _steps, find_series


class RuntimeForecast(StatForecast):
    """Same downstream data contract, without an empirical-performance claim."""

    def describe(self) -> str:
        width = f"{self.q90:.4g}" if self.q90 is not None else "unavailable"
        return (
            f"series '{self.series.header}' in {self.series.doc_id} "
            f"({len(self.series.values)} obs, last {self.last:g}); "
            f"corpus-only rule '{self.method}' -> {self.value:.4g}; "
            f"theoretical normal 90% half-width {width} from observed steps"
        )


def _target_scale(series: Series, target_name: str, ent: dict) -> float:
    """Convert percent-quoted levels to bps while retaining existing unit inference.

Explicit source units take precedence. If a source header omits units, retain
the existing parser path's <30 magnitude inference; this is ambiguous for small
values already in bps and is not evidence of a known source unit.
"""
    bps = r"(?:^|[^a-z])(?:bps?|basis[\s_-]+points?)(?:$|[^a-z])"
    target = f"{target_name} {ent.get('unit') or ''} {ent.get('units') or ''}"
    if series.denom is not None or not re.search(bps, target, re.I):
        return 1.0
    if re.search(bps, series.header, re.I):
        return 1.0
    if re.search(r"%|\bpct\b|\bpercent(?:age)?\b", series.header, re.I):
        return 100.0
    return 100.0 if max(abs(v) for v in series.values) < 30 else 1.0


def _numeric_resolution(values: list[float]) -> float:
    """One unit of the finest retained decimal precision (integers: one unit).

Series parsing has already removed printed trailing zeroes; own-rel log values
also no longer retain the source level's printed precision. This fallback uses
the available numeric representation and does not infer missing source digits.
"""
    exponent = min(0, *(Decimal(str(v)).normalize().as_tuple().exponent for v in values))
    return float(Decimal(10) ** exponent)


def series_half_width(series: Series, target_name: str, horizon_days: int | None, ent: dict) -> float | None:
    """Theoretical central-normal 90% width from this task's observed step RMS.

Ratio steps use each starting denominator; a flat numerator's numeric precision
is converted using the latest denominator. Named next-value columns use one
step, while changes and unnamed own-level series use the existing horizon rule.
"""
    values, denom = series.values, series.denom
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        return None
    if denom is not None and (
        len(denom) != len(values) or not all(math.isfinite(v) and v != 0 for v in denom)
        or not math.isfinite(series.denom_scale) or series.denom_scale <= 0
    ):
        return None
    steps = _steps(series)
    if not steps or not all(math.isfinite(v) for v in steps):
        return None
    rms = math.hypot(*steps) / math.sqrt(len(steps))
    if rms == 0:
        rms = _numeric_resolution(values)
        if denom is not None:
            rms *= abs(series.denom_scale / denom[-1])
    periods = 1.0
    if (_is_change(series, target_name) or series.kind == "own") and horizon_days:
        periods = max(1.0, _horizon_steps(series, horizon_days))
    width = NormalDist().inv_cdf(.95) * rms * math.sqrt(periods) * _target_scale(series, target_name, ent)
    return width if math.isfinite(width) and width > 0 else None


def entity_forecast(corpus: Corpus, doc_ids: list[str], ent: dict, target_name: str,
                    horizon_days: int | None, entity_docs: set[str], allow_own: bool = False) -> StatForecast | None:
    """Forecast from an admitted current-task series, without any method selection."""
    try:
        series = find_series(corpus, doc_ids, ent, target_name, entity_docs, allow_own)
        if series is None:
            return None
        width = series_half_width(series, target_name, horizon_days, ent)
        if width is None:
            return None
        change = _is_change(series, target_name)
        value = 0.0 if change else series.values[-1] * _target_scale(series, target_name, ent)
        if not math.isfinite(value):
            return None
        return RuntimeForecast(
            method="runtime zero" if change else "runtime last", value=value, mae=None,
            q90=width, n_backtest=0, last=series.values[-1], series=series,
        )
    except (ValueError, ZeroDivisionError, IndexError, OverflowError):
        return None
