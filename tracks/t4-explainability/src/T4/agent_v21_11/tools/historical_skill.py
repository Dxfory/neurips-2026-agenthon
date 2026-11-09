"""Out-of-sample replay of the statistical baselines on PRE-CUTOFF history only (development only; never in the image).

For every public regression or ranking unit whose entities have their own series in the corpus tables, and for each of
the last M origins: truncate each series at the origin, let `t4agent.stats.forecast` choose its method on that history
alone, forecast the next value (or the change over the unit's horizon) and score the cross-section the way scorer 5.2.2
scores predictive quality against the naive rule (zero change, or the last value carried forward): regression
naive_MAE / (naive_MAE + MAE), ranking the anchored Spearman. 0.5 = as good as the naive rule. No outcome of any unit
is read: every realized value comes from the corpus, before the cutoff.

    ../.venv/bin/python tools/historical_skill.py
"""
import math
import statistics as st
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
from t4agent.corpus import load_corpus, map_entities  # noqa: E402
from t4agent.stats import Series, _is_change, find_series, forecast  # noqa: E402
from t4agent.task import load_task  # noqa: E402

UNITS = AGENT.parent / "track4-analysis-public" / "units"
M = 8


def spearman(a, b):
    def ranks(x):
        o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = math.sqrt(sum((x - ma) ** 2 for x in ra)), math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (va * vb) if va and vb else 0.0


def anchored(q, a):
    return (1.0 if a >= 1 else 0.5 + 0.5 * (q - a) / (1 - a)) if q >= a else (0.5 * q / a if a > 0 else 0.5)


for unit in sorted(p.name for p in UNITS.iterdir() if p.is_dir()):
    spec = load_task(UNITS / unit / "task.json")
    if spec.target_type == "classification":
        continue
    corpus = load_corpus(UNITS / unit / "corpus", spec)
    specific, shared = map_entities(spec, corpus)
    series = {}
    for e in spec.entities:
        own = specific.get(e["entity_id"]) or []
        s = find_series(corpus, own + shared, e, spec.target_name, set(own))
        if s:
            series[e["entity_id"]] = (e, s)
    if len(series) < 3:
        print(f"{unit:30s} {spec.target_type:10s} series found for {len(series)} entities: not replayable")
        continue
    pqs, chosen = [], []
    for j in range(1, M + 1):
        preds, acts, naive = [], [], []
        for eid, (e, s) in series.items():
            v, d = s.values, s.denom
            change = _is_change(s, spec.target_name)
            if change:
                period = s.period_days or 30.0
                periods = (spec.horizon_days or 30) * (5 / 7) if period <= 1.5 else (spec.horizon_days or 30) / period
                h = max(1, int(round(periods)))
                t = len(v) - h - j  # origin index
                if t < h + 8:
                    continue
                hist = Series(s.doc_id, s.header, v[:t + 1], s.dates[:t + 1], s.period_days, s.exact,
                              d[:t + 1] if d else None, s.denom_scale)
                actual = (v[t + h] - v[t]) / d[t] * s.denom_scale if d else v[t + h] - v[t]
                nv = 0.0
            else:
                t = len(v) - j
                if t < 6:
                    continue
                hist = Series(s.doc_id, s.header, v[:t], s.dates[:t], s.period_days, s.exact, None, 1.0)
                actual, nv = v[t], v[t - 1]
            try:
                f = forecast(hist, spec.target_name, spec.horizon_days, e)
            except (ValueError, ZeroDivisionError, IndexError):
                continue
            scale = 100.0 if f.value and actual and abs(f.value) > 30 * max(abs(actual), 1e-9) else 1.0
            preds.append(f.value / scale); acts.append(actual); naive.append(nv)
            chosen.append(f.method in (("zero", "zero (random walk)") if change else ("last",)))  # the naive rule itself
        if len(acts) < 3:
            continue
        if spec.target_type == "ranking":
            raw, a = (spearman(preds, acts) + 1) / 2, max(0.5, (spearman(naive, acts) + 1) / 2) if len(set(naive)) > 1 else 0.5
            pqs.append(anchored(raw, a))
        else:
            n = st.mean(abs(x - y) for x, y in zip(naive, acts)); e_ = st.mean(abs(x - y) for x, y in zip(preds, acts))
            pqs.append(n / (n + e_) if n + e_ > 0 else 0.5)
    if pqs:
        default_share = sum(chosen) / len(chosen)
        print(f"{unit:30s} {spec.target_type:10s} entities={len(series):2d} origins={len(pqs)}  stat-baseline PQ "
              f"(5.2.2, naive=0.5) mean {st.mean(pqs):.3f}  per origin {[round(x, 2) for x in pqs]}  "
              f"the naive rule itself chosen {default_share:.0%}")
