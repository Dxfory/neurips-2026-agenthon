"""Build Track-4-format units in families no public unit has, from public data resolved before 2022 (development only;
never in the image).

The held-out set "spans many more task families", most unpublished. These ten units test shapes the public units lack:
a change in thousands, a level in percent, three labels with inverted FX quotes, dollar prices a week ahead, a ranking
of 51 states, basis-point changes of percent spreads over a quarter, a table without its latest value, a text-only
decision unit, rows in different units, and an index's percent change. Every number comes from FRED / ALFRED as it was
published (real-time vintages for revised series), statements from federalreserve.gov. Every outcome resolves before
2022 (TRAINING-POLICY: material used for selection must predate the task cutoffs; the Final's resolve after the House
model's training cutoff, at least December 2024).

Outcomes and the declared naive answers go to <out>/reference/<unit>/, apart from the units the agent reads; scoring
copies them into a temporary unit (tools/score_pre2022.py). The House model has probably seen these outcomes (the
public units are memorization-exposed too), so the scores measure robustness and relative change between versions, not
honest skill.

    ../.venv/bin/python tools/build_pre2022_units.py                 # all units -> ../units_pre2022/
    ../.venv/bin/python tools/build_pre2022_units.py --only t4-x-fx-201905
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import shutil
import statistics as st
import time
import urllib.parse
import urllib.request
import uuid
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # track4_my
OUT = ROOT / "units_pre2022"
RAW = OUT / "raw"
FRED = "https://api.stlouisfed.org/fred/"
LICENSE = "Public domain (US government works), accessed via FRED/ALFRED (Federal Reserve Bank of St. Louis)"
LICENSE_FED = "Public domain (US government works: Board of Governors of the Federal Reserve System)"
Z90 = 1.645


# ------------------------------------------------------------------------------------------------ data
def _key() -> str:
    return re.search(r"[0-9a-f]{32}", (ROOT / ".fred.env").read_text()).group(0)


def fred(path: str, **params) -> dict:
    """One FRED/ALFRED API call, cached under raw/ without the key."""
    RAW.mkdir(parents=True, exist_ok=True)
    tag = path + "?" + urllib.parse.urlencode(sorted(params.items()))
    cache = RAW / (hashlib.sha1(tag.encode()).hexdigest()[:20] + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    q = urllib.parse.urlencode({**params, "api_key": _key(), "file_type": "json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(FRED + path + "?" + q, timeout=60) as r:
                d = json.load(r)
            break
        except Exception:  # noqa: BLE001 - retry transient errors
            if attempt == 4:
                raise
            time.sleep(2 + 3 * attempt)
    cache.write_text(json.dumps({"_query": tag, **d}))
    time.sleep(0.4)
    return d


def series_info(sid: str) -> dict:
    return fred("series", series_id=sid)["seriess"][0]


def obs(sid: str, start: str, end: str, asof: str | None = None) -> list[tuple[str, float]]:
    """Observations start..end; as published on `asof` (ALFRED vintage) when given, else the current values."""
    p = {"series_id": sid, "observation_start": start, "observation_end": end}
    if asof:
        p.update(realtime_start=asof, realtime_end=asof)
    d = fred("series/observations", **p)
    return [(o["date"], float(o["value"])) for o in d["observations"] if o["value"] not in (".", "")]


def vintage_dates(sid: str) -> list[str]:
    return fred("series/vintagedates", series_id=sid, realtime_start="1776-07-04", limit=10000)["vintage_dates"]


def vintages_around(sid: str, cutoff: str, k: int = 4) -> tuple[list[str], str]:
    """The last k vintage dates on or before the cutoff, and the first one after it (the resolving release)."""
    vd = vintage_dates(sid)
    before = [v for v in vd if v <= cutoff]
    after = [v for v in vd if v > cutoff]
    return before[-k:], after[0]


def fed_text(url: str) -> str:
    """Plain text of a federalreserve.gov page's article body (cached)."""
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / ("fed_" + hashlib.sha1(url.encode()).hexdigest()[:16] + ".html")
    if not cache.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; track4 unit builder)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cache.write_bytes(r.read())
        time.sleep(1.0)
    raw = cache.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<div[^>]+id="article"[^>]*>(.*?)<div[^>]+id="lastUpdate"', raw, re.S) or \
        re.search(r'<div[^>]+id="article"[^>]*>(.*)', raw, re.S)
    body = m.group(1) if m else raw
    body = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
    body = re.sub(r"(?i)<br\s*/?>|</p>|</h\d>|</li>|</tr>", "\n", body)
    body = re.sub(r"<[^>]+>", " ", body)
    body = html.unescape(body)
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in body.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return re.sub(r"\n{3,}", "\n\n", text)


# ------------------------------------------------------------------------------------------------ helpers
def ymd(d: str) -> str:
    return d.replace("-", "")


def month(d: str) -> str:
    return d[:7]


def sd(xs: list[float]) -> float:
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def fmt(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def vintage_doc(sid: str, info: dict, cutoff: str, vints: list[str], nmonths: int, extra: str = "",
                doc_id: str | None = None) -> dict:
    """An ALFRED vintage table as known on the cutoff, in the style of the public macro units."""
    tabs = {v: dict(obs(sid, "1990-01-01", v, asof=v)) for v in vints}
    last = tabs[vints[-1]]
    months = sorted(last)[-nmonths:]
    rows = []
    for m in months:
        cells = [fmt(tabs[v][m]) if m in tabs[v] else "--" for v in vints]
        rows.append(f"{m[:7]} | " + " | ".join(cells))
    notes = []
    for m in months:
        vals = [(v, tabs[v][m]) for v in vints if m in tabs[v]]
        for (v0, x0), (v1, x1) in zip(vals, vals[1:]):
            if abs(x1 - x0) > 1e-9:
                notes.append(f"- The {m[:7]} estimate was revised {'UP' if x1 > x0 else 'DOWN'} from {fmt(x0)} "
                             f"(as of {v0}) to {fmt(x1)} (as of {v1}).")
    lm = months[-1]
    notes.append(f"- Latest estimate as of {vints[-1]}: {lm[:7]} = {fmt(last[lm])} {info['units']}.")
    if len(months) > 1:
        pm = months[-2]
        notes.append(f"- Change {pm[:7]} -> {lm[:7]} in the {vints[-1]} vintage: {fmt(last[lm] - last[pm])}"
                     f" ({fmt(100 * (last[lm] / last[pm] - 1), 2)}%).")
    text = (f"{sid} - {info['title']}\nSource: ALFRED (Federal Reserve Bank of St. Louis archival FRED), series {sid}. "
            f"Units: {info['units']}, {info.get('seasonal_adjustment', '')}. Frequency: {info['frequency']}.\n\n"
            f"VINTAGE TABLE - the value of {sid} for each reference period exactly as published on each pre-cutoff "
            f"release (vintage) date.\n\nreference_period | " + " | ".join(f"as_of_{v}" for v in vints) + "\n"
            + "\n".join(rows) + "\n\nNOTES (derived from the vintage table above):\n" + "\n".join(notes[-12:])
            + (("\n\n" + extra) if extra else ""))
    return {"doc_id": doc_id or f"ALFRED_{sid}_VINTAGES_{ymd(cutoff)}", "source": "ALFRED", "doc_type": "vintage_table",
            "series_id": sid, "doc_date": vints[-1], "title": f"{sid} ({info['title']}) - real-time vintage table as of "
            f"{vints[-1]}", "url": f"https://alfred.stlouisfed.org/series?seid={sid}", "pii_stripped": True, "text": text}


def series_doc(sid: str, info: dict, cutoff: str, nobs: int, start: str, doc_id: str | None = None,
               asof: str | None = None) -> dict:
    """A recent-history table of a market or weekly series, as of the cutoff."""
    xs = [(d, v) for d, v in obs(sid, start, cutoff, asof=asof) if d <= cutoff]
    tail = xs[-nobs:]
    last_d, last_v = xs[-1]
    vals = [v for _, v in xs]

    def back(days: int) -> tuple[str, float] | None:
        lim = (date.fromisoformat(last_d) - timedelta(days=days)).isoformat()
        prior = [(d, v) for d, v in xs if d <= lim]
        return prior[-1] if prior else None

    lines = [f"- Latest observation: {last_d} = {fmt(last_v, 4)} {info['units']}."]
    for days, name in ((7, "one week"), (30, "one month"), (91, "three months"), (365, "one year")):
        b = back(days)
        if b:
            lines.append(f"- Change over {name} (since {b[0]}): {fmt(last_v - b[1], 4)} "
                         f"({fmt(100 * (last_v / b[1] - 1), 2) if b[1] else 'n/a'}%).")
    year = [v for d, v in xs if d >= (date.fromisoformat(last_d) - timedelta(days=365)).isoformat()]
    lines.append(f"- 52-week range: {fmt(min(year), 4)} to {fmt(max(year), 4)}.")
    text = (f"{sid} - {info['title']}\nSource: FRED (Federal Reserve Bank of St. Louis), series {sid}. Units: "
            f"{info['units']}. Frequency: {info['frequency']}. {info.get('seasonal_adjustment', '')}.\n\n"
            f"RECENT OBSERVATIONS (as of {cutoff})\n\ndate | value\n"
            + "\n".join(f"{d} | {fmt(v, 4)}" for d, v in tail) + "\n\nSUMMARY (derived from the table above):\n"
            + "\n".join(lines))
    return {"doc_id": doc_id or f"FRED_{sid}_{ymd(last_d)}", "source": "FRED", "doc_type": "series_table",
            "series_id": sid, "doc_date": last_d, "title": f"{sid} ({info['title']}) - recent observations to {last_d}",
            "url": f"https://fred.stlouisfed.org/series/{sid}", "pii_stripped": True, "text": text}


def text_doc(doc_id: str, url: str, doc_date: str, title: str, form: str) -> dict:
    return {"doc_id": doc_id, "source": "FederalReserve", "form_type": form, "doc_date": doc_date, "title": title,
            "url": url, "pii_stripped": True, "text": fed_text(url)}


# ------------------------------------------------------------------------------------------------ unit writer
RUBRIC = ("Every claim must cite a passage from a corpus document whose doc_date <= {cutoff}. A claim is supported "
          "when a passage it cites entails the claim text above tau_citation = 0.5; a cited document must be one the "
          "manifest labels with the citing entity or marks shared. Failure codes: T4_UNFAITHFUL_CITATION, "
          "T4_STALE_EVIDENCE.")


def card_toml(u: dict) -> str:
    ttype = u["target"]["type"]
    leg = "\ninterval_leg           = false" if u.get("pure_label") else ""
    return f'''# Track-4-format stress unit built locally from public pre-2022 data (track4_my/units_pre2022); not an official unit.
schema_version = "2.0"

[task]
id              = "{u["id"]}"
track           = "analysis"
title           = {json.dumps(u["title"])}
split           = "public-dev"
family          = "{u["family"]}"
target_type     = "{ttype}"
prompt          = {json.dumps(u["prompt"])}
cutoff_date     = "{u["cutoff"]}"
resolution_date = "{u["resolution"]}"
adversarial     = false
faithfulness_rubric = {json.dumps(RUBRIC.format(cutoff=u["cutoff"]))}

[metadata]
author_name              = "team9-local"
author_email             = "none@example.invalid"
difficulty               = "hard"
category                 = "{u["family"]}"
tags                     = ["analysis", "{ttype}", "pre2022", "stress"]
expert_time_estimate_min = 60.0
junior_time_estimate_min = 180.0

[provenance]
license             = {json.dumps(u.get("license", LICENSE))}
data_source         = {json.dumps(u.get("data_source", "FRED / ALFRED real-time vintages"))}
data_cutoff         = "{u["cutoff"]}"
public_release_date = "2026-10-07"
redistributable     = true
manifest            = "manifest.json"

[contamination]
canary_guid = "{uuid.UUID(bytes=hashlib.md5(("track4-local/" + u["id"]).encode()).digest(), version=4)}"

[scoring]
verifier            = "t4.faithful_analysis"
metric              = "analysis_composite"
admissibility_gates = ["g0_integrity", "g1_schema", "g2_cutoff_resource", "g3_domain_semantics"]

[scoring.params]
faithfulness_threshold = 0.80{leg}
interval_level         = 0.90
target_type            = "{ttype}"
composite_weights      = [0.7, 0.3]
tau_citation           = 0.5
nli_judge_ensemble     = ["cross-encoder/nli-deberta-v3-large", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"]

[environment]
cpus    = 16
memory  = "128G"
gpu     = true
network = "restricted"

[agent]
timeout_sec = 600.0

[verifier]
timeout_sec = 600.0

[corpus]
pii_stripped      = true
manifest_required  = true
manifest_path     = "corpus/manifest.json"

[embargo]
cutoff_field   = "cutoff_date"
doc_date_field = "doc_date"
strict         = true
'''


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_unit(u: dict, out: Path = OUT) -> Path:
    """units/<id>/{task.json, card.toml, manifest.json, corpus/*.json, corpus/manifest.json};
    reference/<id>/{outcome.json, naive_answer.json}."""
    udir, rdir = out / "units" / u["id"], out / "reference" / u["id"]
    for d in (udir, rdir):
        if d.exists():
            shutil.rmtree(d)
    (udir / "corpus").mkdir(parents=True)
    rdir.mkdir(parents=True)
    entries = []
    for doc, ents in u["docs"]:
        assert doc["doc_date"] <= u["cutoff"], (doc["doc_id"], doc["doc_date"])
        p = udir / "corpus" / f"{doc['doc_id']}.json"
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        e = {"path": f"corpus/{p.name}", "role": "corpus", "source": doc["source"], "license": u.get("license", LICENSE),
             "sha256": _sha(p), "bytes": p.stat().st_size, "split": "public-dev", "cutoff": u["cutoff"],
             "redistributable": True, "pii_stripped": True}
        e.update({"shared": True} if ents == "shared" else {"entity_ids": list(ents)})
        entries.append(e)
    cm = {"manifest_version": "2.0", "unit_id": u["id"], "generated_by": "tools/build_pre2022_units.py",
          "files": entries}
    (udir / "corpus" / "manifest.json").write_text(json.dumps(cm, indent=1))
    target = {"name": u["target"]["name"], "type": u["target"]["type"]}
    if u["target"].get("labels"):
        target["labels"] = u["target"]["labels"]
        target["label_assertions"] = u["target"]["label_assertions"]
    task = {"task_id": u["id"], "schema_version": "3", "family": u["family"], "target": target, "prompt": u["prompt"],
            "cutoff_date": u["cutoff"], "resolution_date": u["resolution"], "interval_level": 0.9,
            "corpus_manifest": "corpus/manifest.json", "faithfulness_rubric": RUBRIC.format(cutoff=u["cutoff"]),
            "entities": u["entities"], "notes": "Local stress unit from public pre-2022 data; not an official unit."}
    (udir / "task.json").write_text(json.dumps(task, indent=1, ensure_ascii=False))
    (udir / "card.toml").write_text(card_toml(u))
    root = []
    for name in ("card.toml", "task.json"):
        p = udir / name
        root.append({"path": name, "role": "input", "license": u.get("license", LICENSE), "sha256": _sha(p),
                     "bytes": p.stat().st_size, "split": "public-dev", "redistributable": True})
    root += entries
    p = udir / "corpus" / "manifest.json"
    root.append({"path": "corpus/manifest.json", "role": "corpus", "license": u.get("license", LICENSE),
                 "sha256": _sha(p), "bytes": p.stat().st_size, "split": "public-dev", "redistributable": True})
    (udir / "manifest.json").write_text(json.dumps({"manifest_version": "2.0", "unit_id": u["id"],
                                                     "generated_by": "tools/build_pre2022_units.py", "files": root},
                                                    indent=1))
    (rdir / "outcome.json").write_text(json.dumps({"unit_id": u["id"], "outcomes": u["outcomes"]}, indent=1))
    preds = []
    for n in u["naive"]:
        doc = next(d for d, ents in u["docs"] if ents == "shared" or n["entity_id"] in ents)
        claim = {"doc_id": doc["doc_id"], "span_start": 0, "span_end": min(80, len(doc["text"])),
                 "claim": doc["text"][:min(80, len(doc["text"]))]}
        row = {"entity_id": n["entity_id"], "point_forecast": n["point"],
               "interval": {"level": 0.9, "lo": n["lo"], "hi": n["hi"]}, "claims": [claim]}
        if "label" in n:
            row["label"] = n["label"]
        preds.append(row)
    naive = {"task_id": u["id"], "schema_version": "3", "target_type": u["target"]["type"], "entity_predictions": preds,
             "evidence_trace": "declared naive rule", "notes": {"baseline_id": u["naive_rule"]}}
    (rdir / "naive_answer.json").write_text(json.dumps(naive, indent=1))
    return udir


def naive_row(eid: str, point: float, half: float, label: str | None = None) -> dict:
    half = max(half, 1e-6)
    r = {"entity_id": eid, "point": round(point, 6), "lo": round(point - half, 6), "hi": round(point + half, 6)}
    if label is not None:
        r["label"] = label
    return r


# ------------------------------------------------------------------------------------------------ units
def u_nfp() -> dict:
    """Payroll changes in thousands (first print), eight CES series, May 2019."""
    cutoff, ref = "2019-06-05", "2019-05-01"
    series = ["PAYEMS", "USPRIV", "MANEMP", "USCONS", "USTRADE", "USGOVT", "USEHS", "USLAH"]
    ents, docs, outs, naive = [], [], [], []
    for sid in series:
        info = series_info(sid)
        vints, release = vintages_around(sid, cutoff)
        cur = dict(obs(sid, "2016-01-01", cutoff, asof=vints[-1]))
        months = sorted(cur)
        lm, pm = months[-1], months[-2]
        rel = dict(obs(sid, "2016-01-01", "2019-12-31", asof=release))
        assert ref in rel and lm == "2019-04-01", (sid, lm)
        y = rel[ref] - rel[lm]
        latest_change = cur[lm] - cur[pm]
        hist = [cur[b] - cur[a] for a, b in zip(months[-25:-1], months[-24:])]
        diffs = [b - a for a, b in zip(hist, hist[1:])]
        eid = f"{sid}_2019-05"
        ents.append({"entity_id": eid, "series_id": sid, "series_name": info["title"],
                     "units": "thousands of persons, seasonally adjusted (target: change in thousands)",
                     "ref_month": "2019-05", "latest_level_2019_04": cur[lm], "latest_change_2019_04": round(latest_change, 1),
                     "latest_vintage": vints[-1], "resolving_release_date": release, "corpus_ref": "corpus/"})
        docs.append((vintage_doc(sid, info, cutoff, vints, 12), [eid]))
        outs.append({"entity_id": eid, "y": round(y, 1)})
        naive.append(naive_row(eid, latest_change, Z90 * sd(diffs)))
    info = series_info("ICSA")
    docs.append((series_doc("ICSA", info, cutoff, 16, "2018-06-01", asof=cutoff), "shared"))
    return {"id": "t4-x-nfp-201906", "family": "payroll_change_first_print", "title":
            "Predict the first-print May 2019 employment change (thousands) for eight CES payroll series (regression)",
            "target": {"name": "payroll_change_thousands_first_print", "type": "regression"},
            "prompt": "Using ONLY the frozen evidence corpus (ALFRED real-time vintage tables for eight Current "
                      "Employment Statistics payroll series and the weekly initial jobless claims table, all as known on "
                      "or before the 2019-06-05 cutoff), predict for each row the over-the-month change in employment "
                      "for May 2019, in THOUSANDS of persons, as first published in the Employment Situation release of "
                      "resolving_release_date (the May level minus the April level, both as published that day). Give "
                      "a point forecast and a 90% interval in thousands, and cite corpus passages.",
            "cutoff": cutoff, "resolution": "2019-06-07", "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "last_change_carried_forward"}


def u_unrate() -> dict:
    """Unemployment-rate LEVELS in percent for eight demographic series, September 2018."""
    cutoff, ref = "2018-10-03", "2018-09-01"
    series = ["UNRATE", "U6RATE", "LNS14000001", "LNS14000002", "LNS14000003", "LNS14000006", "LNS14000009",
              "LNS14000012"]
    ents, docs, outs, naive = [], [], [], []
    for sid in series:
        info = series_info(sid)
        vints, release = vintages_around(sid, cutoff, 3)
        cur = obs(sid, "2016-01-01", cutoff, asof=vints[-1])
        rel = dict(obs(sid, "2016-01-01", "2018-12-31", asof=release))
        assert cur[-1][0] == "2018-08-01" and ref in rel, sid
        vals = [v for _, v in cur]
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "series_name": info["title"], "units": "percent, seasonally "
                     "adjusted (target: the level)", "ref_month": "2018-09", "latest_value_2018_08": cur[-1][1],
                     "resolving_release_date": release, "corpus_ref": "corpus/"})
        docs.append((series_doc(sid, info, cutoff, 24, "2015-01-01", doc_id=f"ALFRED_{sid}_{ymd(cutoff)}",
                                asof=vints[-1]), [eid]))
        outs.append({"entity_id": eid, "y": rel[ref]})
        naive.append(naive_row(eid, cur[-1][1], Z90 * sd([b - a for a, b in zip(vals[-25:-1], vals[-24:])])))
    return {"id": "t4-x-unrate-201810", "family": "unemployment_rate_level", "title":
            "Predict the September 2018 unemployment rate (percent level) for eight demographic series (regression)",
            "target": {"name": "unemployment_rate_pct_first_print", "type": "regression"},
            "prompt": "Using ONLY the frozen evidence corpus (unemployment-rate tables for eight Current Population "
                      "Survey series as published on or before the 2018-10-03 cutoff), predict for each row the "
                      "unemployment RATE for September 2018 in PERCENT (the level, e.g. 3.9, not a change), as first "
                      "published on resolving_release_date. Give a point forecast and a 90% interval in percent, and "
                      "cite corpus passages.",
            "cutoff": cutoff, "resolution": "2018-10-05", "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "last_value"}


def u_fx() -> dict:
    """Three-way FX moves against the dollar over May 2019, with quotes in both conventions."""
    cutoff, res, thr = "2019-05-01", "2019-05-31", 1.0
    pairs = [("EUR", "DEXUSEU", True), ("JPY", "DEXJPUS", False), ("GBP", "DEXUSUK", True), ("CAD", "DEXCAUS", False),
             ("AUD", "DEXUSAL", True), ("CHF", "DEXSZUS", False), ("CNY", "DEXCHUS", False), ("MXN", "DEXMXUS", False)]
    labels = ["appreciate", "stable", "depreciate"]
    ents, docs, outs, naive = [], [], [], []
    for ccy, sid, usd_per in pairs:
        info = series_info(sid)
        xs = obs(sid, "2016-01-01", "2019-06-30")
        q0 = [v for d, v in xs if d <= cutoff][-1]
        q1 = [v for d, v in xs if d <= res][-1]
        val = (lambda q: q) if usd_per else (lambda q: 1.0 / q)
        y = 100 * (val(q1) / val(q0) - 1)
        lab = "appreciate" if y > thr else "depreciate" if y < -thr else "stable"
        monthly = {}
        for d, v in xs:
            if d <= cutoff:
                monthly[d[:7]] = v
        ms = sorted(monthly)[-25:]
        chg = [100 * (val(monthly[b]) / val(monthly[a]) - 1) for a, b in zip(ms, ms[1:])]
        eid = f"{ccy}USD"
        ents.append({"entity_id": eid, "currency": ccy, "fred_series": sid, "quote_convention":
                     "U.S. dollars per 1 " + ccy if usd_per else ccy + " per 1 U.S. dollar",
                     "rate_at_cutoff": q0, "as_of": cutoff, "threshold_pct": thr,
                     "unit": f"percent change of the value of 1 {ccy} in U.S. dollars", "corpus_ref": "corpus/"})
        docs.append((series_doc(sid, info, cutoff, 45, "2017-01-01"), [eid]))
        outs.append({"entity_id": eid, "true_label": lab, "y": round(y, 4)})
        naive.append(naive_row(eid, 0.0, Z90 * sd(chg), "stable"))
    info = series_info("DTWEXB")
    docs.append((series_doc("DTWEXB", info, cutoff, 45, "2017-01-01"), "shared"))
    return {"id": "t4-x-fx-201905", "family": "fx_move_three_way", "title":
            "Predict whether eight currencies appreciate, stay stable or depreciate against the U.S. dollar over May 2019",
            "target": {"name": "currency_move_vs_usd_1m", "type": "classification", "labels": labels,
                       "label_assertions": {
                           "appreciate": f"the currency's value in U.S. dollars rises by more than {thr}% from the "
                                         f"{cutoff} noon buying rate to the {res} rate",
                           "stable": f"the currency's value in U.S. dollars changes by {thr}% or less in either "
                                     f"direction between {cutoff} and {res}",
                           "depreciate": f"the currency's value in U.S. dollars falls by more than {thr}% from "
                                         f"{cutoff} to {res}"}},
            "prompt": f"Using ONLY the frozen evidence corpus (daily H.10 noon buying rates from FRED for eight "
                      f"currencies and the broad trade-weighted dollar index, through the {cutoff} cutoff), predict for "
                      f"each currency whether its value in U.S. DOLLARS will APPRECIATE (rise by more than "
                      f"threshold_pct percent), stay STABLE, or DEPRECIATE (fall by more than threshold_pct percent) "
                      f"between the {cutoff} rate and the {res} rate. Mind quote_convention: for a rate quoted as "
                      f"currency per dollar, a HIGHER quote means the currency is WORTH LESS. For each row also give a "
                      f"point forecast of that percent change of the currency's dollar value with a 90% interval, and "
                      f"cite corpus passages.",
            "cutoff": cutoff, "resolution": res, "entities": ents, "docs": docs, "outcomes": outs, "naive": naive,
            "naive_rule": "stable_zero_change"}


def u_energy() -> dict:
    """Energy prices in dollars, one week ahead (daily and weekly series)."""
    cutoff, target_day = "2017-06-05", "2017-06-12"
    series = ["DCOILWTICO", "DCOILBRENTEU", "DHHNGSP", "GASREGW", "GASDESW"]
    ents, docs, outs, naive = [], [], [], []
    for sid in series:
        info = series_info(sid)
        xs = obs(sid, "2015-06-01", "2017-07-31")
        past = [(d, v) for d, v in xs if d <= cutoff]
        tgt = [v for d, v in xs if d == target_day]
        assert tgt, sid
        weekly = info["frequency_short"] == "W"
        vals = [v for _, v in past]
        step, n = (1, 52) if weekly else (5, 252)  # one-week changes over the past year
        chg = [vals[i] - vals[i - step] for i in range(len(vals) - n, len(vals))]
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "name": info["title"], "units": info["units"],
                     "frequency": info["frequency"], "latest_value": past[-1][1], "latest_date": past[-1][0],
                     "target_date": target_day, "corpus_ref": "corpus/"})
        docs.append((series_doc(sid, info, cutoff, 26 if weekly else 40, "2015-06-01"), [eid]))
        outs.append({"entity_id": eid, "y": tgt[0]})
        naive.append(naive_row(eid, past[-1][1], Z90 * sd(chg)))
    return {"id": "t4-x-energy-201706", "family": "energy_price_week_ahead", "title":
            "Predict five U.S. energy prices (dollars) one week ahead, 2017-06-12 (regression)",
            "target": {"name": "price_level_on_target_date", "type": "regression"},
            "prompt": "Using ONLY the frozen evidence corpus (FRED daily spot prices for WTI and Brent crude oil and "
                      "Henry Hub natural gas, and the weekly U.S. retail gasoline and diesel prices, through the "
                      "2017-06-05 cutoff), predict for each row the price on target_date in the row's own units "
                      "(dollars per barrel, per million Btu, or per gallon). For a weekly series, the target is the "
                      "observation dated target_date. Give a point forecast and a 90% interval in those units, and cite "
                      "corpus passages.",
            "cutoff": cutoff, "resolution": target_day, "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "last_value"}


STATES = ("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH "
          "OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY").split()


def u_states() -> dict:
    """Ranking of 51 state unemployment-rate changes (first print), July 2017."""
    cutoff, ref, prev = "2017-08-16", "2017-07-01", "2017-06-01"
    ents, docs, outs, naive = [], [], [], []
    release = None
    for stt in STATES:
        sid = f"{stt}UR"
        info = series_info(sid)
        vints, rel_date = vintages_around(sid, cutoff, 2)
        release = release or rel_date
        cur = obs(sid, "2015-01-01", cutoff, asof=vints[-1])
        rel = dict(obs(sid, "2015-01-01", "2017-12-31", asof=rel_date))
        assert cur[-1][0] == prev and ref in rel, (sid, cur[-1][0])
        vals = [v for _, v in cur]
        last_chg = vals[-1] - vals[-2]
        hist = [b - a for a, b in zip(vals[-25:-1], vals[-24:])]
        eid = sid
        ents.append({"entity_id": eid, "state": stt, "series_id": sid, "series_name": info["title"],
                     "units": "percent, seasonally adjusted", "latest_rate_2017_06": vals[-1],
                     "latest_change_2017_06_pp": round(last_chg, 2), "unit": "percentage-point change",
                     "corpus_ref": "corpus/"})
        docs.append((series_doc(sid, info, cutoff, 18, "2014-01-01", doc_id=f"ALFRED_{sid}_{ymd(cutoff)}",
                                asof=vints[-1]), [eid]))
        outs.append({"entity_id": eid, "y": round(rel[ref] - rel[prev], 2)})
        naive.append(naive_row(eid, last_chg, Z90 * sd(hist)))
    info = series_info("UNRATE")
    docs.append((series_doc("UNRATE", info, cutoff, 18, "2014-01-01", doc_id=f"ALFRED_UNRATE_{ymd(cutoff)}",
                            asof=cutoff), "shared"))
    return {"id": "t4-x-stateur-201708", "family": "state_unemployment_change_rank", "title":
            "Rank the 50 states and DC by the change in their unemployment rate, June to July 2017 (ranking)",
            "target": {"name": "unemployment_rate_change_pp_rank", "type": "ranking"},
            "prompt": "Using ONLY the frozen evidence corpus (state unemployment-rate tables from ALFRED as published "
                      "on or before the 2017-08-16 cutoff, and the national rate), order the 51 rows by the change in "
                      "the seasonally adjusted unemployment rate from June 2017 to July 2017, in percentage points, as "
                      "first published in the state release after the cutoff (the July rate minus the June rate, both "
                      "as published that day). For each row put the predicted CHANGE in percentage points in the point "
                      "forecast (the ordering is read from it: larger change ranks higher) with a 90% interval on that "
                      "change, and cite corpus passages.",
            "cutoff": cutoff, "resolution": release, "entities": ents, "docs": docs, "outcomes": outs, "naive": naive,
            "naive_rule": "last_change_carried_forward"}


def u_spreads() -> dict:
    """Basis-point changes of percent spreads over a quarter, with a statement in the corpus."""
    cutoff, res = "2019-06-28", "2019-09-30"
    series = ["T10Y2Y", "T10Y3M", "BAA10Y", "AAA10Y", "TEDRATE"]
    ents, docs, outs, naive = [], [], [], []
    for sid in series:
        info = series_info(sid)
        xs = obs(sid, "2014-01-01", "2019-10-31")
        past = [(d, v) for d, v in xs if d <= cutoff]
        end = [v for d, v in xs if d <= res][-1]
        vals = [v for _, v in past]
        chg = [100 * (b - a) for a, b in zip(vals[-1260:-63], vals[-1197:])]
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "name": info["title"], "start_spread_pct": past[-1][1],
                     "as_of": past[-1][0], "resolution_close": res, "unit": "bps_change", "corpus_ref": "corpus/"})
        docs.append((series_doc(sid, info, cutoff, 40, "2017-01-01"), [eid]))
        outs.append({"entity_id": eid, "y": round(100 * (end - past[-1][1]), 2)})
        naive.append(naive_row(eid, 0.0, Z90 * sd(chg)))
    docs.append((text_doc("FOMC_STATEMENT_20190619", "https://www.federalreserve.gov/newsevents/pressreleases/"
                          "monetary20190619a.htm", "2019-06-19", "FOMC statement, June 19, 2019", "fomc_statement"),
                 "shared"))
    return {"id": "t4-x-spreads-201909", "family": "spread_change_quarter", "title":
            "Predict the change in five U.S. yield and credit spreads (basis points) over 2019Q3 (regression)",
            "target": {"name": "spread_change_bps_quarter", "type": "regression"},
            "prompt": "Using ONLY the frozen evidence corpus (FRED daily spread series through the 2019-06-28 cutoff "
                      "and the June 19, 2019 FOMC statement), predict for each row the change in the spread from its "
                      "value on the as_of date to its close on resolution_close, in BASIS POINTS (the spreads are "
                      "quoted in percent: a move from 0.25 to 0.10 is -15 bps). Give a point forecast and a 90% "
                      "interval in basis points, and cite corpus passages.",
            "cutoff": cutoff, "resolution": res, "entities": ents, "docs": docs, "outcomes": outs, "naive": naive,
            "naive_rule": "zero_change"}


def u_housing() -> dict:
    """Housing starts (thousands, SAAR) first print, September 2018 -- the table omits the latest value."""
    cutoff, ref = "2018-10-15", "2018-09-01"
    series = ["HOUST", "HOUST1F", "HOUST5F", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW"]
    ents, docs, outs, naive = [], [], [], []
    for sid in series:
        info = series_info(sid)
        vints, release = vintages_around(sid, cutoff)
        cur = obs(sid, "2016-01-01", cutoff, asof=vints[-1])
        rel = dict(obs(sid, "2016-01-01", "2018-12-31", asof=release))
        assert cur[-1][0] == "2018-08-01" and ref in rel, sid
        vals = [v for _, v in cur]
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "series_name": info["title"], "units":
                     "thousands of units, seasonally adjusted annual rate", "ref_month": "2018-09",
                     "resolving_release_date": release, "corpus_ref": "corpus/"})
        docs.append((vintage_doc(sid, info, cutoff, vints, 12), [eid]))
        outs.append({"entity_id": eid, "y": rel[ref]})
        naive.append(naive_row(eid, vals[-1], Z90 * sd([b - a for a, b in zip(vals[-25:-1], vals[-24:])])))
    pinfo = series_info("PERMIT")
    pv, _ = vintages_around("PERMIT", cutoff, 2)
    docs.append((vintage_doc("PERMIT", pinfo, cutoff, pv, 12), "shared"))
    return {"id": "t4-x-housing-201810", "family": "housing_starts_first_print", "title":
            "Predict first-print September 2018 housing starts (thousands, SAAR) for seven series (regression)",
            "target": {"name": "housing_starts_thousands_saar_first_print", "type": "regression"},
            "prompt": "Using ONLY the frozen evidence corpus (ALFRED real-time vintage tables for seven housing-starts "
                      "series and building permits, as known on or before the 2018-10-15 cutoff), predict for each "
                      "row the September 2018 value in THOUSANDS of units at a seasonally adjusted annual rate, as "
                      "first published in the New Residential Construction release of resolving_release_date. The "
                      "latest published values are in the corpus tables, not in the task table. Give a point forecast "
                      "and a 90% interval in those units, and cite corpus passages.",
            "cutoff": cutoff, "resolution": "2018-10-17", "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "latest_estimate"}


def u_fomc() -> dict:
    """Decisions at the next four FOMC meetings from text (statements and minutes) plus two rate tables."""
    cutoff = "2019-07-29"
    meetings = ["2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11"]
    labels = ["cut", "hold", "hike"]
    upper = dict(obs("DFEDTARU", "2018-06-01", "2019-12-31"))
    ents, outs, naive = [], [], []
    for i, m in enumerate(meetings):
        before = upper[(date.fromisoformat(m) - timedelta(days=1)).isoformat()]
        after = upper[(date.fromisoformat(m) + timedelta(days=1)).isoformat()]
        y = round(100 * (after - before), 1)
        lab = "cut" if y < 0 else "hike" if y > 0 else "hold"
        eid = f"FOMC_{ymd(m)}"
        ents.append({"entity_id": eid, "meeting_date": m, "meetings_ahead": i + 1, "target_range_upper_at_cutoff_pct":
                     upper[cutoff], "unit": "bps_change of the target range upper bound at this meeting",
                     "corpus_ref": "corpus/"})
        outs.append({"entity_id": eid, "true_label": lab, "y": y})
        naive.append(naive_row(eid, 0.0, 25.0, "hold"))
    docs = []
    for d, slug in (("2018-12-19", "20181219"), ("2019-01-30", "20190130"), ("2019-03-20", "20190320"),
                    ("2019-05-01", "20190501"), ("2019-06-19", "20190619")):
        docs.append((text_doc(f"FOMC_STATEMENT_{slug}", f"https://www.federalreserve.gov/newsevents/pressreleases/"
                              f"monetary{slug}a.htm", d, f"FOMC statement, {d}", "fomc_statement"), "shared"))
    docs.append((text_doc("FOMC_MINUTES_20190619", "https://www.federalreserve.gov/monetarypolicy/"
                          "fomcminutes20190619.htm", "2019-07-10", "Minutes of the FOMC meeting of June 18-19, 2019 "
                          "(released July 10, 2019)", "fomc_minutes"), "shared"))
    for sid in ("DTB3", "DGS2"):
        docs.append((series_doc(sid, series_info(sid), cutoff, 40, "2018-01-01"), "shared"))
    return {"id": "t4-x-fomc-201907", "family": "policy_rate_decision", "title":
            "Predict the FOMC's decision (cut / hold / hike) at each of its next four meetings, from July 2019",
            "target": {"name": "target_range_decision", "type": "classification", "labels": labels,
                       "label_assertions": {"cut": "the FOMC lowers the target range for the federal funds rate at "
                                                   "that meeting",
                                            "hold": "the FOMC leaves the target range unchanged at that meeting",
                                            "hike": "the FOMC raises the target range at that meeting"}},
            "prompt": "Using ONLY the frozen evidence corpus (FOMC statements from December 2018 to June 2019, the "
                      "minutes of the June 2019 meeting released July 10, and daily 3-month bill and 2-year Treasury "
                      "yields through the 2019-07-29 cutoff), predict for each scheduled FOMC meeting whether the "
                      "Committee will CUT, HOLD or HIKE the target range for the federal funds rate at that meeting "
                      "(each row is one meeting's own decision, not a cumulative path). For each row also give a point "
                      "forecast of that meeting's change in the upper bound in basis points with a 90% interval, and "
                      "cite corpus passages.",
            "cutoff": cutoff, "resolution": meetings[-1], "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "hold_zero_change", "license": LICENSE_FED + "; " + LICENSE,
            "data_source": "federalreserve.gov statements and minutes; FRED DTB3, DGS2, DFEDTARU (outcome only)"}


def u_claims() -> dict:
    """Weekly claims, first print, rows in different units (counts and a percent)."""
    cutoff = "2019-04-17"
    rows = [("ICSA", "2019-04-13"), ("ICNSA", "2019-04-13"), ("CCSA", "2019-04-06"), ("IURSA", "2019-04-06")]
    ents, docs, outs, naive = [], [], [], []
    for sid, wk in rows:
        info = series_info(sid)
        vints, release = vintages_around(sid, cutoff, 2)
        cur = obs(sid, "2017-01-01", cutoff, asof=vints[-1])
        rel = dict(obs(sid, "2017-01-01", "2019-06-30", asof=release))
        assert wk in rel, (sid, wk, release)
        vals = [v for _, v in cur]
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "series_name": info["title"], "units": info["units"],
                     "seasonal_adjustment": info.get("seasonal_adjustment_short"), "week_ending": wk,
                     "latest_value": cur[-1][1], "latest_week_ending": cur[-1][0], "resolving_release_date": release,
                     "corpus_ref": "corpus/"})
        docs.append((series_doc(sid, info, cutoff, 26, "2017-01-01", doc_id=f"ALFRED_{sid}_{ymd(cutoff)}",
                                asof=vints[-1]), [eid]))
        outs.append({"entity_id": eid, "y": rel[wk]})
        naive.append(naive_row(eid, cur[-1][1], Z90 * sd([b - a for a, b in zip(vals[-53:-1], vals[-52:])])))
    return {"id": "t4-x-claims-201904", "family": "weekly_claims_first_print", "title":
            "Predict this week's unemployment-insurance claims figures (first print), April 18, 2019 (regression)",
            "target": {"name": "claims_value_first_print", "type": "regression"},
            "prompt": "Using ONLY the frozen evidence corpus (weekly unemployment-insurance claims tables as published "
                      "on or before the 2019-04-17 cutoff), predict for each row the value for the week ending "
                      "week_ending as first published on resolving_release_date, in the row's OWN units: initial and "
                      "continued claims are numbers of claims (seasonally adjusted or not, as the row says), the "
                      "insured unemployment rate is a percent. Give a point forecast and a 90% interval in those "
                      "units, and cite corpus passages.",
            "cutoff": cutoff, "resolution": "2019-04-18", "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "last_value"}


def u_ip() -> dict:
    """Industrial production: the percent change of an index level (first print), April 2019."""
    cutoff, ref = "2019-05-13", "2019-04-01"
    series = ["INDPRO", "IPMAN", "IPMINE", "IPUTIL", "IPCONGD", "IPBUSEQ", "IPMAT"]
    ents, docs, outs, naive = [], [], [], []
    for sid in series:
        info = series_info(sid)
        vints, release = vintages_around(sid, cutoff)
        cur = obs(sid, "2016-01-01", cutoff, asof=vints[-1])
        rel = dict(obs(sid, "2016-01-01", "2019-12-31", asof=release))
        assert cur[-1][0] == "2019-03-01" and ref in rel, sid
        vals = [v for _, v in cur]
        mom = [100 * (b / a - 1) for a, b in zip(vals[-26:-1], vals[-25:])]
        y = 100 * (rel[ref] / rel["2019-03-01"] - 1)
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "series_name": info["title"], "units":
                     "index 2012=100, seasonally adjusted (target: percent change)", "ref_month": "2019-04",
                     "latest_index_2019_03": vals[-1], "latest_mom_pct_2019_03": round(mom[-1], 3),
                     "resolving_release_date": release, "corpus_ref": "corpus/"})
        docs.append((vintage_doc(sid, info, cutoff, vints, 12), [eid]))
        outs.append({"entity_id": eid, "y": round(y, 4)})
        naive.append(naive_row(eid, mom[-1], Z90 * sd([b - a for a, b in zip(mom[:-1], mom[1:])])))
    return {"id": "t4-x-ip-201905", "family": "industrial_production_mom", "title":
            "Predict the first-print April 2019 month-over-month percent change in seven industrial production "
            "indexes (regression)",
            "target": {"name": "ip_index_mom_pct_first_print", "type": "regression"},
            "prompt": "Using ONLY the frozen evidence corpus (ALFRED real-time vintage tables of seven Federal Reserve "
                      "G.17 industrial production indexes as known on or before the 2019-05-13 cutoff), predict for "
                      "each row the PERCENT CHANGE of the index from March 2019 to April 2019, as first published in "
                      "the G.17 release of resolving_release_date (both months as published that day). Give a point "
                      "forecast and a 90% interval in percent, and cite corpus passages.",
            "cutoff": cutoff, "resolution": "2019-05-15", "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "last_change_carried_forward"}



# ------------------------------------------------------------------------------------------------ v21.2: labels set by numbers
def u_yield3c() -> dict:
    """Three-way yield moves over a month, stated as moves: rises / falls by more than 10 bp, or 10 bp or less in either
    direction (the label wording v21 misread)."""
    cutoff, res, thr = "2019-10-31", "2019-11-29", 10.0
    series = ["DGS3MO", "DGS2", "DGS5", "DGS10", "DGS30", "DFII10", "T10YIE"]
    labels = ["rise", "steady", "fall"]
    ents, docs, outs, naive = [], [], [], []
    for sid in series:
        info = series_info(sid)
        xs = obs(sid, "2016-01-01", "2019-12-31")
        past = [(d, v) for d, v in xs if d <= cutoff]
        v0, v1 = past[-1][1], [v for d, v in xs if d <= res][-1]
        y = round(100 * (v1 - v0), 1)
        lab = "rise" if y > thr else "fall" if y < -thr else "steady"
        vals = [v for _, v in past]
        chg = [100 * (vals[i] - vals[i - 21]) for i in range(len(vals) - 750, len(vals))]
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "name": info["title"], "start_yield_pct": v0, "as_of": past[-1][0],
                     "resolution_close": res, "threshold_bps": thr,
                     "unit": "bps change of the yield between as_of and resolution_close", "corpus_ref": "corpus/"})
        docs.append((series_doc(sid, info, cutoff, 45, "2018-01-01"), [eid]))
        outs.append({"entity_id": eid, "true_label": lab, "y": y})
        naive.append(naive_row(eid, 0.0, Z90 * sd(chg), "steady"))
    docs.append((text_doc("FOMC_STATEMENT_20191030", "https://www.federalreserve.gov/newsevents/pressreleases/"
                          "monetary20191030a.htm", "2019-10-30", "FOMC statement, October 30, 2019", "fomc_statement"),
                 "shared"))
    return {"id": "t4-x-yield3c-201911", "family": "yield_move_three_way", "title":
            "Predict whether seven U.S. yields rise by more than 10 bp, stay within 10 bp, or fall by more than 10 bp "
            "over November 2019 (classification)",
            "target": {"name": "yield_move_class_1m", "type": "classification", "labels": labels,
                       "label_assertions": {
                           "rise": f"the yield rises by more than {thr:g} basis points from its {cutoff} close to its "
                                   f"{res} close",
                           "steady": f"the yield changes by {thr:g} basis points or less in either direction between "
                                     f"its {cutoff} close and its {res} close",
                           "fall": f"the yield falls by more than {thr:g} basis points from its {cutoff} close to its "
                                   f"{res} close"}},
            "prompt": f"Using ONLY the frozen evidence corpus (FRED daily yield tables through the {cutoff} cutoff and the "
                      f"FOMC statement of October 30, 2019), predict for each row whether the yield will RISE by more "
                      f"than {thr:g} basis points, stay STEADY (a change of {thr:g} basis points or less in either "
                      f"direction), or FALL by more than {thr:g} basis points between its {cutoff} close and its {res} "
                      f"close. For each row also give a point forecast of that change in basis points with a 90% "
                      f"interval, and cite corpus passages.",
            "cutoff": cutoff, "resolution": res, "entities": ents, "docs": docs, "outcomes": outs, "naive": naive,
            "naive_rule": "steady_zero_change", "license": LICENSE_FED + "; " + LICENSE}


def u_cpithr() -> dict:
    """CPI components above or at/below a 0.2% month-over-month first print (an absolute threshold)."""
    cutoff, ref, prev, thr = "2019-06-10", "2019-05-01", "2019-04-01", 0.2
    series = ["CPIAUCSL", "CPILFESL", "CUSR0000SAH1", "CPIENGSL", "CPIUFDSL", "CUSR0000SETA02", "CPIAPPSL", "CPIMEDSL"]
    labels = ["above", "at_or_below"]
    ents, docs, outs, naive = [], [], [], []
    for sid in series:
        info = series_info(sid)
        vints, release = vintages_around(sid, cutoff)
        cur = obs(sid, "2016-01-01", cutoff, asof=vints[-1])
        rel = dict(obs(sid, "2016-01-01", "2019-12-31", asof=release))
        assert cur[-1][0] == prev and ref in rel and prev in rel, (sid, cur[-1][0])
        vals = [v for _, v in cur]
        mom = [100 * (b / a - 1) for a, b in zip(vals[-26:-1], vals[-25:])]
        y = round(100 * (rel[ref] / rel[prev] - 1), 3)
        lab = "above" if y > thr else "at_or_below"
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "name": info["title"], "ref_month": "2019-05",
                     "latest_published_mom_pct": round(mom[-1], 3), "latest_published_ref_month": "2019-04",
                     "threshold_mom_pct": thr, "resolving_release_date": release, "unit": "mom_pct_change_sa",
                     "corpus_ref": "corpus/"})
        docs.append((vintage_doc(sid, info, cutoff, vints, 12), [eid]))
        outs.append({"entity_id": eid, "true_label": lab, "y": y})
        naive.append(naive_row(eid, mom[-1], Z90 * sd([b - a for a, b in zip(mom[:-1], mom[1:])]),
                               "above" if mom[-1] > thr else "at_or_below"))
    docs.append((series_doc("GASREGW", series_info("GASREGW"), cutoff, 26, "2017-01-01", asof=cutoff), "shared"))
    return {"id": "t4-x-cpithr-201906", "family": "cpi_component_threshold", "title":
            "Predict whether eight CPI components print above 0.2% month over month for May 2019 (classification)",
            "target": {"name": "cpi_mom_above_threshold", "type": "classification", "labels": labels,
                       "label_assertions": {
                           "above": f"the component's seasonally adjusted month-over-month percent change for May 2019, "
                                    f"as first published, is above {thr}%",
                           "at_or_below": f"the component's seasonally adjusted month-over-month percent change for May "
                                          f"2019, as first published, is {thr}% or below"}},
            "prompt": f"Using ONLY the frozen evidence corpus (ALFRED real-time vintage tables of eight CPI-U series and "
                      f"the weekly retail gasoline price, as known on or before the {cutoff} cutoff), predict for each "
                      f"row whether the seasonally adjusted month-over-month percent change for May 2019, as first "
                      f"published on resolving_release_date, will be ABOVE {thr}% or AT_OR_BELOW it. For each row also "
                      f"give a point forecast of that percent change with a 90% interval, and cite corpus passages.",
            "cutoff": cutoff, "resolution": "2019-06-12", "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "last_print_carried_forward"}


def u_claimsdir() -> dict:
    """State initial claims (NSA) up or down against the previous week: a direction stated against a reference."""
    cutoff, wk, prev = "2019-08-07", "2019-07-27", "2019-07-20"  # state weeks are published a week after the nation's
    states = ["CA", "TX", "NY", "FL", "PA", "OH", "IL", "MI", "GA", "NJ"]
    labels = ["up", "down"]
    ents, docs, outs, naive = [], [], [], []
    release = "2019-08-08"
    for stt in states:
        sid = f"{stt}ICLAIMS"
        try:
            info = series_info(sid)
            vints, release = vintages_around(sid, cutoff, 2)
            cur = obs(sid, "2017-01-01", cutoff, asof=vints[-1])
            rel = dict(obs(sid, "2017-01-01", "2019-12-31", asof=release))
        except Exception:  # noqa: BLE001 - a series without vintages is read as published now
            info = series_info(sid)
            allx = obs(sid, "2017-01-01", "2019-12-31")
            cur = [(d, v) for d, v in allx if d <= prev]
            rel = dict(allx)
        if cur[-1][0] != prev or wk not in rel or prev not in rel:
            continue
        vals = [v for _, v in cur]
        y = rel[wk] - rel[prev]
        last_chg = vals[-1] - vals[-2]
        eid = sid
        ents.append({"entity_id": eid, "state": stt, "series_id": sid, "name": info["title"],
                     "latest_week_ending": prev, "latest_claims": vals[-1], "target_week_ending": wk,
                     "unit": "change in the number of initial claims from the previous week, not seasonally adjusted",
                     "corpus_ref": "corpus/"})
        docs.append((series_doc(sid, info, cutoff, 26, "2017-01-01", doc_id=f"FRED_{sid}_{ymd(cutoff)}"), [eid]))
        outs.append({"entity_id": eid, "true_label": "up" if y > 0 else "down", "y": y})
        diffs = [b - a for a, b in zip(vals[-53:-1], vals[-52:])]
        naive.append(naive_row(eid, last_chg, Z90 * sd(diffs), "up" if last_chg > 0 else "down"))
    return {"id": "t4-x-claimsdir-201908", "family": "state_claims_direction", "title":
            "Predict whether initial unemployment claims rose or fell week over week in ten states, week ending "
            "2019-07-27 (classification)",
            "target": {"name": "claims_direction_wow", "type": "classification", "labels": labels,
                       "label_assertions": {
                           "up": f"the state's initial claims for the week ending {wk}, as first published, are higher "
                                 f"than for the week ending {prev}",
                           "down": f"the state's initial claims for the week ending {wk}, as first published, are lower "
                                   f"than or equal to those for the week ending {prev}"}},
            "prompt": f"Using ONLY the frozen evidence corpus (weekly state initial-claims tables, not seasonally "
                      f"adjusted, through the {cutoff} cutoff), predict for each state whether initial claims for the "
                      f"week ending {wk} will be UP or DOWN against the week ending {prev}. For each row also give a "
                      f"point forecast of the week-over-week change in claims with a 90% interval, and cite corpus "
                      f"passages.",
            "cutoff": cutoff, "resolution": release, "entities": ents, "docs": docs, "outcomes": outs,
            "naive": naive, "naive_rule": "last_direction_carried_forward"}


def u_rsband() -> dict:
    """Retail sales components strong / steady / weak, the bands stated as condition columns of the entity table."""
    cutoff, ref, prev, band = "2019-08-13", "2019-07-01", "2019-06-01", 0.5
    series = ["RSAFS", "RSXFS", "RSFSDP", "RSGASS", "RSNSR", "RSHPCS", "RSCCAS", "RSMVPD"]
    labels = ["strong", "steady", "weak"]
    ents, docs, outs, naive = [], [], [], []
    release = None
    for sid in series:
        info = series_info(sid)
        vints, rel_date = vintages_around(sid, cutoff)
        release = release or rel_date
        cur = obs(sid, "2016-01-01", cutoff, asof=vints[-1])
        rel = dict(obs(sid, "2016-01-01", "2019-12-31", asof=rel_date))
        assert cur[-1][0] == prev and ref in rel and prev in rel, (sid, cur[-1][0])
        vals = [v for _, v in cur]
        mom = [100 * (b / a - 1) for a, b in zip(vals[-26:-1], vals[-25:])]
        y = round(100 * (rel[ref] / rel[prev] - 1), 3)
        lab = "strong" if y > band else "weak" if y < -band else "steady"
        eid = sid
        ents.append({"entity_id": eid, "series_id": sid, "name": info["title"], "ref_month": "2019-07",
                     "latest_mom_pct": round(mom[-1], 3), "strong_condition": f"mom_pct > {band}",
                     "weak_condition": f"mom_pct < {-band}", "resolving_release_date": rel_date,
                     "unit": "mom_pct_change_sa (advance estimate)", "corpus_ref": "corpus/"})
        docs.append((vintage_doc(sid, info, cutoff, vints, 12), [eid]))
        outs.append({"entity_id": eid, "true_label": lab, "y": y})
        naive.append(naive_row(eid, mom[-1], Z90 * sd([b - a for a, b in zip(mom[:-1], mom[1:])]),
                               "strong" if mom[-1] > band else "weak" if mom[-1] < -band else "steady"))
    return {"id": "t4-x-rsband-201908", "family": "retail_sales_band", "title":
            "Predict whether eight retail-sales components print strong, steady or weak for July 2019 (classification)",
            "target": {"name": "retail_sales_band", "type": "classification", "labels": labels,
                       "label_assertions": {"strong": "the row's strong_condition holds for the advance estimate",
                                            "steady": "neither the strong_condition nor the weak_condition holds",
                                            "weak": "the row's weak_condition holds for the advance estimate"}},
            "prompt": f"Using ONLY the frozen evidence corpus (ALFRED real-time vintage tables of eight retail-sales "
                      f"series as known on or before the {cutoff} cutoff), predict for each row the band of the "
                      f"seasonally adjusted month-over-month percent change (mom_pct) for July 2019 in the advance "
                      f"estimate published on resolving_release_date: STRONG when the row's strong_condition holds, "
                      f"WEAK when its weak_condition holds, STEADY otherwise. For each row also give a point forecast "
                      f"of mom_pct with a 90% interval, and cite corpus passages.",
            "cutoff": cutoff, "resolution": release, "entities": ents, "docs": docs, "outcomes": outs, "naive": naive,
            "naive_rule": "last_print_carried_forward"}


BUILDERS = {"t4-x-nfp-201906": u_nfp, "t4-x-unrate-201810": u_unrate, "t4-x-fx-201905": u_fx, "t4-x-energy-201706": u_energy,
            "t4-x-stateur-201708": u_states, "t4-x-spreads-201909": u_spreads, "t4-x-housing-201810": u_housing,
            "t4-x-fomc-201907": u_fomc, "t4-x-claims-201904": u_claims, "t4-x-ip-201905": u_ip,
            "t4-x-yield3c-201911": u_yield3c, "t4-x-cpithr-201906": u_cpithr, "t4-x-claimsdir-201908": u_claimsdir,
            "t4-x-rsband-201908": u_rsband}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    names = [n for n in args.only.split(",") if n] or list(BUILDERS)
    for n in names:
        try:
            u = BUILDERS[n]()
            udir = write_unit(u, out)
            ys = [o.get("y") for o in u["outcomes"]]
            print(f"OK   {n:18s} {len(u['entities']):2d} entities, {len(u['docs']):2d} docs, "
                  f"cutoff {u['cutoff']} -> {u['resolution']}; truth {['%.4g' % y for y in ys if y is not None][:6]}"
                  f"{' labels ' + str([o['true_label'] for o in u['outcomes']]) if u['target']['type'] == 'classification' else ''}")
        except Exception as exc:  # noqa: BLE001 - report and go on
            print(f"FAIL {n:18s} {type(exc).__name__}: {str(exc)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
