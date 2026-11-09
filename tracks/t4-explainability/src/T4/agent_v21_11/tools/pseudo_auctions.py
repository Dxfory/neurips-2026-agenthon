"""How much does the model's own forecast beat the statistical baseline? A pre-2022 pseudo-unit test (development only).

The agent blends every regression forecast 50/50 with its statistical baseline (`config.STAT_WEIGHT`). That weight is
the literature's untuned default; replays of the baselines alone (tools/historical_skill.py) say they are about as good
as the naive rule, so the blend mostly shrinks the model halfway to it. Whether that helps depends on the MODEL's
accuracy, which no baseline replay measures. This builds units in the Development auction unit's own format from
TreasuryDirect results published 2016-2019 (../calib_data/tdirect_*.json, public domain), runs the agent with the blend
switched off (T4_STAT_WEIGHT=0: its points are the model's own), recomputes the baseline offline with the same code,
and scores the blend weights w in {0, 0.25, 0.5, 0.75, 1} with scorer 5.2.2's predictive quality against two naive
rules (the tenor's last bid-to-cover; its six-auction mean).

PRE-REGISTERED DECISION (written before any run): STAT_WEIGHT changes from 0.5 only if 0.25 or 0.75 raises the mean
predictive quality over the pseudo-units by at least 0.01 under both naive rules AND wins in at least two thirds of
the pseudo-units; otherwise it stays.

Caveats: the House model may remember the period's events (the cutoffs avoid the 2016 election and 2020); exact
bid-to-cover ratios are obscure enough that recall is unlikely but not impossible. The units carry no narrative
context (the Development unit's announcement text mentions the election and the FOMC), so the model has less to work
with than in a real unit.

    ../.venv/bin/python tools/pseudo_auctions.py build     # write the units (no model calls)
    ../.venv/bin/python tools/pseudo_auctions.py run --parallel 2
    ../.venv/bin/python tools/pseudo_auctions.py score
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import statistics as st
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT = HERE.parent
ROOT = AGENT.parent
DATA = ROOT / "calib_data"
TEMPLATE = ROOT / "track4-analysis-public" / "units" / "t4-auction-btc-202411-us7"
OUT = HERE / ".runs" / "pseudo_auctions"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(AGENT))

CUTOFFS = ["2016-03-31", "2016-06-30", "2017-01-31", "2017-04-30", "2017-07-31", "2017-10-31", "2018-01-31",
           "2018-04-30", "2018-07-31", "2018-10-31", "2019-01-31", "2019-04-30"]
TENORS = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def auctions() -> list[dict]:
    out = []
    for f in ("tdirect_Note_2013_2021.json", "tdirect_Bond_2013_2021.json"):
        for a in json.loads((DATA / f).read_text()):
            if a.get("floatingRate") == "Yes" or a.get("inflationIndexSecurity") == "Yes" or not a.get("bidToCoverRatio"):
                continue
            tenor = a.get("originalSecurityTerm") or a.get("securityTerm")
            if tenor not in TENORS:
                continue
            try:
                ca = float(a["competitiveAccepted"])
                out.append({"date": a["auctionDate"][:10], "ann": (a.get("announcementDate") or "")[:10], "tenor": tenor,
                            "term": a["securityTerm"], "reopen": a.get("reopening") == "Yes",
                            "size": float(a["offeringAmount"]) / 1e9, "btc": float(a["bidToCoverRatio"]),
                            "yield": float(a["highYield"]) if a.get("highYield") else None,
                            "indirect": 100.0 * float(a["indirectBidderAccepted"]) / ca if ca else None})
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda x: x["date"])


def _short(tenor: str) -> str:
    return tenor.split("-")[0] + "Y"


def build_unit(cutoff: str, allx: list[dict]) -> tuple[Path, dict[str, float]] | None:
    c = dt.date.fromisoformat(cutoff)
    m_start = (c + dt.timedelta(days=1)).isoformat()
    m_end = (dt.date(c.year + (c.month == 11), (c.month + 1) % 12 + 1, 1) - dt.timedelta(days=1)).isoformat()
    lo = (c - dt.timedelta(days=372)).isoformat()
    targets = [a for a in allx if m_start <= a["date"] <= m_end]
    if len({a["tenor"] for a in targets}) < 5:
        return None
    uid = f"t4-pseudo-auction-{cutoff[:7]}"
    unit = OUT / "units" / uid
    if unit.exists():
        shutil.rmtree(unit)
    (unit / "corpus").mkdir(parents=True)
    files, truth, ents = [], {}, []
    for tenor in TENORS:
        hist = [a for a in allx if a["tenor"] == tenor and lo <= a["date"] <= cutoff]
        if not hist:
            continue
        did = f"TDIRECT_AUCTIONS_{_short(tenor)}_{cutoff.replace('-', '')}"
        lines = [f"U.S. Treasury {tenor} {'Bond' if tenor in ('20-Year', '30-Year') else 'Note'} (new issues and reopenings) "
                 f"auctions -- results from {hist[0]['date'][:7]} through {hist[-1]['date'][:7]}.",
                 "Source: TreasuryDirect auction results API (www.treasurydirect.gov/TA_WS/securities/search), U.S. Department of "
                 "the Treasury, public domain. Bid-to-cover ratio = total bids tendered divided by total accepted; indirect share = "
                 "indirect bidder accepted / competitive accepted.", "",
                 "auction_date | term | new/reopen | offering_$B | bid_to_cover | high_yield_% | indirect_share_%"]
        for a in hist:
            lines.append(f"{a['date']} | {a['term']} | {'reopen' if a['reopen'] else 'new'} | {a['size']:.1f} | {a['btc']:.2f} | "
                         f"{a['yield'] if a['yield'] is not None else ''} | {a['indirect']:.1f}" if a["indirect"] is not None else "")
        last6 = [a["btc"] for a in hist[-6:]]
        lines += ["", f"NOTES (derived from the table above): across the {len(hist)} auctions shown, the bid-to-cover ratio ranged "
                      f"from {min(a['btc'] for a in hist):.2f} to {max(a['btc'] for a in hist):.2f}; the average over the six most "
                      f"recent auctions is {st.mean(last6):.3f}. The most recent auction ({hist[-1]['date']}) drew a bid-to-cover "
                      f"of {hist[-1]['btc']:.2f}."]
        doc = {"doc_id": did, "source": "TreasuryDirect", "doc_type": "auction_history_table", "doc_date": hist[-1]["date"],
               "title": f"Treasury {tenor} auction results, {hist[0]['date'][:7]} to {hist[-1]['date'][:7]}",
               "url": "https://www.treasurydirect.gov/TA_WS/securities/search?format=json", "pii_stripped": True,
               "text": "\n".join(l for l in lines if l is not None)}
        tg = [a for a in targets if a["tenor"] == tenor]
        ids = []
        for a in tg:
            eid = f"AUC_{_short(tenor)}_{a['date'].replace('-', '')}"
            announced = bool(a["ann"]) and a["ann"] <= cutoff
            ents.append({"entity_id": eid, "name": f"{tenor} {'Bond' if tenor in ('20-Year', '30-Year') else 'Note'} auction, {a['date']}",
                         "tenor": tenor, "auction_date": a["date"], "new_or_reopening": "reopen" if a["reopen"] else "new",
                         "offering_amount_usd_bn": round(a["size"], 1) if announced else None,
                         "size_announced_precutoff": announced, "unit": "bid_to_cover_ratio", "corpus_ref": "corpus/"})
            truth[eid] = a["btc"]
            ids.append(eid)
        path = unit / "corpus" / f"{did}.json"
        path.write_text(json.dumps(doc, indent=1))
        files.append({"path": f"corpus/{did}.json", "role": "corpus", "source": "TreasuryDirect", "license": "Public domain",
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
                      "split": "public-dev", "cutoff": cutoff, "redistributable": True, "pii_stripped": True, "entity_ids": ids})
    up_lines = [f"{m_start[:7]} Treasury coupon auctions -- terms announced on or before {cutoff} and the tentative schedule.",
                "Source: U.S. Department of the Treasury / TreasuryDirect, public domain.", ""]
    for e in sorted(ents, key=lambda e: e["auction_date"]):
        size = f"${e['offering_amount_usd_bn']:.0f} billion" if e["size_announced_precutoff"] else "size to be announced"
        up_lines.append(f"- {e['tenor']} {e['name'].split()[1]}, {size}, auction {e['auction_date']}, "
                        f"{'reopening' if e['new_or_reopening'] == 'reopen' else 'new issue'}.")
    did = f"TDIRECT_UPCOMING_{cutoff.replace('-', '')}"
    path = unit / "corpus" / f"{did}.json"
    path.write_text(json.dumps({"doc_id": did, "source": "TreasuryDirect", "doc_type": "auction_announcement", "doc_date": cutoff,
                                "title": f"{m_start[:7]} Treasury coupon auctions: announced sizes and tentative schedule",
                                "url": "https://www.treasurydirect.gov/auctions/upcoming/", "pii_stripped": True,
                                "text": "\n".join(up_lines)}, indent=1))
    files.append({"path": f"corpus/{did}.json", "role": "corpus", "source": "TreasuryDirect", "license": "Public domain",
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size, "split": "public-dev",
                  "cutoff": cutoff, "redistributable": True, "pii_stripped": True, "shared": True})
    (unit / "corpus" / "manifest.json").write_text(json.dumps({"manifest_version": "2.0", "unit_id": uid, "files": files}, indent=1))
    task = json.loads((TEMPLATE / "task.json").read_text())
    task.update({"task_id": uid, "cutoff_date": cutoff, "resolution_date": m_end, "entities": ents})
    task["prompt"] = (f"Using ONLY the frozen evidence corpus (TreasuryDirect auction-results tables for the nominal coupon "
                      f"tenors over the year to the {cutoff} cutoff, plus the announced {m_start[:7]} auction terms), predict for "
                      f"each {m_start[:7]} U.S. Treasury coupon auction in the table its BID-TO-COVER RATIO (total bids tendered "
                      f"divided by amount accepted). Reason from each tenor's own auction history (level and trend of "
                      f"bid-to-cover, sensitivity to auction size, new-issue vs reopening months, indirect-bidder participation) "
                      f"and from the announced sizes. For each auction provide a point_forecast of the bid-to-cover ratio and a "
                      f"90% interval around it. Support every claim with citations to passages in the frozen corpus.")
    (unit / "task.json").write_text(json.dumps(task, indent=1))
    card = (TEMPLATE / "card.toml").read_text()
    card = re.sub(r'(\nid\s*=\s*)"[^"]+"', rf'\1"{uid}"', card)
    card = re.sub(r'cutoff_date\s*=\s*"[^"]+"', f'cutoff_date     = "{cutoff}"', card)
    card = re.sub(r'resolution_date\s*=\s*"[^"]+"', f'resolution_date = "{m_end}"', card)
    (unit / "card.toml").write_text(card)
    (OUT / "truth").mkdir(parents=True, exist_ok=True)
    (OUT / "truth" / f"{uid}.json").write_text(json.dumps(truth, indent=1))  # outside the unit: the agent never sees it
    return unit, truth


def build() -> list[str]:
    allx = auctions()
    made = []
    for c in CUTOFFS:
        r = build_unit(c, allx)
        if r:
            made.append(r[0].name)
    print(f"built {len(made)} pseudo-units:", made)
    return made


def run(parallel: int, agent: Path = AGENT, backend: str = "openrouter", tag: str = "runs") -> None:
    """v21: --agent runs another version's code on the same pseudo-units (v20 against v21), --backend picks the route
    (openrouter-deepinfra: the paid bf16 route) and --tag the output folder under .runs/pseudo_auctions."""
    import run_units as RU

    env = RU.openrouter_env(23, parallel, prod_clocks=True, free=backend == "openrouter")
    env["T4_STAT_WEIGHT"] = "0"  # the model's own points
    units = sorted((OUT / "units").iterdir())

    def one(u: Path):
        out = OUT / tag / u.name
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        p = subprocess.run([str(RU.PY), "-m", "t4agent", "analyze", "--task", str(u / "task.json"), "--corpus",
                            str(u / "corpus"), "--out", str(out / "answer.json")], cwd=agent, env=env,
                           capture_output=True, text=True, timeout=2400)
        (out / "agent.log").write_text(p.stderr)
        m = re.search(r"requests used (\d+), ok (\d+), llm entities (\d+)/(\d+)", p.stderr)
        print(f"{u.name}: exit={p.returncode} {time.time() - t0:.0f}s {m.group(0) if m else 'no final answer'}", flush=True)

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, units))


def score(tag: str = "runs") -> dict:
    from t4agent.corpus import load_corpus, map_entities
    from t4agent.llm import HouseClient
    from t4agent.predict import Evidence, Forecaster
    from t4agent.task import load_task

    rows = []
    for u in sorted((OUT / "units").iterdir()):
        ans = OUT / tag / u.name / "answer.json"
        if not ans.exists():
            continue
        truth = json.loads((OUT / "truth" / f"{u.name}.json").read_text())
        spec = load_task(u / "task.json")
        corpus = load_corpus(u / "corpus", spec)
        f = Forecaster(spec, corpus, Evidence(spec, corpus, *map_entities(spec, corpus)), HouseClient(deadline=0), 0)
        preds = {p["entity_id"]: p for p in json.loads(ans.read_text())["entity_predictions"]}
        log = (OUT / tag / u.name / "agent.log").read_text()
        for e in spec.entities:
            eid = e["entity_id"]
            sf = f.stat.get(eid)
            hist = [float(x) for x in re.findall(r"\| (\d\.\d\d) \|", corpus.docs[[d for d in corpus.docs if d.startswith("TDIRECT_AUCTIONS_" + eid.split("_")[1] + "_")][0]].text)]
            llm = re.search(rf"INFO t4agent:\s+{re.escape(eid)}:.*?point=", log) is not None
            rows.append({"unit": u.name, "eid": eid, "truth": truth[eid], "model": preds[eid]["point_forecast"],
                         "stat": sf.value if sf is not None else None, "last": hist[-1], "mean6": st.mean(hist[-6:]), "llm": llm})
    units = sorted({r["unit"] for r in rows})

    def pq(sub, naive, w):
        n = st.mean(abs(r[naive] - r["truth"]) for r in sub)
        pts = [(1 - w) * r["model"] + w * (r["stat"] if r["stat"] is not None else r[naive]) for r in sub]
        e = st.mean(abs(p - r["truth"]) for p, r in zip(pts, sub))
        return n / (n + e) if n + e > 0 else 0.5

    res = {"forecasts": len(rows), "units": len(units), "model_points": sum(r["llm"] for r in rows)}
    for naive in ("last", "mean6"):
        per = {w: [pq([r for r in rows if r["unit"] == u], naive, w) for u in units] for w in WEIGHTS}
        res[naive] = {"mean_pq": {w: round(st.mean(v), 4) for w, v in per.items()},
                      "wins_vs_0.5": {w: sum(a > b for a, b in zip(per[w], per[0.5])) for w in (0.25, 0.75)}}
    res["stat_is_naive_share"] = round(st.mean(abs((r["stat"] or r["last"]) - r["last"]) < 1e-9 for r in rows), 2)
    print(json.dumps(res, indent=1))
    (OUT / f"score_{tag}.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "score"])
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--agent", type=Path, default=AGENT, help="the agent folder whose code runs (default: this one)")
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "openrouter-deepinfra"])
    ap.add_argument("--tag", default="runs", help="output folder under .runs/pseudo_auctions")
    a = ap.parse_args()
    {"build": build, "run": lambda: run(a.parallel, a.agent.resolve(), a.backend, a.tag),
     "score": lambda: score(a.tag)}[a.cmd]()
