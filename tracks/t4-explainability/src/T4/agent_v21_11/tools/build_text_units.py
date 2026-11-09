"""Build text-heavy Track-4-format stress units from SEC filings published before 2022 (development only; never in the
image). Companion of tools/build_pre2022_units.py (same writer, same output folder ../units_pre2022).

The public text units carry whole filings flattened to text (a 10-Q is about 200,000 characters), not excerpts. These
units do the same: for each company, its latest 10-Q before the cutoff and its latest earnings press release (8-K
Exhibit 99.1), whole, as flattened text. The outcome is the quarter's GAAP diluted EPS as first reported in XBRL
(the 10-Q filed after the cutoff); the reference is the same quarter a year earlier as reported then.

  t4-x-epsbeat-2019Q3   beat / inline / miss against last year's quarter +/-5%, the rule stated in the entity table's
                        condition columns (the public example's form)
  t4-x-epsgrowth3-2021Q1  strong_growth / moderate / decline against last year's quarter, stated only in words
                        relative to that quarter (rules the code does not read: the model has to judge)

SEC fair access: a User-Agent with a contact (track4_my/.sec.env, never printed) and at most about two requests a
second; every response is cached under ../units_pre2022/raw/sec_*.

    ../.venv/bin/python tools/build_text_units.py [--only t4-x-epsbeat-2019Q3]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import statistics as st
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_pre2022_units as B  # noqa: E402

ROOT = B.ROOT
RAW = B.RAW
LICENSE_SEC = "Public domain (US government works: SEC EDGAR filings), retrieved from sec.gov"
_last = [0.0]


def _ua() -> str:
    m = re.search(r"=\s*(.+)", (ROOT / ".sec.env").read_text())
    return m.group(1).strip()


def sec_get(url: str, binary: bool = False):
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / ("sec_" + hashlib.sha1(url.encode()).hexdigest()[:20])
    if cache.exists():
        data = cache.read_bytes()
    else:
        wait = 0.6 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": _ua(), "Accept-Encoding": "identity"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read()
                break
            except Exception:  # noqa: BLE001 - retry transient errors
                if attempt == 3:
                    raise
                time.sleep(3 + 3 * attempt)
        _last[0] = time.time()
        cache.write_bytes(data)
    return data if binary else data.decode("utf-8", errors="replace")


def submissions(cik: int) -> list[dict]:
    """Every filing of the company (recent page plus the older pages), newest first."""
    d = json.loads(sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    pages = [d["filings"]["recent"]]
    for f in d["filings"].get("files", []):
        pages.append(json.loads(sec_get(f"https://data.sec.gov/submissions/{f['name']}")))
    out = []
    for p in pages:
        keys = list(p)
        for i in range(len(p["accessionNumber"])):
            out.append({k: p[k][i] for k in keys})
    return sorted(out, key=lambda x: x["filingDate"], reverse=True)


def html_to_text(raw: str) -> str:
    """Flatten a filing's HTML the way the public units read: block ends become line breaks, table cells spaces."""
    t = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", raw)
    t = re.sub(r"(?is)<ix:header>.*?</ix:header>", " ", t)
    t = re.sub(r"(?i)</(?:p|div|tr|li|h[1-6]|table)\s*>|<br\s*/?>", "\n", t)
    t = re.sub(r"(?i)</t[dh]\s*>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t).replace("\xa0", " ")
    lines = [re.sub(r"[ \t​]+", " ", ln).strip() for ln in t.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return re.sub(r"\n{2,}", "\n", text)


def filing_url(cik: int, f: dict, name: str | None = None) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{f['accessionNumber'].replace('-', '')}/{name or f['primaryDocument']}"


def exhibit_991(cik: int, f: dict) -> str | None:
    """The press release (Exhibit 99.1, else the first EX-99) file name of an 8-K, read off the filing index page's
    document types (file names do not always say 'ex99')."""
    acc = f["accessionNumber"]
    page = sec_get(filing_url(cik, f, f"{acc}-index.htm"))
    best = None
    for row in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", page):
        m = re.search(r'href="[^"]*/([^"/]+\.(?:htm|html|txt))"', row, re.I)
        typ = re.search(r"(?i)>\s*(EX-99[^<\s]*)\s*<", row)
        if m and typ:
            if typ.group(1).upper() in ("EX-99.1", "EX-99.01", "EX-99"):
                return m.group(1)
            best = best or m.group(1)
    return best


def eps_facts(cik: int) -> list[dict]:
    d = json.loads(sec_get(f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/EarningsPerShareDiluted.json"))
    return d["units"].get("USD/shares", [])


def quarter_eps(facts: list[dict], end: str, filed_after: str | None = None, filed_before: str | None = None):
    """A three-month diluted EPS for the quarter ending `end` (+/- 10 days), the first filing in the window."""
    e0 = date.fromisoformat(end)
    best = None
    for x in facts:
        if not x.get("start") or x.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
            continue
        s, e = date.fromisoformat(x["start"]), date.fromisoformat(x["end"])
        if abs((e - e0).days) > 10 or not 80 <= (e - s).days <= 100:
            continue
        if filed_after and x["filed"] <= filed_after or filed_before and x["filed"] > filed_before:
            continue
        if best is None or x["filed"] < best["filed"]:
            best = x
    return best


def company_docs(cik: int, ticker: str, name: str, cutoff: str) -> list[dict]:
    """The latest 10-Q before the cutoff and the latest earnings 8-K's Exhibit 99.1, as corpus documents."""
    subs = submissions(cik)
    docs = []
    q = next((f for f in subs if f["form"] == "10-Q" and f["filingDate"] <= cutoff), None)
    k = next((f for f in subs if f["form"] == "8-K" and f["filingDate"] <= cutoff and "2.02" in str(f.get("items", ""))), None)
    if q:
        text = html_to_text(sec_get(filing_url(cik, q)))
        docs.append({"doc_id": f"EDGAR_{cik:010d}_10Q_{q['filingDate'].replace('-', '')}", "source": "EDGAR",
                     "cik": f"{cik:010d}", "ticker": ticker, "form_type": "10-Q", "filed_date": q["filingDate"],
                     "doc_date": q["filingDate"], "period_of_report": q.get("reportDate"),
                     "accession_number": q["accessionNumber"], "title": f"{name} 10-Q (10-Q)",
                     "url": filing_url(cik, q), "pii_stripped": True, "text": text})
    if k:
        ex = exhibit_991(cik, k)
        if ex:
            text = html_to_text(sec_get(filing_url(cik, k, ex)))
            docs.append({"doc_id": f"EDGAR_{cik:010d}_8K_{k['filingDate'].replace('-', '')}", "source": "EDGAR",
                         "cik": f"{cik:010d}", "ticker": ticker, "form_type": "8-K", "filed_date": k["filingDate"],
                         "doc_date": k["filingDate"], "period_of_report": k.get("reportDate"),
                         "accession_number": k["accessionNumber"],
                         "title": f"{name} earnings press release (8-K Exhibit 99.1)",
                         "url": filing_url(cik, k, ex), "pii_stripped": True, "text": text})
    return docs


def ciks() -> dict[str, tuple[int, str]]:
    d = json.loads(sec_get("https://www.sec.gov/files/company_tickers.json"))
    return {v["ticker"]: (int(v["cik_str"]), v["title"]) for v in d.values()}


def eps_unit(uid: str, family: str, title: str, tickers: list[str], cutoff: str, resolution: str, q_end: str,
             prior_end: str, labels: list[str], assertions: dict, prompt: str, rule, table_conditions: bool) -> dict:
    """One EPS unit: per company the reported quarter's diluted EPS (truth) against last year's quarter."""
    cmap = ciks()
    ents, docs, outs, naive = [], [], [], []
    for tk in tickers:
        cik, name = cmap[tk]
        facts = eps_facts(cik)
        truth = quarter_eps(facts, q_end, filed_after=cutoff)
        prior = quarter_eps(facts, prior_end, filed_before=cutoff)
        if not truth or not prior or prior["val"] <= 0:
            print(f"  skip {tk}: truth {truth and truth['val']} prior {prior and prior['val']}")
            continue
        cdocs = company_docs(cik, tk, name, cutoff)
        if len(cdocs) < 2:
            print(f"  skip {tk}: {len(cdocs)} documents")
            continue
        hist = sorted((x for x in facts if x.get("start") and x["filed"] <= cutoff and 80 <= (date.fromisoformat(x["end"])
                       - date.fromisoformat(x["start"])).days <= 100), key=lambda x: x["end"])
        by_end = {}
        for x in hist:
            by_end.setdefault(x["end"], x["val"])
        ends = sorted(by_end)
        yoy = [by_end[e] - by_end[p] for e in ends for p in ends
               if 360 <= (date.fromisoformat(e) - date.fromisoformat(p)).days <= 370]
        half = B.Z90 * (st.pstdev(yoy[-8:]) if len(yoy) >= 3 else 0.25 * abs(prior["val"]))
        y = truth["val"]
        lab = rule(y, prior["val"])
        ent = {"entity_id": tk, "name": name, "cik": f"{cik:010d}", "quarter_reported": f"three months ended {q_end}",
               "prior_year_quarter": f"three months ended {prior_end}", "prior_year_q_eps": prior["val"],
               "expected_report_date": truth["filed"], "currency": "USD/share", "corpus_ref": "corpus/"}
        if table_conditions:
            ent.update({"threshold_pct": 0.05,
                        "beat_condition": "reported_eps > prior_year_q_eps * (1 + threshold_pct)",
                        "miss_condition": "reported_eps < prior_year_q_eps * (1 - threshold_pct)",
                        "inline_condition": "abs(reported_eps - prior_year_q_eps) / prior_year_q_eps <= threshold_pct"})
        ents.append(ent)
        docs += [(d, [tk]) for d in cdocs]
        outs.append({"entity_id": tk, "true_label": lab, "y": y})
        naive.append(B.naive_row(tk, prior["val"], half, rule(prior["val"], prior["val"])))
        print(f"  {tk}: prior {prior['val']} -> {y} ({lab}); docs {[len(d['text']) for d in cdocs]}")
    return {"id": uid, "family": family, "title": title,
            "target": {"name": "eps_outcome_vs_prior_year" if table_conditions else "eps_growth_band", "type": "classification",
                       "labels": labels, "label_assertions": assertions},
            "prompt": prompt, "cutoff": cutoff, "resolution": resolution, "entities": ents, "docs": docs,
            "outcomes": outs, "naive": naive, "naive_rule": "prior_year_eps_carried_forward", "license": LICENSE_SEC,
            "data_source": "SEC EDGAR 10-Q and 8-K Exhibit 99.1 (corpus); SEC XBRL companyconcept EarningsPerShareDiluted "
                           "(outcome, the 10-Q filed after the cutoff)"}


def u_epsbeat() -> dict:
    def rule(y, p):
        return "beat" if y > p * 1.05 else "miss" if y < p * 0.95 else "inline"
    labels = ["beat", "miss", "inline"]
    return eps_unit(
        "t4-x-epsbeat-2019Q3", "eps_beat_prior_year", "Predict whether ten companies' Q3 2019 diluted EPS beats, misses or "
        "lands inline with the same quarter of 2018 (5% band; classification)",
        ["IBM", "UNP", "HON", "MCD", "TXN", "CAT", "MMM", "INTC", "LMT", "UPS", "PFE", "MRK"], "2019-10-15", "2019-11-08",
        "2019-09-30", "2018-09-30", labels,
        {"beat": "will report Q3 2019 GAAP diluted earnings per share more than 5% above the same quarter of 2018",
         "miss": "will report Q3 2019 GAAP diluted earnings per share more than 5% below the same quarter of 2018",
         "inline": "will report Q3 2019 GAAP diluted earnings per share within 5% of the same quarter of 2018"},
        "Using ONLY the frozen evidence corpus (each company's latest 10-Q and latest earnings press release filed on or "
        "before 2019-10-15), predict for each company whether its GAAP diluted earnings per share for the three months "
        "ended 2019-09-30, as first reported in its 10-Q, will BEAT, MISS or land INLINE with the same quarter of 2018 "
        "(prior_year_q_eps), at the 5% threshold the table's condition columns state. Also give a point forecast of the "
        "quarter's diluted EPS in USD with a 90% interval, and cite corpus passages.",
        rule, True)


def u_epsgrowth3() -> dict:
    def rule(y, p):
        g = 100 * (y / p - 1)
        return "strong_growth" if g > 20 else "decline" if g < -20 else "moderate"
    labels = ["strong_growth", "moderate", "decline"]
    return eps_unit(
        "t4-x-epsgrowth3-2021Q1", "eps_growth_band_text", "Predict the band of ten companies' Q1 2021 diluted EPS growth "
        "against Q1 2020 (classification)",
        ["PEP", "UNH", "JNJ", "IBM", "VZ", "INTC", "UNP", "HON", "LMT", "MMM", "UPS", "TXN"], "2021-04-12", "2021-05-10",
        "2021-03-31", "2020-03-31", labels,
        {"strong_growth": "diluted earnings per share for the first quarter of 2021 grows by more than 20 percent compared "
                          "with the first quarter of 2020",
         "moderate": "diluted earnings per share for the first quarter of 2021 is within 20 percent of the first quarter "
                     "of 2020, above or below",
         "decline": "diluted earnings per share for the first quarter of 2021 falls by more than 20 percent compared with "
                    "the first quarter of 2020"},
        "Using ONLY the frozen evidence corpus (each company's latest 10-Q and latest earnings press release filed on or "
        "before 2021-04-12), predict for each company whether GAAP diluted earnings per share for the first quarter of "
        "2021, as first reported in its 10-Q, shows STRONG_GROWTH (more than 20 percent above the first quarter of 2020), "
        "is MODERATE (within 20 percent of it), or shows a DECLINE (more than 20 percent below it). Reason from each "
        "company's reported results, one-off items and outlook. Also give a point forecast of the quarter's diluted EPS "
        "in USD with a 90% interval, and cite corpus passages.",
        rule, False)


BUILDERS = {"t4-x-epsbeat-2019Q3": u_epsbeat, "t4-x-epsgrowth3-2021Q1": u_epsgrowth3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    for n in [x for x in args.only.split(",") if x] or list(BUILDERS):
        try:
            u = BUILDERS[n]()
            B.write_unit(u)
            print(f"OK   {n}: {len(u['entities'])} entities, {len(u['docs'])} docs, "
                  f"{sum(len(d['text']) for d, _ in u['docs']) // 1000} k chars; labels {[o['true_label'] for o in u['outcomes']]}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {n}: {type(exc).__name__}: {str(exc)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
