"""S8: a claim is a full sentence or table row about that entity, never a header or another entity's line."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from t4agent.answer import build_answer
from t4agent.cite import (
    _has_alias,
    build_citations,
    claim_score,
    entity_aliases,
    is_header,
    quote_units,
)
from t4agent.corpus import Corpus, Doc, load_corpus, map_entities
from t4agent.predict import Evidence, Forecast
from t4agent.task import TaskSpec, load_task

UNITS = Path(__file__).resolve().parents[2] / "track4-analysis-public" / "units"
_FOCUS = (
    "t4-auction-btc-202411-us7",
    "t4-cpicomp-202410-us11",
    "t4-cotpos-202411-us10",
    "t4-macrorev-20240930-us6",
)


class _Search:
    def search(self, *_args, **_kwargs):
        return []


class _Ev:
    def __init__(self):
        self.facts = {}
        self.specific = {}
        self.bm = _Search()


def _spec(tmp_path: Path, entities: list[dict]) -> TaskSpec:
    path = tmp_path / "task.json"
    path.write_text(json.dumps({
        "task_id": "mini",
        "schema_version": "3",
        "target": {"name": "outcome", "type": "regression"},
        "prompt": "Predict.",
        "cutoff_date": "2024-10-31",
        "entities": entities,
    }), encoding="utf-8")
    return load_task(path)


def _corpus(text: str) -> Corpus:
    return Corpus(docs={"D1": Doc("D1", date(2024, 10, 1), text, {})}, chunks=[])


def test_header_and_score_follow_the_selection_rule():
    vintage = ("VINTAGE TABLE — the value of PI for each reference month exactly as published "
               "on each pre-cutoff release (vintage) date.")
    notes = "REVISION NOTES (derived from the vintage table above):"
    row = "2024-07 | -- | -- | 24015.4 | 24803.2"
    methodology = ("Because contract sizes and open interest differ enormously across markets, changes in net "
                   "positioning are commonly compared across markets as a percent of open interest rather than "
                   "in raw contracts.")
    assert is_header(vintage) and is_header(notes)
    assert is_header("CONSUMER PRICE INDEX - SEPTEMBER 2024")
    assert is_header("Report mechanics:")
    assert not is_header(row) and not is_header(methodology)
    assert claim_score(vintage, ["PI"], []) == 0  # own, no figure, header
    assert claim_score(notes, ["RSAFS"], []) == -3
    assert claim_score(row, ["2024-07"], ["2024-08"]) == 4
    assert claim_score("- 3-Year Note, $58 billion, auction 2024-11-04, new issue.", ["10-Year"], ["3-Year"]) == -3
    assert claim_score("- 10-Year Note, $42 billion, auction 2024-11-05, new issue.", ["10-Year"], ["3-Year"]) == 4
    assert claim_score(methodology, ["Corn"], ["Silver"]) == 0  # no alias, no digit, not a header


def test_alias_boundaries_do_not_swallow_a_neighbour():
    assert _has_alias("20-Year Bond", ["2-Year"]) is False
    assert _has_alias("9-Year 10-Month | reopen", ["10-Year"]) is False
    assert _has_alias("- 10-Year Note, $42 billion", ["10-Year"]) is True
    assert _has_alias("as_of_2024-07-26", ["2024-07"]) is False
    assert _has_alias("2024-07 | -- | 24015.4", ["2024-07"]) is True
    assert _has_alias("The shelter index increased 4.9 percent.", ["Shelter"]) is True
    assert _has_alias("The shelter index increased 4.9 percent.", ["Medical care"]) is False


def test_quote_units_are_full_sentences_or_rows():
    text = (
        "U.S. Treasury 10-Year Note auctions — results from October 2023 through October 2024. "
        "Source: TreasuryDirect.\n"
        "The shelter index\n"
        "increased 4.9 percent over the last year, accounting for over 65 percent of the total increase.\n"
        "2024-09 | +0.18 | +0.31 | +0.22\n"
        "- Medical care: September 2024 first print +0.41%.\n"
    )
    units = [text[s:e] for s, e in quote_units(text)]
    assert units[0] == ("U.S. Treasury 10-Year Note auctions — results from October 2023 through October 2024.")
    assert units[1] == "Source: TreasuryDirect."
    assert units[2].startswith("The shelter index\nincreased 4.9 percent")
    assert "65 percent" in units[2] and not units[2].startswith("increased")
    assert units[3].startswith("2024-09 |")
    assert units[4].startswith("- Medical care:")
    # a decimal is not a sentence boundary
    decimal = "The index rose 2.50 percent in September. The next index rose too."
    parts = [decimal[s:e] for s, e in quote_units(decimal)]
    assert parts == ["The index rose 2.50 percent in September.", "The next index rose too."]


def test_a_foreign_or_header_only_document_leaves_the_task_row_path(tmp_path):
    spec = _spec(tmp_path, [
        {"entity_id": "AUC_10Y_20241105", "name": "10-Year Note auction", "tenor": "10-Year",
         "offering_amount_usd_bn": 42},
        {"entity_id": "AUC_3Y_20241104", "name": "3-Year Note auction", "tenor": "3-Year",
         "offering_amount_usd_bn": 58},
    ])
    corpus = _corpus(
        "VINTAGE TABLE — the value of nothing in particular for a later reader.\n"
        "REVISION NOTES (derived from the vintage table above):\n"
        "- 3-Year Note, $58 billion, auction 2024-11-04, new issue.\n"
    )
    ev = _Ev()
    ten = build_citations(spec, corpus, ev, Forecast(eid="AUC_10Y_20241105"), [])
    three = build_citations(spec, corpus, ev, Forecast(eid="AUC_3Y_20241104"), [])
    assert ten == []
    assert three and "3-Year" in three[0]["claim"] and three[0]["claim"] == corpus.docs["D1"].text[
        three[0]["span_start"]:three[0]["span_end"]]
    rows = {
        "AUC_10Y_20241105": {"point_forecast": 2.5, "lo": 2.0, "hi": 3.0, "claims": ten},
        "AUC_3Y_20241104": {"point_forecast": 2.5, "lo": 2.0, "hi": 3.0, "claims": three},
    }
    ans = build_answer(spec, rows, "", None, corpus)
    by_id = {p["entity_id"]: p for p in ans["entity_predictions"]}
    assert by_id["AUC_10Y_20241105"]["claims"][0]["doc_id"] == "task"
    assert by_id["AUC_10Y_20241105"]["point_forecast"] == 2.5
    assert "3-Year" in by_id["AUC_3Y_20241104"]["claims"][0]["claim"]


def test_a_cut_shelter_sentence_is_not_cited_for_medical(tmp_path):
    spec = _spec(tmp_path, [
        {"entity_id": "CPI_MEDICAL", "name": "Medical care"},
        {"entity_id": "CPI_SHELTER", "name": "Shelter"},
    ])
    text = (
        "The shelter index\n"
        "increased 4.9 percent over the last year, accounting for over 65 percent of the total "
        "12-month increase in the all items index.\n"
        "Other indexes with notable increases include medical care (+3.3 percent) and apparel (+1.8 percent).\n"
        "- Medical care: September 2024 first print +0.41%.\n"
    )
    corpus = _corpus(text)
    ev = _Ev()
    medical = build_citations(spec, corpus, ev, Forecast(eid="CPI_MEDICAL"), [])
    shelter = build_citations(spec, corpus, ev, Forecast(eid="CPI_SHELTER"), [])
    assert medical and all("65 percent" not in c["claim"] for c in medical)
    assert any("Medical care" in c["claim"] or "medical care" in c["claim"] for c in medical)
    assert any(c["claim"].startswith("The shelter index\nincreased 4.9 percent") for c in shelter)
    assert all(not c["claim"].startswith("increased") for c in shelter)


def test_public_focus_units_drop_the_known_bad_claims():
    needles = {
        ("t4-auction-btc-202411-us7", "AUC_10Y_20241105"): "3-Year Note, $58",
        ("t4-cpicomp-202410-us11", "CPI_MEDICAL"): "accounting for over 65 percent",
        ("t4-cotpos-202411-us10", "SILVER_CMX"): "2-year Treasury yield rose",
        ("t4-cotpos-202411-us10", "CORN_CBT"): "compared across markets as a percent of open interest",
        ("t4-macrorev-20240930-us6", "PI_2024-07_20241031"): "VINTAGE TABLE",
        ("t4-macrorev-20240930-us6", "PI_2024-08_20241127"): "VINTAGE TABLE",
        ("t4-macrorev-20240930-us6", "RSAFS_2024-07_20241017"): "REVISION NOTES",
        ("t4-macrorev-20240930-us6", "RSAFS_2024-08_20241115"): "REVISION NOTES",
    }
    wants = {
        ("t4-auction-btc-202411-us7", "AUC_10Y_20241105"): "10-Year",
        ("t4-cpicomp-202410-us11", "CPI_MEDICAL"): "edical",
        ("t4-cotpos-202411-us10", "SILVER_CMX"): "ilver",
        ("t4-cotpos-202411-us10", "CORN_CBT"): "orn",
        ("t4-macrorev-20240930-us6", "PI_2024-07_20241031"): "2024-07",
        ("t4-macrorev-20240930-us6", "RSAFS_2024-07_20241017"): "2024-07",
    }
    for uname in _FOCUS:
        spec = load_task(UNITS / uname / "task.json")
        aliases = entity_aliases(spec)
        corpus = load_corpus(UNITS / uname / "corpus", spec)
        specific, shared = map_entities(spec, corpus)
        ev = Evidence(spec, corpus, specific, shared)
        ev.build_packs([])
        for eid in spec.entity_ids:
            own = aliases[eid]
            other = tuple(a for k, als in aliases.items() if k != eid for a in als
                          if a.casefold() not in {x.casefold() for x in own})
            cites = build_citations(spec, corpus, ev, Forecast(eid=eid), [])
            assert cites, eid
            blob = "\n".join(c["claim"] for c in cites)
            needle = needles.get((uname, eid))
            if needle:
                assert needle not in blob, (eid, needle)
            want = wants.get((uname, eid))
            if want:
                assert want in blob, (eid, want, blob[:240])
            for c in cites:
                text = corpus.docs[c["doc_id"]].text
                assert c["claim"] == text[c["span_start"]:c["span_end"]]
                assert c["span_start"] in {s for s, _e in quote_units(text)}
                assert not is_header(c["claim"])
                assert not (_has_alias(c["claim"], other) and not _has_alias(c["claim"], own))
