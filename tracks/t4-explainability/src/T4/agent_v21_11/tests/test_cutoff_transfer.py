"""Synthetic transfer/admissibility and noninterference, never scored holdout evidence.

All observations here are independently invented and dated before 2019. Nothing is
copied or redated from a public unit. Fake House replies exercise the prompt and
aggregation path without network calls; the CLI cases run with model env removed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
sys.path.insert(0, str(AGENT.parent / "track4-analysis-public"))

from qfbench2_common.taskcard import schema_path
from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.corpus import CorpusIndex
from t4agent.calibrate import finalize, table_halves
from t4agent.corpus import load_corpus, map_entities
from t4agent.llm import HouseClient, Reply
from t4agent.predict import Evidence, Forecast, Forecaster
from t4agent.task import load_task


class CapturedHouse(HouseClient):
    """Replace only the remote boundary, retaining production reply parsing."""

    def __init__(self, items):
        super().__init__(deadline=time.time() + 3600)
        self.items = items
        self.captured = []

    def chat(self, messages, **kwargs):
        self.captured.append((kwargs.get("tag"), messages))
        return Reply(json.dumps({"forecasts": self.items}), "", "stop", 50, .01)


def write_unit(path, target_type="regression", *, target_name="instrument_level", entities=None, documents=None):
    path.mkdir()
    (path / "corpus").mkdir()
    entities = entities or [
        {"entity_id": "SYNTH_A", "name": "Fictional series A", "unit": "index units", "latest_value": 18},
        {"entity_id": "SYNTH_B", "name": "Fictional series B", "unit": "index units", "latest_value": 20},
    ]
    target = {"type": target_type, "name": target_name}
    if target_type == "classification":
        target["labels"] = ["up", "down"]
    task = {
        "task_id": "synthetic-pre-2020-" + target_type, "schema_version": "3", "target": target,
        "prompt": "Predict the next quantity using only this independently invented historical corpus.",
        "cutoff_date": "2018-09-01", "resolution_date": "2018-10-01", "interval_level": .9,
        "corpus_manifest": "corpus/manifest.json", "entities": entities,
    }
    (path / "task.json").write_text(json.dumps(task))
    if documents is None:
        documents = []
        for entity, values in zip(entities, ([10, 12, 11, 14, 15, 18], [30, 27, 28, 25, 23, 20])):
            eid = entity["entity_id"]
            text = f"Fictional observations for {eid}.\ndate | {eid}\n" + "\n".join(
                f"2017-{i + 1:02} | {value}" for i, value in enumerate(values)
            )
            documents.append(({"doc_id": "HISTORY_" + eid, "doc_date": "2018-08-01", "text": text}, [eid]))
        documents.append(({"doc_id": "FUTURE_DECOY", "doc_date": "2019-01-01", "text": "Unusable future value 999999."}, [entities[0]["entity_id"]]))
    files = []
    for doc, eids in documents:
        fp = path / "corpus" / (doc["doc_id"] + ".json")
        fp.write_text(json.dumps(doc))
        files.append({"path": "corpus/" + fp.name, "role": "corpus", "entity_ids": eids,
                      "sha256": hashlib.sha256(fp.read_bytes()).hexdigest()})
    manifest = {"manifest_version": "2.0", "files": files}
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "corpus" / "manifest.json").write_text(json.dumps(manifest))
    return path


def forecaster(unit, items=(), cutoff=None):
    spec = load_task(unit / "task.json")
    if cutoff is not None:
        spec = replace(spec, cutoff=cutoff, cutoff_str=cutoff.isoformat())
    corpus = load_corpus(unit / "corpus", spec)
    ev = Evidence(spec, corpus, *map_entities(spec, corpus))
    client = CapturedHouse(list(items))
    return Forecaster(spec, corpus, ev, client, time.time())


def test_recheck_prompt_never_calls_unmeasured_runtime_scale_backtest_error(tmp_path):
    unit = write_unit(tmp_path / "unit")
    items = [{"entity_id": "SYNTH_A", "median": 19, "p05": 15, "p95": 23}]
    f = forecaster(unit, items)
    forecasts = {"SYNTH_A": Forecast("SYNTH_A", median=200, p05=190, p95=210, source="llm")}
    f.recheck(forecasts)
    prompts = [messages for tag, messages in f.client.captured if tag == "recheck"]
    assert prompts, "the deliberately implausible model quantity must exercise an actual recheck request"
    text = json.dumps(prompts).lower()
    assert "theoretical" in text
    assert "90% backtest error" not in text
    assert forecasts["SYNTH_A"].rechecked


@pytest.mark.parametrize("target_type", ["classification", "ranking", "regression"])
def test_pre_2020_synthetic_cli_answer_passes_official_structural_contract(tmp_path, target_type):
    unit = write_unit(tmp_path / target_type, target_type)
    out = tmp_path / "answer.json"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("MODEL_", "T4_"))}
    proc = subprocess.run(
        [sys.executable, "-m", "t4agent", "analyze", "--task", str(unit / "task.json"),
         "--corpus", str(unit / "corpus"), "--out", str(out)],
        cwd=AGENT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    answer = json.loads(out.read_text())
    task = json.loads((unit / "task.json").read_text())
    jsonschema.Draft202012Validator(json.loads(Path(schema_path("analysis.schema.json")).read_text())).validate(answer)
    aligned = align_predictions(answer, EntityRoster.from_task(task), target_type=target_type, interval_level=.9)
    cutoff = dt.date(2018, 9, 1)
    corpus_index = CorpusIndex.from_unit(unit).with_task_table(task, cutoff)
    assert corpus_index.embargo_report(aligned.all_citations(), cutoff).clean
    rows = {row["entity_id"]: row for row in answer["entity_predictions"]}
    assert set(rows) == {"SYNTH_A", "SYNTH_B"}
    for row in rows.values():
        assert row["interval"]["lo"] < row["interval"]["hi"]
        assert row["claims"]
        assert all(claim["doc_id"] != "FUTURE_DECOY" for claim in row["claims"])
    if target_type != "classification":
        # Both independent histories are named; the same last-value rule carries
        # their distinct latest observations through the complete CLI path.
        assert rows["SYNTH_A"]["point_forecast"] == 18
        assert rows["SYNTH_B"]["point_forecast"] == 20


def eps_unit(path, target_type):
    ent = {"entity_id": "FICTIONAL_EPS", "name": "Invented earnings issuer", "unit": "USD/share",
           "prior_year_eps": 1.0, "target_quarter": "2018-09-30"}
    text = ("Three months ended June 30, 2018 and June 30, 2017\n"
            "Diluted earnings per share 2.00 1.00\n" + "Synthetic filing narrative.\n" * 800)
    doc = {"doc_id": "SYNTHETIC_10Q", "doc_date": "2018-08-01", "form_type": "10-Q",
           "period_of_report": "2018-06-30", "text": text}
    return write_unit(path, target_type, target_name="eps_level" if target_type == "regression" else "eps_direction",
                      entities=[ent], documents=[(doc, [ent["entity_id"]])])


def revision_unit(path, target_type):
    ent = {"entity_id": "FICTIONAL_REVISION", "name": "Invented release series", "unit": "units",
           "reference_period": "2018-06", "latest_vintage": "2018-07-15", "latest_precutoff_estimate": 150,
           "resolving_release": "2018-08-15"}
    lines = ["reference | " + " | ".join(f"as_of_2018-{month:02}-15" for month in range(2, 8))]
    for month in range(1, 7):
        values = ["NA"] * (month - 1) + [str(90 + 10 * month + month * step) for step in range(7 - month)]
        lines.append(f"2018-{month:02} | " + " | ".join(values))
    doc = {"doc_id": "SYNTHETIC_VINTAGES", "doc_date": "2018-07-16", "text": "\n".join(lines)}
    return write_unit(path, target_type, target_name="next_estimate" if target_type == "regression" else "revision_direction",
                      entities=[ent], documents=[(doc, [ent["entity_id"]])])


def model_snapshot(unit, *, kind, cutoff=None):
    spec = load_task(unit / "task.json")
    eid = spec.entity_ids[0]
    median = 1.3 if kind == "eps" else 151.0
    item = {"entity_id": eid, "median": median, "p05": median - 1, "p95": median + 1}
    if spec.target_type == "classification":
        item = {"entity_id": eid, "probs": {"up": .6, "down": .4},
                "quantity": {"median": median, "p05": median - 1, "p95": median + 1}}
    f = forecaster(unit, [item], cutoff)
    f.plan = {"interval_quantity": "GAAP diluted EPS in USD/share" if kind == "eps" else "revised estimate in units",
              "quantity_units": "USD/share" if kind == "eps" else "units", "interval_is_probability": False,
              "baseline_field": "prior_year_eps" if kind == "eps" else "latest_precutoff_estimate"}
    f.build_anchors()
    items = f._one_call([eid], 0)
    forecasts = f.aggregate({eid: [items[eid]]})
    row = finalize(f.spec, forecasts[eid], f.plan, ent=f.spec.entity(eid),
                   table_half=table_halves(f.spec, f.corpus, f.ev)[eid])
    return f, (row, f.client.captured)


@pytest.mark.parametrize("target_type", ["classification", "regression"])
def test_eps_constants_cannot_change_early_predictions_or_actual_analyst_prompt(tmp_path, monkeypatch, target_type):
    from t4agent import eps

    unit = eps_unit(tmp_path / "eps", target_type)
    old, before = model_snapshot(unit, kind="eps")
    modern, positive_control = model_snapshot(unit, kind="eps", cutoff=dt.date(2023, 1, 1))
    eid = old.spec.entity_ids[0]
    assert not old.anchor and old.stat[eid] is None
    # A real EPS parse on the same invented filing is required: an empty fixture
    # could otherwise make both sides invariant even if the policy were broken.
    anchor = modern.anchor.get(eid) or modern.stat[eid]
    assert anchor.latest.cur == 2 and anchor.latest.prior == 1
    assert anchor.level == pytest.approx(1.4)
    for name, value in {"PHI": .9, "CARRY_FAR": .81, "EPS_SCALE_NEXT": 4.0, "EPS_SCALE_FAR": 5.0,
                        "EPS_P_UP_AFTER_UP": .05, "EPS_P_UP_AFTER_DOWN": .95}.items():
        monkeypatch.setattr(eps, name, value)
    assert model_snapshot(unit, kind="eps")[1] == before
    assert model_snapshot(unit, kind="eps", cutoff=dt.date(2023, 1, 1))[1] != positive_control


@pytest.mark.parametrize("target_type", ["classification", "regression"])
def test_revision_constants_cannot_change_early_predictions_or_actual_analyst_prompt(tmp_path, monkeypatch, target_type):
    from t4agent import revisions

    unit = revision_unit(tmp_path / "revision", target_type)
    old, before = model_snapshot(unit, kind="revision")
    modern, positive_control = model_snapshot(unit, kind="revision", cutoff=dt.date(2023, 1, 1))
    eid = old.spec.entity_ids[0]
    assert not old.prior
    assert modern.prior[eid].n_all > 0 and modern.prior[eid].q90 > 0
    monkeypatch.setattr(revisions, "DIRECTION_WEIGHT", .95)
    monkeypatch.setattr(revisions, "SCALE_ONE_RELEASE", 4.0)
    monkeypatch.setattr(revisions, "SCALE_MORE_RELEASES", 5.0)
    assert model_snapshot(unit, kind="revision")[1] == before
    assert model_snapshot(unit, kind="revision", cutoff=dt.date(2023, 1, 1))[1] != positive_control


@pytest.mark.parametrize("target_type", ["regression", "ranking"])
def test_pre_2020_model_points_are_not_shrunk_by_selected_blend_weight(tmp_path, monkeypatch, target_type):
    from t4agent import config

    unit = write_unit(tmp_path / "blend", target_type)
    items = [{"entity_id": "SYNTH_A", "median": 19, "p05": 15, "p95": 23},
             {"entity_id": "SYNTH_B", "median": 21, "p05": 17, "p95": 25}]

    def run(cutoff):
        f = forecaster(unit, items, cutoff)
        parsed = f._one_call(f.spec.entity_ids, 0)
        rows = f.aggregate({eid: [item] for eid, item in parsed.items()})
        return {eid: row.median for eid, row in rows.items()}

    monkeypatch.setattr(config, "STAT_WEIGHT", .05)
    before = run(dt.date(2018, 9, 1))
    modern = run(dt.date(2023, 1, 1))
    assert before == {"SYNTH_A": 19, "SYNTH_B": 21}
    monkeypatch.setattr(config, "STAT_WEIGHT", .95)
    assert run(dt.date(2018, 9, 1)) == before
    assert run(dt.date(2023, 1, 1)) != modern


def test_nowcast_fit_and_performance_cannot_change_early_prompt_or_final_output(tmp_path, monkeypatch):
    """A usable invented pump-price case must stay independent before fit eligibility."""
    from copy import deepcopy
    from t4agent import nowcast

    eid = "FICTIONAL_CPI"
    ent = {"entity_id": eid, "name": "Invented gasoline CPI", "series_fred": "CUSR0000SETB01",
           "unit": "mom_pct_change_sa", "target_month": "2018-08"}
    cpi = {"doc_id": "SYNTHETIC_CPI_HISTORY", "doc_date": "2018-08-01",
           "text": "date | cpi_mom_pct_change_sa\n" + "\n".join(
               f"2018-{month:02} | {value}" for month, value in enumerate([-.2, .1, .3, -.1, .4, .2, .5], 1))}
    dates = [dt.date(2018, 7, 2) + dt.timedelta(days=7 * step) for step in range(9)]
    pump = {"doc_id": "SYNTHETIC_GASOLINE", "doc_date": "2018-08-28",
            "title": "Invented weekly retail gasoline price in dollars per gallon",
            "text": "Synthetic gasoline observations.\ndate | price\n" + "\n".join(
                f"{day} | {2.0 if day.month == 7 else 2.2}" for day in dates)}
    unit = write_unit(tmp_path / "nowcast", target_name="cpi_mom_pct_change_sa", entities=[ent],
                      documents=[(cpi, [eid]), (pump, [eid])])

    def snapshot(cutoff=None):
        item = {"entity_id": eid, "median": .6, "p05": -1, "p95": 2}
        f = forecaster(unit, [item], cutoff)
        f.plan = {"interval_quantity": "seasonally adjusted CPI month-over-month percent change",
                  "quantity_units": "percentage points", "interval_is_probability": False}
        parsed = f._one_call([eid], 0)
        forecasts = f.aggregate({eid: [parsed[eid]]})
        row = finalize(f.spec, forecasts[eid], f.plan, ent=f.spec.entity(eid),
                       table_half=table_halves(f.spec, f.corpus, f.ev)[eid])
        return f, row, f.client.captured

    old, old_output, old_prompt = snapshot()
    modern, modern_output, modern_prompt = snapshot(dt.date(2023, 1, 1))
    assert old.stat[eid].value == .5
    assert not isinstance(old.stat[eid], nowcast.PumpNowcast)
    # These corpus checks make the positive control meaningful: both months have
    # actual weekly rows, and the eligible production nowcast reads their means.
    assert isinstance(modern.stat[eid], nowcast.PumpNowcast)
    assert modern.stat[eid].weeks == 4
    assert modern.stat[eid].driver == pytest.approx(10)

    mutated = deepcopy(nowcast.NOWCAST)
    for fitted in mutated.values():
        fitted.update(beta=3.0, gap={month: 100.0 for month in range(1, 13)},
                      q90={2: 80.0, 3: 90.0}, mae=123.25, ar1_mae=234.5,
                      months="SYNTHETIC MUTATED FIT", n=1234)
    monkeypatch.setattr(nowcast, "NOWCAST", mutated)
    _, old_output_after, old_prompt_after = snapshot()
    _, modern_output_after, modern_prompt_after = snapshot(dt.date(2023, 1, 1))
    assert old_output_after == old_output
    assert old_prompt_after == old_prompt
    assert modern_output_after != modern_output
    assert modern_prompt_after != modern_prompt
    assert "SYNTHETIC MUTATED FIT" in json.dumps(modern_prompt_after)
    assert "SYNTHETIC MUTATED FIT" not in json.dumps(old_prompt_after)
