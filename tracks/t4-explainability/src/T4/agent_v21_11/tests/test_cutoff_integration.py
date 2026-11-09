"""No external artifacts may affect pre-eligibility tasks, including prompts and scales.

These are noninterference tests, not predictive-quality measurements.
"""
import datetime as dt
from dataclasses import replace

import pytest

from test_offline import UNITS, _dated, _table_corpus, _v2

_v2()

from t4agent import config, stats
from t4agent.calibrate import table_halves
from t4agent.corpus import Corpus, Doc, Chunk
from t4agent.llm import HouseClient
from t4agent.predict import Evidence, Forecaster
from t4agent.task import load_task


def synthetic(cutoff, eid="UNSEEN", target="level_first_print", unit="units"):
    # Entire synthetic history predates the earliest cutoff tested. It is not a
    # redated public outcome or a Final task identifier.
    corpus = _table_corpus("date | value", _dated([100, 103, 102, 105, 106, 108, 109, 112]))
    text = corpus.docs["d"].text
    corpus = Corpus({"d": Doc("d", dt.date(2018, 9, 1), text, {})}, [Chunk(0, "d", 0, len(text), text)])
    ent = {"entity_id": eid, "units": unit}
    template = load_task(UNITS / "t4-cpicomp-202410-us11" / "task.json")
    spec = replace(template, task_id="unseen-transfer", target_type="regression",
                   target_name=target, cutoff=cutoff, cutoff_str=cutoff.isoformat() if cutoff else "",
                   resolution_str=(cutoff + dt.timedelta(days=14)).isoformat() if cutoff else "",
                   entities=[ent], entity_ids=[eid], labels=[], unit_dir=None)
    ev = Evidence(spec, corpus, {eid: ["d"]}, [])
    f = Forecaster(spec, corpus, ev, HouseClient(deadline=0), 0)
    return spec, corpus, ev, f


@pytest.mark.parametrize("cutoff", [None, dt.date(2019, 1, 1)])
@pytest.mark.parametrize("eid,target,unit", [
    ("H", "level_first_print", "thousands"),
    ("RENAMED", "new_metric_change", "units"),
    ("FX_NEW", "index_mom_pct_change", "index (target: percent change)"),
])
def test_old_tasks_do_not_call_selected_backends(monkeypatch, cutoff, eid, target, unit):
    def forbidden(*args, **kwargs):
        pytest.fail("a selected historical backend was reached")
    monkeypatch.setattr(stats, "forecast", forbidden)
    monkeypatch.setattr(stats, "rw_k", forbidden)
    spec, corpus, ev, f = synthetic(cutoff, eid, target, unit)
    assert f.stat[eid] is not None
    assert f.stat[eid].n_backtest == 0
    assert f.stat[eid].mae is None
    assert table_halves(spec, corpus, ev)[eid] > 0
    text = f.baselines([eid]).lower()
    assert "out-of-sample" not in text and "best simple" not in text


def test_old_to_new_to_old_has_no_policy_state_leak(monkeypatch):
    def snapshot(day):
        spec, corpus, ev, f = synthetic(day)
        return f.stat["UNSEEN"].value, table_halves(spec, corpus, ev), f.baselines(["UNSEEN"])
    old = dt.date(2019, 1, 1)
    new = dt.date(2023, 1, 1)
    before, modern = snapshot(old), snapshot(new)
    monkeypatch.setattr(stats, "rw_k", lambda *args: 9999.0)
    monkeypatch.setattr(stats, "_inflation", lambda *args: 9999.0)
    assert snapshot(old) == before
    assert snapshot(new) != modern  # positive control: mutation must be observable
    assert snapshot(old) == before


def test_selected_blend_weight_is_per_task_and_not_environment_override(monkeypatch):
    for weight in (0.01, 0.99):
        monkeypatch.setattr(config, "STAT_WEIGHT", weight)
        assert synthetic(None)[3].stat_weight == 0.0
        assert synthetic(dt.date(2019, 12, 31))[3].stat_weight == 0.0
        assert synthetic(dt.date(2020, 1, 1))[3].stat_weight == weight


def test_low_level_artifact_entrypoints_default_to_denied(monkeypatch):
    from t4agent import eps, nowcast, revisions
    def forbidden(*args, **kwargs):
        pytest.fail("historical artifact parser reached without an eligible policy")
    monkeypatch.setattr(eps, "target_period", forbidden)
    monkeypatch.setattr(nowcast, "_code", forbidden)
    monkeypatch.setattr(revisions, "_field", forbidden)
    assert eps.eps_anchor(None, set(), {}, "level") is None
    assert nowcast.pump_nowcast(None, [], {}, "", None) is None
    assert revisions.revision_prior(None, set(), {}) is None


def test_model_plan_cannot_enable_ineligible_eps_or_priors(monkeypatch):
    from t4agent import eps, nowcast, revisions
    def forbidden(*args, **kwargs):
        pytest.fail("model content bypassed the task-cutoff gate")
    monkeypatch.setattr(eps, "target_period", forbidden)
    monkeypatch.setattr(nowcast, "_code", forbidden)
    monkeypatch.setattr(revisions, "_field", forbidden)
    spec, corpus, ev, f = synthetic(dt.date(2019, 1, 1))
    f.plan = {"interval_quantity": "EPS", "quantity_units": "level", "cutoff": "2099-01-01"}
    f.build_anchors()
    assert not f.anchor and not f.prior
