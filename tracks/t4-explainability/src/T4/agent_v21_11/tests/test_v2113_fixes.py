"""v21.13 regression tests: model-stage guard at the LLM deadline, in-flight requests given up at the deadline,
truncated reasons salvaged, a leading minus on the first EPS figure kept."""
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from t4agent import facts
from t4agent.llm import HouseClient, LLMUnavailable, salvage_array

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT.parent / "track4-analysis-public" / "units"


def test_salvage_keeps_complete_objects_of_a_cut_array():
    text = '{"reasons": [{"premise_id": "P1", "mechanism": "a"}, {"premise_id": "P2", "mechanism": "b"}, {"premise_id": "P3", "mech'
    obj = salvage_array(text, "reasons")
    assert obj == {"reasons": [{"premise_id": "P1", "mechanism": "a"}, {"premise_id": "P2", "mechanism": "b"}]}


def test_salvage_returns_none_without_a_complete_object():
    assert salvage_array('{"reasons": [{"premise_id": "P1", "mecha', "reasons") is None
    assert salvage_array("no json here", "reasons") is None
    assert salvage_array("", "reasons") is None


def test_salvage_takes_the_last_array_and_tolerates_slips():
    text = 'draft {"reasons": [{"x": 1}]} final {"reasons": [{"x": 2, }, {"x": NaN}, {"x"'
    assert salvage_array(text, "reasons") == {"reasons": [{"x": 2}, {"x": None}]}


@pytest.mark.parametrize("lead,eaten", [(" -0.40", " "), (" - 0.40", " - "), (": -.40", ": "), (" — 0.40", " — "),
                                        ("Diluted -0.40", "Diluted ")])
def test_sublabel_keeps_the_minus_of_a_figure(lead, eaten):
    m = facts._SUBLABEL.match(lead)
    assert m is not None and m.group(0) == eaten


def test_first_column_minus_parses_negative():
    rest = " -0.40 0.80"
    m = facts._SUBLABEL.match(rest)
    vals = [facts._num(t) for t in __import__("re").findall(facts._VAL, rest[m.end():])]
    assert vals == [-0.40, 0.80]


class _Hang:
    def post(self, *a, **k):
        time.sleep(30)


def test_request_in_flight_is_given_up_at_the_llm_deadline(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("MODEL_TOKEN", "x")
    c = HouseClient(deadline=time.time() + 22.0)
    c._session = _Hang()
    t0 = time.time()
    with pytest.raises((LLMUnavailable, Exception)):
        c.chat([{"role": "user", "content": "hi"}], reasoning_budget=None, max_attempts=1)
    assert time.time() - t0 < 22.0 + 1.5  # was the deadline + 15 s


def test_closed_client_sends_nothing(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("MODEL_TOKEN", "x")
    c = HouseClient(deadline=time.time() + 300)
    c.close()
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "hi"}], reasoning_budget=None)
    assert c.used == 0


def test_hung_model_stage_is_abandoned_at_the_guard(tmp_path):
    """A model stage that never returns (fomc-2022, R1): the process exits 0 before the hard stop, the answer on disk
    is valid, and every stage start is logged."""
    unit = UNITS / "t4-fomc-curve-20220728"
    out = tmp_path / "answer.json"
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from t4agent import predict, cli
        predict.Forecaster.run_plan = lambda self: time.sleep(600)
        sys.exit(cli.main(["analyze", "--task", {str(unit / "task.json")!r}, "--corpus", {str(unit / "corpus")!r},
                           "--out", {str(out)!r}]))
    """)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "T4_"))}
    env.update(MODEL_ENDPOINT="http://127.0.0.1:9", MODEL_NAME="m", MODEL_TOKEN="x", PYTHONHASHSEED="0",
               T4_TIME_BUDGET="100", T4_LLM_DEADLINE="5")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=90)
    wall = time.time() - t0
    assert p.returncode == 0
    assert wall < 40  # guard = LLM deadline 5 s + 20 s grace, well before the 100 s hard stop
    assert "stage plan: start" in p.stderr
    assert "still running" in p.stderr and "fallback answer stands" in p.stderr
    import json
    ans = json.loads(out.read_text())
    assert len(ans["entity_predictions"]) == 6


def test_salvage_falls_back_to_an_earlier_array_when_the_last_has_no_complete_object():
    text = 'draft {"reasons": [{"x": 1}, {"x": 2}]} final {"reasons": [{"x":'
    assert salvage_array(text, "reasons") == {"reasons": [{"x": 1}, {"x": 2}]}


def _run_hung(tmp_path, patch: str):
    unit = UNITS / "t4-fomc-curve-20220728"
    out = tmp_path / "answer.json"
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from t4agent import predict, cli
        {patch}
        sys.exit(cli.main(["analyze", "--task", {str(unit / "task.json")!r}, "--corpus", {str(unit / "corpus")!r},
                           "--out", {str(out)!r}]))
    """)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MODEL_", "T4_"))}
    env.update(MODEL_ENDPOINT="http://127.0.0.1:9", MODEL_NAME="m", MODEL_TOKEN="x", PYTHONHASHSEED="0",
               T4_CAP_TIME_BUDGET="360", T4_CAP_LLM_DEADLINE="290")  # the production 360/290 caps
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=400)
    return p, time.time() - t0, out


def test_plan_hang_on_the_360_290_clock(tmp_path):
    p, wall, out = _run_hung(tmp_path, "predict.Forecaster.run_plan = lambda self: time.sleep(900)")
    assert p.returncode == 0
    assert "hard stop 360 s, LLM deadline 290 s" in p.stderr
    assert 305 <= wall < 340  # guard 290 + 20 = 310 s, before the 360 s hard stop
    assert "stage plan: start" in p.stderr and "stage plan still running" in p.stderr
    assert "fallback answer stands" in p.stderr
    import json
    assert len(json.loads(out.read_text())["entity_predictions"]) == 6


def test_reasons_hang_keeps_the_forecast_answer_on_disk(tmp_path):
    patch = ("predict.Forecaster.run_plan = lambda self: None\n"
             "        predict.Forecaster.run_analysts = lambda self: {}\n"
             "        _br = cli.build_reasons\n"
             "        cli.build_reasons = lambda *a, **k: time.sleep(900) if a[-1] is not None else _br(*a, **k)")
    p, wall, out = _run_hung(tmp_path, patch)
    assert p.returncode == 0
    assert 305 <= wall < 340
    for st in ("plan", "analysts", "extra-votes", "aggregate", "forecast-answer"):
        assert f"stage {st}: start" in p.stderr and f"stage {st}: end" in p.stderr
    assert "forecast answer written" in p.stderr
    assert "stage reasons still running" in p.stderr and "forecasts answer stands" in p.stderr
    import json
    ans = json.loads(out.read_text())
    assert len(ans["entity_predictions"]) == 6
    assert "House requests used" in ans["evidence_trace"]  # the model-stage answer, not the fallback (no client)
