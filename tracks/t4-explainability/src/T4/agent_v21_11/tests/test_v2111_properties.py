"""Property checks for the v21.11 planner-flag and canary guards.

Axes from TASK_GROK G1: target_type × planner mark (none / false / true) × dimension
word in the target name × row baseline inside or outside [0, 1] × interval_leg × a
probability column. "none" is represented twice: no plan, and a plan that omits the mark.
"""
from __future__ import annotations

import importlib
import itertools
import json
import sys
from pathlib import Path

import pytest

T4 = Path(__file__).resolve().parents[2]
ROOT11 = T4 / "agent_v21_11"
ROOT10 = T4 / "agent_v21_10"
GUID = "11111111-2222-4333-8444-555555555555"

TYPES = ("classification", "regression", "ranking")
# 无 / false / true. 无: no plan, and a plan whose mark is absent.
MARKS = (
    ("none", None),
    ("absent", {}),
    ("false", {"interval_is_probability": False}),
    ("true", {"interval_is_probability": True}),
)
NAME_DIM = (False, True)
BASELINE = ("inside", "outside")
LEG = (False, True)
PROB_COL = (False, True)
# Identical shared unit strings _text_numeric already treats as measured. "%" is separate.
SHARED_UNIT = (None, "bps", "percent", "ratio", "level")
MEDS = {"inside": (0.2, 0.5, 0.8), "outside": (-2.5, 3.5, 8.0)}


def _load(root: Path):
    root_s = str(root)
    for key in list(sys.modules):
        if key == "t4agent" or key.startswith("t4agent."):
            del sys.modules[key]
    sys.path[:] = [p for p in sys.path if Path(p).name not in {"agent_v21_10", "agent_v21_11"}]
    sys.path.insert(0, root_s)
    return (
        importlib.import_module("t4agent.calibrate"),
        importlib.import_module("t4agent.predict"),
        importlib.import_module("t4agent.task"),
        importlib.import_module("t4agent.answer"),
    )


def _libs():
    """The v21.11 package. cwd of the mandated pytest command puts it on sys.path."""
    cal = importlib.import_module("t4agent.calibrate")
    pred = importlib.import_module("t4agent.predict")
    task = importlib.import_module("t4agent.task")
    ans = importlib.import_module("t4agent.answer")
    loaded = Path(cal.__file__).resolve()
    assert loaded.parents[1] == ROOT11, loaded
    return cal, pred, task, ans


def _spec(task, *, tt, name_dim, baseline, leg, prob_col, unit, name=None):
    meds = MEDS[baseline]
    rows = []
    for i, m in enumerate(meds):
        row = {"entity_id": f"E{i}", "latest_value": m}
        if unit:
            row["unit"] = unit
        if prob_col:
            row["implied_prob"] = 0.3  # matches _PROB_COLUMN, not _LEVEL_FIELD
        rows.append(row)
    if name is None:
        name = "bid_to_cover_ratio" if name_dim else "p_default"
    labels = ["yes", "no"] if tt == "classification" else []
    return task.TaskSpec(
        raw={}, task_id="t", target_type=tt, target_type_trusted=True, target_name=name,
        labels=labels, prompt="", cutoff=None, cutoff_str="", resolution_str="",
        interval_level=0.9, entities=rows, entity_ids=[r["entity_id"] for r in rows],
        unit_dir=None, card_params={"interval_leg": leg},
    )


def _mark_true(plan) -> bool:
    return bool(plan) and plan.get("interval_is_probability") is True


def _cases():
    for tt, (mark, plan), name_dim, baseline, leg, prob_col, unit in itertools.product(
        TYPES, MARKS, NAME_DIM, BASELINE, LEG, PROB_COL, SHARED_UNIT
    ):
        yield tt, mark, plan, name_dim, baseline, leg, prob_col, unit


def test_enumeration_i1_i2_i3_i4():
    """I1 when the mark is not true; I2 on every ranking; I3 when the name or a shared
    measured unit (bps / percent / ratio / level) says so; I4 when a classification
    card has interval_leg false. A clean probability regression stays on [0, 1]."""
    cal, pred, task, _ans = _libs()
    i1_bad, i2_bad, i3_bad, i4_bad, prob_bad = [], [], [], [], []
    n = 0
    for tt, mark, plan, name_dim, baseline, leg, prob_col, unit in _cases():
        n += 1
        spec = _spec(task, tt=tt, name_dim=name_dim, baseline=baseline, leg=leg, prob_col=prob_col, unit=unit)
        # MARKS tuples are shared across the product; numeric_target_plan writes the dict in place.
        plan_i = None if plan is None else dict(plan)
        outs = []
        for i, m in enumerate(MEDS[baseline]):
            fc = pred.Forecast(f"E{i}", label="yes", median=m, p05=m - 0.1, p95=m + 0.1)
            ent = spec.entity(f"E{i}")
            out = cal.finalize(spec, fc, plan_i, ent=ent)
            outs.append(out)
            where = f"{tt} mark={mark} name_dim={name_dim} base={baseline} leg={leg} prob_col={prob_col} unit={unit!r} med={m}"
            if not _mark_true(plan_i):
                if cal.is_probability(plan_i, spec) != cal.is_probability(None, spec):
                    i1_bad.append(f"I1 is_probability {where}: {cal.is_probability(plan_i, spec)} != {cal.is_probability(None, spec)}")
                other = cal.finalize(spec, fc, None, ent=ent)
                if out != other:
                    i1_bad.append(f"I1 finalize {where}: {out} != {other}")
                snapshot = None if plan_i is None else dict(plan_i)
                changed = cal.numeric_target_plan(None if plan_i is None else plan_i, spec)
                after = None if plan_i is None else dict(plan_i)
                if changed or after != snapshot:
                    i1_bad.append(f"I1 numeric_target_plan mutated {where}: {snapshot} -> {after}")
            if tt == "ranking" and (out["lo"], out["hi"]) == (0.0, 1.0):
                i2_bad.append(f"I2 interval {where}: {out}")
            measured = name_dim or unit is not None
            if tt == "regression" and measured:
                if (out["lo"], out["hi"]) == (0.0, 1.0):
                    i3_bad.append(f"I3 interval {where}: {out}")
                if m > 1 and not out["point_forecast"] > 1:
                    i3_bad.append(f"I3 clamp-hi {where}: {out}")
                if m < 0 and not out["point_forecast"] < 0:
                    i3_bad.append(f"I3 clamp-lo {where}: {out}")
            if tt == "classification" and leg is False and (out["lo"], out["hi"]) != (0.0, 1.0):
                i4_bad.append(f"I4 {where}: {out}")
            if (tt == "regression" and not name_dim and unit is None and _mark_true(plan) and baseline == "inside"):
                if (out["lo"], out["hi"]) != (0.0, 1.0) or abs(out["point_forecast"] - m) > 0.02:
                    prob_bad.append(f"clean probability {where}: {out}")
        if tt == "ranking":
            pts = [o["point_forecast"] for o in outs]
            if pts != sorted(pts) or len(set(pts)) != len(pts):
                i2_bad.append(
                    f"I2 order mark={mark} name_dim={name_dim} base={baseline} leg={leg} "
                    f"prob_col={prob_col} unit={unit!r}: meds={MEDS[baseline]} pts={pts}"
                )
    print(f"enumeration_cases={n} rows={n * 3} i1={len(i1_bad)} i2={len(i2_bad)} i3={len(i3_bad)} i4={len(i4_bad)} clean_prob={len(prob_bad)}")
    assert not i1_bad, "\n".join(i1_bad[:12])
    assert not i2_bad, "\n".join(i2_bad[:12])
    assert not i3_bad, "\n".join(i3_bad[:12])
    assert not i4_bad, "\n".join(i4_bad[:12])
    assert not prob_bad, "\n".join(prob_bad[:12])


def test_i1_matches_v21_10_when_mark_is_not_true():
    """Planner-normal path: flag absent or false. v21.11 finalize and is_probability
    match v21.10 on the same axes, including a plan that carries other planner fields."""
    cal11, pred11, task11, _ans11 = _libs()
    try:
        cal10, _pred10, task10, _ans10 = _load(ROOT10)
        # Reload names; _load swapped sys.modules under us.
        cal11, pred11, task11, _ans11 = _load(ROOT11)
        bad = []
        n = 0
        marks = (
            ("none", None),
            ("absent", {}),
            ("false", {"interval_is_probability": False}),
            ("rich", {
                "interval_is_probability": False,
                "interval_quantity": "level of the series",
                "baseline_field": "latest_value",
                "typical_abs_deviation": 0.4,
                "decimals": 2,
            }),
        )
        for tt, (mark, plan), name_dim, baseline, leg, prob_col in itertools.product(
            TYPES, marks, NAME_DIM, BASELINE, LEG, PROB_COL
        ):
            kw = dict(tt=tt, name_dim=name_dim, baseline=baseline, leg=leg, prob_col=prob_col, unit=None)
            s11 = _spec(task11, **kw)
            s10 = _spec(task10, **kw)
            for i, m in enumerate(MEDS[baseline]):
                fc = pred11.Forecast(f"E{i}", label="yes", median=m, p05=m - 0.1, p95=m + 0.1)
                a = cal11.finalize(s11, fc, plan, ent=s11.entity(f"E{i}"))
                b = cal10.finalize(s10, fc, plan, ent=s10.entity(f"E{i}"))
                n += 1
                if a != b or cal11.is_probability(plan, s11) != cal10.is_probability(plan, s10):
                    bad.append(f"{tt} {mark} name_dim={name_dim} base={baseline} leg={leg} prob_col={prob_col} med={m} v11={a} v10={b}")
        print(f"cross_version_rows={n} mismatches={len(bad)}")
        assert not bad, "\n".join(bad[:8])
    finally:
        _load(ROOT11)


def test_i3_percent_sign_is_a_measured_unit():
    """I3: a regression whose target name or unit column carries a dimension word, including
    the '%' the strategy lists next to bps / ratio / level, never submits [0, 1]."""
    cal, pred, task, _ans = _libs()
    plan = {"interval_is_probability": True}
    bad = []
    samples = [
        ("unit", "%", "p_default", [{"entity_id": "X", "unit": "%", "latest_value": 0.4}], 0.4),
        ("unit", "％", "p_default", [{"entity_id": "X", "unit": "％", "latest_value": 0.4}], 0.4),
        ("unit", "pp", "p_default", [{"entity_id": "X", "unit": "pp", "latest_value": 0.4}], 0.4),
        ("unit", "per cent", "p_default", [{"entity_id": "X", "unit": "per cent", "latest_value": 0.4}], 0.4),
        ("name", "%", "cpi_%", [{"entity_id": "X", "latest_value": 0.4}], 0.4),
        ("name", "%", "move_%", [{"entity_id": "X", "latest_value": 0.4}], 0.4),
    ]
    for kind, word, name, rows, med in samples:
        spec = task.TaskSpec(
            raw={}, task_id="t", target_type="regression", target_type_trusted=True, target_name=name,
            labels=[], prompt="", cutoff=None, cutoff_str="", resolution_str="", interval_level=0.9,
            entities=rows, entity_ids=["X"], unit_dir=None, card_params={"interval_leg": True},
        )
        out = cal.finalize(spec, pred.Forecast("X", median=med), plan, ent=spec.entity("X"))
        flag = cal.is_probability(plan, spec)
        if flag or (out["lo"], out["hi"]) == (0.0, 1.0) or not (out["point_forecast"] == pytest.approx(med) or out["lo"] < med < out["hi"]):
            # A numeric reading may round the point; it must not be the probability interval.
            if (out["lo"], out["hi"]) == (0.0, 1.0) or flag:
                bad.append(f"{kind} {word!r} name={name} is_probability={flag} out={out}")
    assert not bad, "\n".join(bad)


def test_i3_heterogeneous_dimension_unit_strings():
    """I3: every row unit is a dimension word, even when the strings are not identical."""
    cal, pred, task, _ans = _libs()
    rows = [
        {"entity_id": "A", "unit": "bps", "latest_value": 0.4},
        {"entity_id": "B", "unit": "basis points", "latest_value": 0.5},
    ]
    spec = task.TaskSpec(
        raw={}, task_id="t", target_type="regression", target_type_trusted=True, target_name="outcome",
        labels=[], prompt="", cutoff=None, cutoff_str="", resolution_str="", interval_level=0.9,
        entities=rows, entity_ids=["A", "B"], unit_dir=None, card_params={"interval_leg": True},
    )
    plan = {"interval_is_probability": True}
    out = cal.finalize(spec, pred.Forecast("A", median=0.4), plan, ent=spec.entity("A"))
    ip = cal.is_probability(plan, spec)
    assert ip is False and (out["lo"], out["hi"]) != (0.0, 1.0), f"is_probability={ip} out={out}"


def test_true_probability_row_is_not_flipped_by_a_sibling_baseline():
    """A p_default regression the planner marks as a probability stays [0, 1] on a row whose
    own median and baseline are inside [0, 1]. Another row's out-of-range baseline must not
    pull this row onto the numeric path (strategy: only that row's median does)."""
    cal, pred, task, _ans = _libs()
    rows = [
        {"entity_id": "A", "latest_value": 0.2},
        {"entity_id": "B", "latest_value": 1.5},
    ]
    spec = task.TaskSpec(
        raw={}, task_id="t", target_type="regression", target_type_trusted=True, target_name="p_default",
        labels=[], prompt="", cutoff=None, cutoff_str="", resolution_str="", interval_level=0.9,
        entities=rows, entity_ids=["A", "B"], unit_dir=None, card_params={},
    )
    plan = {"interval_is_probability": True}
    out = cal.finalize(spec, pred.Forecast("A", median=0.2), plan, ent=spec.entity("A"))
    ip = cal.is_probability(plan, spec)
    assert ip is True and (out["lo"], out["hi"]) == (0.0, 1.0) and out["point_forecast"] == pytest.approx(0.2), (
        f"is_probability={ip} out={out}"
    )


def test_true_probability_is_not_flipped_by_a_calendar_column():
    """latest_year is not the probability. A median inside [0, 1] stays on [0, 1]."""
    cal, pred, task, _ans = _libs()
    ent = {"entity_id": "X", "latest_year": 2024, "implied_prob": 0.2}
    spec = task.TaskSpec(
        raw={}, task_id="t", target_type="regression", target_type_trusted=True, target_name="p_default",
        labels=[], prompt="", cutoff=None, cutoff_str="", resolution_str="", interval_level=0.9,
        entities=[ent], entity_ids=["X"], unit_dir=None, card_params={},
    )
    plan = {"interval_is_probability": True}
    out = cal.finalize(spec, pred.Forecast("X", median=0.2), plan, ent=ent)
    ip = cal.is_probability(plan, spec)
    base = pred.baseline_value(spec, ent, plan)
    assert ip is True and (out["lo"], out["hi"]) == (0.0, 1.0) and out["point_forecast"] == pytest.approx(0.2), (
        f"is_probability={ip} baseline={base} out={out}"
    )


def test_problem_count_column_does_not_force_a_probability_interval():
    """_PROB_COLUMN is a substring. 'problem_count' is not a probability column, so an EPS
    row outside [0, 1] must keep a numeric interval when the planner flag is the only signal."""
    cal, pred, task, _ans = _libs()
    rows = [{"entity_id": "A", "consensus_eps": 2.5, "problem_count": 3}]
    spec = task.TaskSpec(
        raw={}, task_id="t", target_type="classification", target_type_trusted=True, target_name="eps_outcome",
        labels=["beat", "miss"], prompt="", cutoff=None, cutoff_str="", resolution_str="",
        interval_level=0.9, entities=rows, entity_ids=["A"], unit_dir=None,
        card_params={"interval_leg": True},
    )
    plan = {"interval_is_probability": True}
    out = cal.finalize(spec, pred.Forecast("A", label="beat", median=2.2, p05=2.0, p95=2.4), plan, ent=spec.entity("A"))
    assert (out["lo"], out["hi"]) != (0.0, 1.0), out
    assert out["point_forecast"] > 1, out


def _answer_spec(task, raw_entities):
    return task.TaskSpec(
        raw={"entities": raw_entities}, task_id="t", target_type="classification", target_type_trusted=True,
        target_name="outcome", labels=["yes", "no"], prompt="", cutoff=None, cutoff_str="",
        resolution_str="", interval_level=0.9, entities=raw_entities,
        entity_ids=[e["entity_id"] for e in raw_entities], unit_dir=None, card_params={},
        canaries=(GUID,),
    )


def _blob(ans_obj) -> str:
    return json.dumps(ans_obj, ensure_ascii=False)


def test_i5_canary_stripped_from_claims_reasons_and_trace(tmp_path):
    """Quoted canary text is dropped, case-insensitively. A task-row fallback that has a
    numeric pair quotes that pair and does not carry the guid. card_canaries reads the guid."""
    _cal, _pred, task, ans = _libs()
    card = tmp_path / "card.toml"
    card.write_text(
        '[contamination]\ncanary_guid = "%s"\nshort = "abc"\nn = 1\n' % GUID,
        encoding="utf-8",
    )
    assert task.card_canaries(tmp_path) == (GUID,)
    assert task.card_canaries(None) == ()
    raw = [{"entity_id": "X", "latest_value": 1.5, "commentary": GUID}]
    spec = _answer_spec(task, raw)
    claim = {"doc_id": "d1", "span_start": 0, "span_end": 8, "claim": f"see {GUID}"}
    upper = {"doc_id": "d1", "span_start": 0, "span_end": 8, "claim": GUID.upper()}
    clean = {"doc_id": "d1", "span_start": 0, "span_end": 5, "claim": "hello"}
    row = {"lo": 0.0, "hi": 1.0, "point_forecast": 0.4, "label": "yes"}
    for label, rows, trace, reasons in (
        ("claim", {"X": {**row, "claims": [claim]}}, "", None),
        ("upper", {"X": {**row, "claims": [upper]}}, "", None),
        ("reason", {"X": {**row, "claims": [clean]}}, "",
         [{"reason_id": "r", "premise": GUID, "mechanism": "m", "answer_implication": "a"}]),
        ("trace", {"X": {**row, "claims": [clean]}}, f"trace {GUID}", None),
        ("safe-fallback", {"X": {**row, "claims": [claim]}}, "", None),
    ):
        built = ans.build_answer(spec, rows, trace, reasons, None)
        text = _blob(built)
        assert GUID.lower() not in text.lower(), f"{label}: {text}"
    # the safe fallback quotes the numeric pair, not the commentary
    built = ans.build_answer(spec, {"X": {**row, "claims": [claim]}}, "", None, None)
    assert "latest_value" in _blob(built)


def test_i5_task_row_fallback_does_not_reintroduce_the_canary():
    """When every claim is dropped and the task row's only quotable field contains the guid,
    build_answer must still not emit it."""
    _cal, _pred, task, ans = _libs()
    raw = [{"entity_id": "X", "commentary": f"note {GUID}"}]
    spec = _answer_spec(task, raw)
    bad = {"doc_id": "d1", "span_start": 0, "span_end": 8, "claim": GUID}
    built = ans.build_answer(
        spec, {"X": {"lo": 0.0, "hi": 1.0, "point_forecast": 0.4, "claims": [bad]}}, "", None, None
    )
    text = _blob(built)
    errs = ans.validate(spec, type("C", (), {"docs": {}})(), built)
    assert GUID.lower() not in text.lower(), f"answer={text} validate={errs}"


def test_i5_label_does_not_carry_the_canary():
    """I5 is the whole build_answer object. A label string is not filtered."""
    _cal, _pred, task, ans = _libs()
    raw = [{"entity_id": "X", "latest_value": 1.5}]
    spec = _answer_spec(task, raw)
    clean = {"doc_id": "d1", "span_start": 0, "span_end": 5, "claim": "hello"}
    built = ans.build_answer(
        spec,
        {"X": {"lo": 0.0, "hi": 1.0, "point_forecast": 0.4, "label": GUID, "claims": [clean]}},
        "", None, None,
    )
    text = _blob(built)
    assert GUID.lower() not in text.lower(), text
