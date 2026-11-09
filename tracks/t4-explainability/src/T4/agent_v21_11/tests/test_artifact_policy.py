"""Offline artifact eligibility: cutoff dates govern every task independently."""
from __future__ import annotations

import datetime as dt
from dataclasses import FrozenInstanceError

import pytest


BOUNDS = {
    "statistics": dt.date(2022, 2, 1),
    "nowcast": dt.date(2022, 1, 1),
    "revision": dt.date(2022, 1, 1),
    "eps": dt.date(2022, 1, 1),
    "blend": dt.date(2020, 1, 1),
}


@pytest.mark.parametrize("capability,bound", BOUNDS.items())
def test_artifact_is_denied_before_its_bound_and_allowed_on_and_after(capability, bound):
    from t4agent.artifact_policy import ArtifactPolicy

    assert not ArtifactPolicy(bound - dt.timedelta(days=1)).permits(capability)
    assert ArtifactPolicy(bound).permits(capability)
    assert ArtifactPolicy(bound + dt.timedelta(days=1)).permits(capability)


class DateSubclass(dt.date):
    pass


@pytest.mark.parametrize("cutoff", [
    None, "2024-01-01", "", "not-a-date", 20240101, True, False, 0.0,
    {}, [], dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
    DateSubclass(2024, 1, 1),
])
def test_missing_or_non_exact_date_cutoffs_fail_closed(cutoff):
    from t4agent.artifact_policy import ArtifactPolicy

    policy = ArtifactPolicy(cutoff)
    assert all(not policy.permits(capability) for capability in BOUNDS)


@pytest.mark.parametrize("capability", ["", "unknown", "Statistics", "all", "*", None, [], {}])
def test_unknown_capabilities_fail_closed_even_for_late_cutoffs(capability):
    from t4agent.artifact_policy import ArtifactPolicy

    assert not ArtifactPolicy(dt.date(2099, 1, 1)).permits(capability)


def test_capabilities_have_independent_availability_dates():
    from t4agent.artifact_policy import ArtifactPolicy

    policy = ArtifactPolicy(dt.date(2022, 1, 15))
    assert not policy.permits("statistics")
    assert all(policy.permits(capability) for capability in ("nowcast", "revision", "eps", "blend"))
    earlier = ArtifactPolicy(dt.date(2021, 1, 1))
    assert earlier.permits("blend")
    assert all(not earlier.permits(capability) for capability in ("statistics", "nowcast", "revision", "eps"))


def test_policy_and_registry_cannot_be_mutated():
    from t4agent.artifact_policy import ARTIFACTS, ArtifactPolicy

    policy = ArtifactPolicy(dt.date(2021, 1, 1))
    with pytest.raises(FrozenInstanceError):
        policy.cutoff = dt.date(2099, 1, 1)
    with pytest.raises(TypeError):
        ARTIFACTS["nowcast"] = ARTIFACTS["blend"]
    with pytest.raises(FrozenInstanceError):
        ARTIFACTS["nowcast"].eligible_from = dt.date(1900, 1, 1)
    assert not policy.permits("nowcast")


def test_environment_and_other_tasks_cannot_enable_ineligible_artifacts(monkeypatch):
    from t4agent.artifact_policy import ArtifactPolicy

    monkeypatch.setenv("T4_ARTIFACT_POLICY", "allow")
    monkeypatch.setenv("T4_ARTIFACT_CUTOFF", "2099-01-01")
    monkeypatch.setenv("T4_TASK_ID", "t4-cpicomp-202410-us11")
    early = ArtifactPolicy(dt.date(2019, 12, 31))
    late = ArtifactPolicy(dt.date(2024, 1, 1))
    missing = ArtifactPolicy(None)
    for policy, expected in ((late, True), (early, False), (late, True), (missing, False), (early, False)):
        assert all(policy.permits(capability) is expected for capability in BOUNDS)


def test_registry_documents_conservative_bounds_and_all_dependency_groups():
    from t4agent.artifact_policy import ARTIFACTS

    assert set(ARTIFACTS) == set(BOUNDS)
    for capability, bound in BOUNDS.items():
        artifact = ARTIFACTS[capability]
        assert artifact.capability == capability
        assert artifact.eligible_from == bound
        assert artifact.data_available and artifact.conservative_bound
        assert artifact.evidence and artifact.includes
        assert isinstance(artifact.evidence, tuple)
        assert isinstance(artifact.includes, tuple)
