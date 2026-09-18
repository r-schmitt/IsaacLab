# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the shared OVStage owner."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import isaaclab_ov.ovstage_owner as owner_module
import pytest
from isaaclab_ov.ovstage_owner import OvStageOwner, ovstage_owner

USDA = "#usda 1.0\n"


class FakeSharedStage:
    """Records the lifecycle calls the owner is expected to make."""

    population_ordinal = 1

    def __init__(self, name: str, *, events: list, fail_population: bool = False) -> None:
        self.name = name
        self.stage = f"stage:{name}"
        self.destroyed = False
        self._events = events
        self._fail_population = fail_population
        events.append(("create", name))

    def populate_from_usda(self, usda: str, *, domains) -> None:
        self._events.append(("populate", usda, domains))
        if self._fail_population:
            raise RuntimeError("population failed")

    def seal(self, ordinal: int) -> None:
        self._events.append(("seal", ordinal))

    def destroy(self) -> None:
        self.destroyed = True
        self._events.append(("destroy", self.name))


@pytest.fixture
def events(monkeypatch) -> list:
    """Install a fake ``SharedOvStage`` and a minimal ``ovstage`` module, and record the calls."""
    recorded: list = []

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.PopulationDomain = SimpleNamespace(ALL="all")
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    monkeypatch.setattr(owner_module, "SharedOvStage", lambda name: FakeSharedStage(name, events=recorded))
    return recorded


def test_first_acquire_populates_and_seals_before_returning(events):
    """A consumer may attach immediately, so the population must already be sealed."""
    owner = OvStageOwner()

    shared = owner.acquire(USDA)

    assert owner.is_populated
    assert owner.consumer_count == 1
    # The seal must follow the population: a consumer attaching at an unsealed ordinal parses the
    # scene as empty rather than failing.
    assert events == [("create", "isaaclab"), ("populate", USDA, "all"), ("seal", 1)]
    assert shared.stage == "stage:isaaclab"


def test_second_consumer_joins_the_same_stage_without_repopulating(events):
    """The point of sharing: one population and one resident copy of the scene, not two."""
    owner = OvStageOwner()

    first = owner.acquire(USDA)
    events.clear()
    second = owner.acquire(USDA)

    assert second is first
    assert owner.consumer_count == 2
    assert events == []


def test_stage_survives_until_the_last_consumer_releases(events):
    """Physics closing must not pull the stage out from under a renderer still attached to it."""
    owner = OvStageOwner()
    shared = owner.acquire(USDA)
    owner.acquire(USDA)
    events.clear()

    owner.release()
    assert not shared.destroyed
    assert owner.consumer_count == 1
    assert events == []

    owner.release()
    assert shared.destroyed
    assert owner.consumer_count == 0
    assert not owner.is_populated
    assert events == [("destroy", "isaaclab")]


def test_release_is_idempotent_once_the_stage_is_gone(events):
    """A consumer may release a stage another consumer already finished off."""
    owner = OvStageOwner()
    owner.acquire(USDA)
    owner.release()
    events.clear()

    owner.release()

    assert events == []


def test_a_different_serialization_rebuilds_the_stage_once_it_is_free(events):
    """Reloading a stage between simulations must not reuse the previous scene."""
    owner = OvStageOwner()
    first = owner.acquire(USDA)
    owner.release()
    events.clear()

    second = owner.acquire("#usda 1.0\n# another stage\n")

    assert first.destroyed
    assert second is not first
    assert [event[0] for event in events] == ["create", "populate", "seal"]


def test_a_different_serialization_is_refused_while_a_consumer_holds_the_stage(events):
    """Repopulating underneath an attached consumer would leave it reading a freed scene."""
    owner = OvStageOwner()
    shared = owner.acquire(USDA)

    with pytest.raises(RuntimeError, match="still hold the one populated from the previous scene"):
        owner.acquire("#usda 1.0\n# another stage\n")

    assert not shared.destroyed
    assert owner.consumer_count == 1


def test_an_equal_but_distinct_serialization_still_shares(events):
    """A consumer that re-resolves an unchanged scene must join, not be turned away on identity."""
    owner = OvStageOwner()
    first = owner.acquire(USDA)
    events.clear()

    second = owner.acquire("".join(["#usda 1.0", "\n"]))

    assert second is first
    assert owner.consumer_count == 2
    assert events == []


def test_failed_population_registers_no_consumer_and_frees_the_allocation(monkeypatch):
    """Otherwise the next acquire hands back a stage nobody populated."""
    recorded: list = []
    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.PopulationDomain = SimpleNamespace(ALL="all")
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    monkeypatch.setattr(
        owner_module,
        "SharedOvStage",
        lambda name: FakeSharedStage(name, events=recorded, fail_population=True),
    )
    owner = OvStageOwner()

    with pytest.raises(RuntimeError, match="population failed"):
        owner.acquire(USDA)

    assert owner.consumer_count == 0
    assert not owner.is_populated
    assert ("destroy", "isaaclab") in recorded


def test_accessor_returns_one_process_wide_owner():
    """OVPhysX's runtime is a process-global singleton, so its stage has to be one too."""
    assert ovstage_owner() is ovstage_owner()
