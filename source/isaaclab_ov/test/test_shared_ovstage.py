# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the lifetime of an OVStage and the resources bound to it."""

from __future__ import annotations

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "ovstage")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]


@pytest.fixture()
def shared_stage_seams(monkeypatch: pytest.MonkeyPatch):
    """Replace stage and path-dictionary construction with fakes that record their teardown.

    A real stage would apply Isaac Lab's process-wide hierarchy model, which ovstage refuses to
    change while another stage is live. Faking the two seams keeps these lifetime assertions
    independent of whatever stage the rest of the session holds.
    """
    import ovstage

    from isaaclab_ov import stage as stage_module

    events: list[tuple[str, object]] = []

    class FakeStage:
        def __init__(self, name: str) -> None:
            self.name = name

        def advance_write_floor(self, *, ordinal: int):
            events.append(("seal", ordinal))
            return _FakeOp()

        def destroy(self) -> None:
            events.append(("destroy_stage", self.name))

    class FakePathDictionary:
        def __init__(self, stage) -> None:
            self.stage = stage

        def destroy(self) -> None:
            events.append(("destroy_paths", self.stage.name))

    class _FakeOp:
        def wait(self) -> None:
            return None

    monkeypatch.setattr(stage_module, "create_ovstage", FakeStage)
    monkeypatch.setattr(ovstage, "PathDictionary", FakePathDictionary)
    monkeypatch.setattr(
        ovstage.population,
        "open_usd_from_string",
        lambda stage, usda, ordinal, domains: events.append(("populate", (usda, ordinal, domains))),
    )
    return events


def test_teardown_destroys_the_path_dictionary_before_the_stage(shared_stage_seams: list):
    """The dictionary is created against the stage, so outliving it would leave it dangling."""
    from isaaclab_ov.stage import SharedOvStage

    SharedOvStage("lifetime.order").destroy()

    assert shared_stage_seams == [("destroy_paths", "lifetime.order"), ("destroy_stage", "lifetime.order")]


def test_teardown_is_idempotent(shared_stage_seams: list):
    """A consumer may close a stage another consumer already closed."""
    from isaaclab_ov.stage import SharedOvStage

    shared = SharedOvStage("lifetime.idempotent")
    shared.destroy()
    shared.destroy()

    assert shared_stage_seams.count(("destroy_stage", "lifetime.idempotent")) == 1


def test_exiting_the_context_destroys_the_stage(shared_stage_seams: list):
    from isaaclab_ov.stage import SharedOvStage

    with SharedOvStage("lifetime.context") as shared:
        assert shared.stage.name == "lifetime.context"

    assert ("destroy_stage", "lifetime.context") in shared_stage_seams


@pytest.mark.parametrize("attribute", ["stage", "paths"])
def test_accessing_a_destroyed_stage_raises_instead_of_handing_back_a_dangling_handle(
    shared_stage_seams: list, attribute: str
):
    from isaaclab_ov.stage import SharedOvStage

    shared = SharedOvStage("lifetime.destroyed")
    shared.destroy()

    with pytest.raises(RuntimeError, match="has been destroyed"):
        getattr(shared, attribute)


def test_a_failed_path_dictionary_does_not_leak_the_stage(monkeypatch: pytest.MonkeyPatch, shared_stage_seams: list):
    """Construction is all-or-nothing: a half-built wrapper has no owner to destroy it."""
    import ovstage

    from isaaclab_ov.stage import SharedOvStage

    def _failing_path_dictionary(stage):
        raise RuntimeError("no path dictionary for you")

    monkeypatch.setattr(ovstage, "PathDictionary", _failing_path_dictionary)

    with pytest.raises(RuntimeError, match="no path dictionary"):
        SharedOvStage("lifetime.failed")

    assert shared_stage_seams == [("destroy_stage", "lifetime.failed")]


def test_population_and_its_seal_land_on_the_population_ordinal(shared_stage_seams: list):
    """Consumers attach at the population ordinal, so the initial scene must be written there."""
    import ovstage

    from isaaclab_ov.stage import SharedOvStage

    shared = SharedOvStage("lifetime.populate")
    shared.populate_from_usda("#usda 1.0", domains=ovstage.PopulationDomain.ALL)
    shared.seal(shared.population_ordinal)

    assert shared_stage_seams == [
        ("populate", ("#usda 1.0", shared.population_ordinal, ovstage.PopulationDomain.ALL)),
        ("seal", shared.population_ordinal),
    ]


def test_population_leaves_room_for_later_edits(shared_stage_seams: list):
    """Setup writes share the population ordinal; both lanes must allocate above it."""
    from isaaclab_ov.stage import SharedOvStage

    shared = SharedOvStage("lifetime.lanes")

    assert shared.ordinals.next_control() > shared.population_ordinal
    assert shared.ordinals.next_output() > shared.population_ordinal
