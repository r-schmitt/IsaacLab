# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVStage control/output ordinal lanes."""

from __future__ import annotations

import importlib.util

import pytest

_REQUIRED_MODULES = ("isaaclab_ov",)
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.ovstage_ordinals import POPULATION_ORDINAL, OvStageOrdinalLanes  # noqa: E402
else:
    POPULATION_ORDINAL = None
    OvStageOrdinalLanes = None


def test_both_lanes_start_above_the_sealed_population_ordinal():
    """Population seals its own ordinal before either consumer attaches, so lanes start above it."""
    lanes = OvStageOrdinalLanes(POPULATION_ORDINAL)

    assert lanes.next_control() > POPULATION_ORDINAL
    assert lanes.next_output() > POPULATION_ORDINAL


def test_ordinals_increase_monotonically_across_both_lanes():
    """One counter feeds both lanes, so neither can write beneath a floor the other sealed."""
    lanes = OvStageOrdinalLanes(POPULATION_ORDINAL)

    reserved = [
        lanes.next_control(),
        lanes.next_output(),
        lanes.next_output(),
        lanes.next_control(),
        lanes.next_output(),
    ]

    assert reserved == sorted(reserved)
    assert len(set(reserved)) == len(reserved)


def test_lanes_never_reserve_the_same_ordinal_twice():
    """An ordinal carrying output must never also be a control ordinal."""
    lanes = OvStageOrdinalLanes(POPULATION_ORDINAL)

    control = {lanes.next_control() for _ in range(16)}
    output = {lanes.next_output() for _ in range(16)}

    assert control.isdisjoint(output)


def test_drain_range_covers_only_the_current_control_ordinal():
    """Naming just the new ordinal keeps producer ownership explicit."""
    lanes = OvStageOrdinalLanes(POPULATION_ORDINAL)
    ordinal = lanes.next_control()

    assert lanes.drain_range(ordinal) == (ordinal, ordinal)


def test_drain_range_rejects_an_output_ordinal():
    """Draining an output ordinal would re-ingest simulation output as an authored edit."""
    lanes = OvStageOrdinalLanes(POPULATION_ORDINAL)
    lanes.next_control()
    output = lanes.next_output()

    with pytest.raises(ValueError, match="not the current control ordinal"):
        lanes.drain_range(output)


def test_drain_range_rejects_a_superseded_control_ordinal():
    """A stale ordinal names edits OVPhysX already consumed, which hides the new ones."""
    lanes = OvStageOrdinalLanes(POPULATION_ORDINAL)
    stale = lanes.next_control()
    lanes.next_control()

    with pytest.raises(ValueError, match="not the current control ordinal"):
        lanes.drain_range(stale)


def test_latest_lane_ordinals_are_unset_before_the_first_reservation():
    lanes = OvStageOrdinalLanes(POPULATION_ORDINAL)

    assert lanes.latest_control is None
    assert lanes.latest_output is None


def test_latest_lane_ordinals_track_the_most_recent_reservation():
    lanes = OvStageOrdinalLanes(POPULATION_ORDINAL)
    control = lanes.next_control()
    output = lanes.next_output()

    assert lanes.latest_control == control
    assert lanes.latest_output == output


@pytest.mark.parametrize("population_ordinal", [0, -1])
def test_a_non_positive_population_ordinal_is_rejected(population_ordinal: int):
    """OVStage ordinals start at 1, so there is no valid lane below it."""
    with pytest.raises(ValueError, match="positive OVStage ordinal"):
        OvStageOrdinalLanes(population_ordinal)
