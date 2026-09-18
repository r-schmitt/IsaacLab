# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single host-stage serialization that OVStage consumers share."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "ovstage", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]


@pytest.fixture()
def two_env_stage():
    """A stage with two environment roots, each holding a distinguishable cube."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    for env_index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_index}")
        UsdGeom.Cube.Define(stage, f"/World/envs/env_{env_index}/Cube_{env_index}")
    return stage


def _clone_plan(*sources: str, num_envs: int = 2):
    """A clone plan replicating ``sources`` across ``num_envs`` environments."""
    from isaaclab.cloner import ClonePlan

    return ClonePlan(
        sources=sources,
        destinations=tuple(f"/World/envs/env_{index}" for index in range(num_envs)),
        clone_mask=np.ones((len(sources), num_envs), dtype=bool),
        env_ids=np.arange(num_envs, dtype=np.int32),
    )


def test_resolve_keeps_every_source_row_and_drops_other_env_roots(two_env_stage):
    """Trimming is driven by the plan's sources, not by a fixed ``env_0``."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    usda = SharedStageUsda().resolve(two_env_stage, _clone_plan("/World/envs/env_1"))

    assert "Cube_1" in usda
    assert "Cube_0" not in usda


def test_resolve_keeps_all_sources_for_heterogeneous_prototypes(two_env_stage):
    """Both prototypes survive when the plan names both as sources."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    usda = SharedStageUsda().resolve(two_env_stage, _clone_plan("/World/envs/env_0", "/World/envs/env_1"))

    assert "Cube_0" in usda
    assert "Cube_1" in usda


def test_resolve_without_a_clone_plan_trims_nothing(two_env_stage):
    """A stage built without a clone plan is serialized whole."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    usda = SharedStageUsda().resolve(two_env_stage, None)

    assert "Cube_0" in usda
    assert "Cube_1" in usda


def test_resolve_serializes_once_and_returns_the_same_text(two_env_stage):
    """A second consumer gets the first serialization rather than a fresh one."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    from pxr import UsdGeom

    shared = SharedStageUsda()
    plan = _clone_plan("/World/envs/env_0")
    first = shared.resolve(two_env_stage, plan)

    # Author after the first resolve: the second consumer must not see it, or the two consumers
    # would be populating different scenes.
    UsdGeom.Cube.Define(two_env_stage, "/World/envs/env_0/LateCube")

    assert shared.resolve(two_env_stage, plan) == first
    assert "LateCube" not in first


def test_contributions_follow_the_serialized_stage(two_env_stage):
    """Contributed text is appended in order, after the host stage."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    shared = SharedStageUsda()
    shared.contribute('def Scope "RenderFirst" {}')
    shared.contribute('def Scope "RenderSecond" {}')

    usda = shared.resolve(two_env_stage, _clone_plan("/World/envs/env_0"))

    assert usda.index("Cube_0") < usda.index("RenderFirst") < usda.index("RenderSecond")


def test_contribute_after_resolve_raises(two_env_stage):
    """A contribution that could not reach the consumers is an error, not a silent drop."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    shared = SharedStageUsda()
    shared.resolve(two_env_stage, _clone_plan("/World/envs/env_0"))

    with pytest.raises(RuntimeError, match="already resolved"):
        shared.contribute('def Scope "TooLate" {}')


def test_resolve_with_a_different_stage_raises(two_env_stage):
    """Handing out a serialization of the wrong stage is an error, not a stale success."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    from pxr import Usd

    shared = SharedStageUsda()
    shared.resolve(two_env_stage, _clone_plan("/World/envs/env_0"))

    with pytest.raises(RuntimeError, match="different USD stage"):
        shared.resolve(Usd.Stage.CreateInMemory(), None)


def test_invalidate_drops_the_text_and_the_contributions(two_env_stage):
    """After invalidation the next resolve starts over, without the stale contributions."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    shared = SharedStageUsda()
    shared.contribute('def Scope "Stale" {}')
    shared.resolve(two_env_stage, _clone_plan("/World/envs/env_0"))

    shared.invalidate()
    assert not shared.is_resolved

    usda = shared.resolve(two_env_stage, _clone_plan("/World/envs/env_0"))
    assert "Stale" not in usda


def test_invalidate_is_idempotent():
    """Tearing down twice is not an error."""
    from isaaclab_ov.stage_usda import SharedStageUsda

    shared = SharedStageUsda()
    shared.invalidate()
    shared.invalidate()

    assert not shared.is_resolved


def test_shared_accessor_returns_one_instance():
    """Consumers reach the same serialization without passing it to each other."""
    from isaaclab_ov.stage_usda import shared_stage_usda

    assert shared_stage_usda() is shared_stage_usda()
