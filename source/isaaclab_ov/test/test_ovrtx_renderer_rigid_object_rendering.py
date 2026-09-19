# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX adapter for the shared rigid-object rendering contract."""

import importlib.util
import sys
from pathlib import Path

import pytest

from isaaclab.sim import SimulationCfg, build_simulation_context

_CONTRACT_DIR = Path(__file__).resolve().parents[2] / "isaaclab" / "test" / "renderers"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

from rigid_object_rendering_contract import (  # noqa: E402
    RigidObjectRenderingBackend,
    run_rigid_object_scale_and_pose_rendering_contract,
)

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "ovphysx", "isaaclab_newton", "newton")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]
_OVSTAGE_AVAILABLE = importlib.util.find_spec("ovstage") is not None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.rendering,
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(bool(_MISSING_MODULES), reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}"),
]

if not _MISSING_MODULES:
    from isaaclab_newton.physics import NewtonManager  # noqa: E402
    from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
else:
    NewtonManager = None
    OVRTXRendererCfg = None
    OvPhysxCfg = None


def _record_ovstage_acquisitions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the shared stage's consumer count after every acquisition during a run.

    Args:
        monkeypatch: Fixture used to wrap the owner's acquire for the duration of the test.

    Returns:
        Consumer count after each acquisition, in order, filled as the run proceeds.
    """
    from isaaclab_ov.ovstage_owner import OvStageOwner

    counts: list[int] = []
    acquire = OvStageOwner.acquire

    def recording_acquire(self, stage_usda: str):
        shared = acquire(self, stage_usda)
        counts.append(self.consumer_count)
        return shared

    monkeypatch.setattr(OvStageOwner, "acquire", recording_acquire)
    return counts


@pytest.mark.parametrize(
    "use_ovstage",
    [
        pytest.param(False, id="legacy"),
        pytest.param(
            True,
            id="ovstage",
            marks=pytest.mark.skipif(not _OVSTAGE_AVAILABLE, reason="requires optional module: ovstage"),
        ),
    ],
)
def test_kinematic_rigid_object_scale_and_pose_are_rendered(monkeypatch: pytest.MonkeyPatch, use_ovstage: bool) -> None:
    """Kinematic OVPhysX transforms and root scale must reach OVRTX."""
    assert NewtonManager is not None
    assert OVRTXRendererCfg is not None
    assert OvPhysxCfg is not None
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", str(int(use_ovstage)))

    # On the OVStage path physics and the renderer draw from one stage, so the consumer count after
    # each acquisition tells whether the renderer joined the stage physics holds or opened a second.
    consumer_counts = _record_ovstage_acquisitions(monkeypatch) if use_ovstage else None

    sim_cfg = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, 0.0), physics=OvPhysxCfg())
    run_rigid_object_scale_and_pose_rendering_contract(
        RigidObjectRenderingBackend(
            name=f"ovrtx (OVPhysX, {'ovstage' if use_ovstage else 'legacy'})",
            simulation_context_factory=lambda: build_simulation_context(sim_cfg=sim_cfg),
            renderer_cfg=OVRTXRendererCfg(),
            cleanup=NewtonManager.clear,
        )
    )

    if consumer_counts is not None:
        assert consumer_counts == [1, 2], (
            "Expected physics and the renderer to share one OVStage, acquired in turn, got consumer counts"
            f" {consumer_counts}."
        )
