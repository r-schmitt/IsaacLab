# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The one serialization of the host USD stage that OVStage consumers populate from.

An OVStage is populated from USDA text, so every consumer that wants the scene has to serialize
the host USD stage to get it. With OVPhysX and OVRTX each doing that independently, the scene is
flattened, exported, and parsed twice, and the two results can disagree: whatever one consumer
authors after the other has already serialized is missing from that other consumer's copy.

:class:`SharedStageUsda` makes it one serialization. Consumers author whatever they need onto the
host stage, contribute the USDA they cannot author there (OVRTX's ``/Render`` scope has no place on
the host stage), and then ask for the text. The first ask serializes; every later ask gets the same
string back, so the consumers cannot drift apart. Contributing after that point raises rather than
being silently dropped from a string already handed out.

Neither consumer can own this. OVRTX runs on Newton as well as OVPhysX, so the physics manager is
not always there to serialize, and OVRTX itself does not exist in a headless run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from isaaclab_ov.renderers.ovrtx_usd import export_stage_to_string

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.cloner import ClonePlan

logger = logging.getLogger(__name__)


class SharedStageUsda:
    """USDA text serialized once from the host USD stage and shared by every OVStage consumer.

    The serialization keeps one row per :class:`~isaaclab.cloner.ClonePlan` source and drops the
    environment roots that are not sources: both consumers replicate the remaining prototypes
    themselves, so shipping the cloned environments would make OVStage ingest every environment as
    independently authored content and cost what the clone exists to avoid. Keeping every source
    row rather than only ``env_0`` is what makes heterogeneous prototypes work.

    Not thread-safe; it is driven from the simulation's setup path on one thread.
    """

    def __init__(self) -> None:
        self._usda: str | None = None
        self._stage: Usd.Stage | None = None
        self._contributions: list[str] = []

    @property
    def is_resolved(self) -> bool:
        """Whether the host stage has been serialized and the text handed out."""
        return self._usda is not None

    def contribute(self, usda: str) -> None:
        """Add USDA text to the serialization, ahead of the first :meth:`resolve`.

        For content that belongs in an OVStage but not on the host USD stage, such as OVRTX's
        ``/Render`` scope: render products describe how a consumer draws the scene, not the scene
        itself, and authoring them onto the host stage would expose them to everything else
        reading it.

        Args:
            usda: USDA text appended after the serialized host stage. Must stand alone at the
                root level, since it is concatenated rather than composed.

        Raises:
            RuntimeError: If the serialization was already resolved. The text is already in the
                hands of a consumer, so a later contribution could not reach it.
        """
        if self._usda is not None:
            raise RuntimeError(
                "This stage serialization was already resolved, so a contribution made now would not reach the"
                " consumers holding it. Contribute before the first resolve."
            )
        self._contributions.append(usda)

    def resolve(self, stage: Usd.Stage, clone_plan: ClonePlan | None) -> str:
        """Return the serialization, producing it on the first call.

        Args:
            stage: Host USD stage to serialize.
            clone_plan: Published clone plan, whose sources decide what survives the trim. When
                ``None``, nothing is trimmed and the whole stage is serialized.

        Returns:
            USDA text of the trimmed host stage followed by every contribution.

        Raises:
            RuntimeError: If ``stage`` is not the stage already serialized. The cached text
                describes a different scene, and returning it would silently hand a consumer the
                wrong one.
        """
        if self._usda is not None:
            if stage is not self._stage:
                raise RuntimeError(
                    "This stage serialization describes a different USD stage. Invalidate it when the stage is"
                    " replaced."
                )
            return self._usda

        num_envs = int(clone_plan.clone_mask.shape[1]) if clone_plan is not None else 1
        sources = clone_plan.sources if clone_plan is not None else ()
        # keep_env_roots=False: both consumers recreate the environment roots when they clone.
        exported = export_stage_to_string(stage, num_envs, sources, keep_env_roots=False)
        self._usda = "\n\n".join([exported, *self._contributions])
        self._stage = stage
        logger.info(
            "Serialized the host USD stage for OVStage: %d source(s), %d env(s), %d contribution(s)",
            len(sources),
            num_envs,
            len(self._contributions),
        )
        return self._usda

    def invalidate(self) -> None:
        """Drop the serialization and every contribution, so the next resolve starts over.

        Call this when the host stage is replaced or the simulation is torn down. Contributions go
        too: they name paths on the stage that is going away. Idempotent.
        """
        self._usda = None
        self._stage = None
        self._contributions.clear()


_SHARED_STAGE_USDA = SharedStageUsda()


def shared_stage_usda() -> SharedStageUsda:
    """Return the process-wide stage serialization every OVStage consumer shares.

    Process-wide because its consumers are: OVPhysX's runtime is a process-global singleton, and
    only one simulation is live at a time.
    """
    return _SHARED_STAGE_USDA
