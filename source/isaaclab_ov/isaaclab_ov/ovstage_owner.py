# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The one OVStage a simulation's consumers share, and its lifetime.

:class:`~isaaclab_ov.stage.SharedOvStage` bundles what a stage's consumers need; this module
decides that there is exactly *one* of them per simulation and keeps it alive for as long as
someone is attached. Before this, OVPhysX and OVRTX each built their own, which cost two
populations of the same scene and two resident columnar copies of it.

Consumers acquire and release rather than construct and destroy, because neither of them owns the
stage's lifetime. OVPhysX is not always present -- OVRTX also runs on Newton -- and OVRTX does not
exist in a headless run, so whichever consumer arrives first populates the stage and whichever
leaves last destroys it.

The population uses :attr:`ovstage.PopulationDomain.ALL` no matter who arrives first. Domains are
fixed when a stage is populated and cannot be widened afterwards, so a stage populated for
rendering alone could never accept physics.

What this does *not* do is unify cloning. A single pre-attach ``stage.clone`` was the original
plan, and it does realize the cloned bodies, but OVPhysX's tensor bindings cannot address them:
``create_tensor_binding`` reports one instance where it should report one per environment, by glob
and by explicit path list alike. Isaac Lab sizes every view from that count, so the environments
have to be replicated through ``physx.clone()`` after attach. The render consumer keeps its own
``stage.clone`` and performs it at an output ordinal, which OVPhysX never drains.
"""

from __future__ import annotations

import logging

from isaaclab_ov.stage import SharedOvStage

logger = logging.getLogger(__name__)


class OvStageOwner:
    """Owns the single OVStage of a simulation on behalf of its consumers.

    Not thread-safe; consumers acquire and release from the simulation's setup and teardown paths
    on one thread.
    """

    def __init__(self) -> None:
        self._shared: SharedOvStage | None = None
        self._consumers = 0
        # The serialization the live stage was populated from, kept to notice that a consumer is
        # asking for a different scene than the one currently up.
        self._usda: str | None = None

    @property
    def is_populated(self) -> bool:
        """Whether a populated stage is currently held."""
        return self._shared is not None

    @property
    def consumer_count(self) -> int:
        """How many consumers currently hold the stage."""
        return self._consumers

    def acquire(self, usda: str) -> SharedOvStage:
        """Return the shared stage, populating it on first use, and register one more consumer.

        The population ordinal is sealed before returning, so the caller may attach immediately.
        Callers that need to author setup writes of their own do so at an ordinal from
        :attr:`~isaaclab_ov.stage.SharedOvStage.ordinals`, because the population ordinal is
        already closed by the time a second consumer arrives.

        Args:
            usda: Serialization to populate from, normally
                :meth:`~isaaclab_ov.stage_usda.SharedStageUsda.resolve`'s result. Compared by
                value against the live stage's, so a consumer that re-resolves an unchanged scene
                joins the existing stage instead of being turned away over string identity.

        Returns:
            The shared stage. Every consumer receives the same instance.

        Raises:
            RuntimeError: If ``usda`` describes a different scene than the live stage was
                populated from while another consumer still holds it. Repopulating would pull the
                scene out from under an attached consumer. This is also what a full-stage physics
                load hits if a render consumer is already attached, since that load builds its own
                serialization.
        """
        if self._shared is not None and usda != self._usda:
            if self._consumers > 0:
                raise RuntimeError(
                    f"A serialization describing a different scene was handed to the shared OVStage while"
                    f" {self._consumers} consumer(s) still hold the one populated from the previous scene. Release"
                    " it first."
                )
            logger.info("The serialized scene changed; rebuilding the shared OVStage")
            self._destroy()

        if self._shared is None:
            self._populate(usda)

        self._consumers += 1
        logger.info("Acquired the shared OVStage (%d consumer(s))", self._consumers)
        return self._shared

    def release(self) -> None:
        """Deregister one consumer, destroying the stage once the last one has left.

        Idempotent once the stage is gone, so a consumer may release a stage that was already
        destroyed.
        """
        if self._shared is None:
            return
        self._consumers = max(0, self._consumers - 1)
        if self._consumers > 0:
            logger.info("Released the shared OVStage (%d consumer(s) remain)", self._consumers)
            return
        logger.info("Last consumer released the shared OVStage; destroying it")
        self._destroy()

    def _populate(self, usda: str) -> None:
        """Create the stage, ingest ``usda``, and seal the population ordinal."""
        shared = SharedOvStage("isaaclab")
        try:
            # Imported here so this module stays importable without the wheel installed.
            import ovstage  # noqa: PLC0415

            shared.populate_from_usda(
                usda,
                # FIXME: Use PHYSICS once OVStage includes native-instance collider dependencies
                # in physics-only population. ALL is required regardless while this stage is
                # shared with a render consumer.
                domains=ovstage.PopulationDomain.ALL,
            )
            # Consumers read sealed data only: population completes the writes but never commits
            # the ordinal, so attaching at an unsealed ordinal parses as an empty scene.
            shared.seal(shared.population_ordinal)
        except Exception:
            shared.destroy()
            raise
        self._shared = shared
        self._usda = usda

    def _destroy(self) -> None:
        """Destroy the held stage and forget the serialization it came from."""
        if self._shared is not None:
            self._shared.destroy()
        self._shared = None
        self._usda = None
        self._consumers = 0


_OVSTAGE_OWNER = OvStageOwner()


def ovstage_owner() -> OvStageOwner:
    """Return the process-wide owner of the shared OVStage.

    Process-wide because its consumers are: OVPhysX's runtime is a process-global singleton, and
    only one simulation is live at a time.
    """
    return _OVSTAGE_OWNER
