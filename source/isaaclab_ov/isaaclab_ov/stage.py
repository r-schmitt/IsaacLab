# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for creating ovstage stages and describing their attribute columns.

``ovstage`` is a hard dependency of ``isaaclab_ov``, so it is imported unconditionally here.
"""

from __future__ import annotations

import logging

import numpy as np
import ovstage
import warp as wp

from isaaclab_ov.ovstage_compat import HIERARCHY_COMPUTATION_MODEL
from isaaclab_ov.ovstage_ordinals import POPULATION_ORDINAL, OvStageOrdinalLanes

logger = logging.getLogger(__name__)

# DLDataType for a 4x4 double matrix (``omni:xform`` column). ovstage stores omni:xform as one
# 16-lane float64 element per prim; wp.mat44d maps to the same layout via __dlpack__.
OVSTAGE_XFORM_DTYPE = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16)

# DLDataType for a float32 3-vector (``points`` column). ovstage stores ``point3f[] points`` as one
# 3-lane float32 element per vertex.
OVSTAGE_POINT_DTYPE = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=32, lanes=3)


def create_ovstage(name: str) -> ovstage.Stage:
    """Create an ovstage stage using Isaac Lab's process-wide stage configuration.

    ovstage's hierarchy computation model drives its automatic world-transform updates. It is
    process-scoped rather than per-stage: ovstage applies it when the first process reference is
    acquired and raises if a later stage asks for a conflicting model while another stage is live.
    Every Isaac Lab stage is therefore created through this helper so the whole process agrees on
    one model.

    The model is requested explicitly rather than left implicit, so the model in force is visible
    at the call site. Which one is requested depends on the installed ovstage version; see
    :mod:`isaaclab_ov.ovstage_compat`. A selected model the installed enum does not carry falls
    back to the host model, so a version gate that runs ahead of the runtime degrades rather than
    preventing stage creation.

    Args:
        name: Instance name used for ovstage diagnostics.

    Returns:
        The created :class:`ovstage.Stage`.
    """
    hierarchy_computation_model = getattr(ovstage.HierarchyComputationModel, HIERARCHY_COMPUTATION_MODEL, None)
    if hierarchy_computation_model is None:
        logger.warning(
            "This ovstage does not expose HierarchyComputationModel.%s; falling back to CPU_INCREMENTAL.",
            HIERARCHY_COMPUTATION_MODEL,
        )
        # Left unguarded: an ovstage without the host model is broken, and should say so loudly.
        hierarchy_computation_model = ovstage.HierarchyComputationModel.CPU_INCREMENTAL
    config = ovstage.StageConfig(runtime_default_hierarchy_computation_model=hierarchy_computation_model)
    return ovstage.Stage(name, config=config)


class SharedOvStage:
    """An ovstage stage bundled with the per-stage resources its consumers need.

    Every consumer of a stage needs the same three things to read or write it: the
    :class:`ovstage.Stage`, a :class:`ovstage.PathDictionary` to intern the paths and tokens its
    queries are built from, and an ordinal to write at. Bundling them is what lets a stage be
    handed to more than one consumer: a second consumer joins the path dictionary and the ordinal
    lanes already in use instead of starting its own, so neither can write to an ordinal the other
    owns. See :mod:`isaaclab_ov.ovstage_ordinals` for why that matters.

    The initial scene is populated and sealed at :data:`~isaaclab_ov.ovstage_ordinals`'s population
    ordinal. Both lanes allocate above it, so a consumer may attach at the population ordinal and
    still see every later edit.
    """

    def __init__(self, name: str, *, population_ordinal: int = POPULATION_ORDINAL) -> None:
        """Create the stage and the resources bound to it.

        Args:
            name: Instance name used for ovstage diagnostics.
            population_ordinal: Ordinal the initial scene is populated and sealed at.
        """
        self._ordinals = OvStageOrdinalLanes(population_ordinal)
        self._stage: ovstage.Stage | None = create_ovstage(name)
        try:
            self._paths: ovstage.PathDictionary | None = ovstage.PathDictionary(self._stage)
        except Exception:
            self._stage.destroy()
            self._stage = None
            raise

    @property
    def stage(self) -> ovstage.Stage:
        """The underlying ovstage stage.

        Raises:
            RuntimeError: If the stage has been destroyed.
        """
        if self._stage is None:
            raise RuntimeError("This SharedOvStage has been destroyed.")
        return self._stage

    @property
    def paths(self) -> ovstage.PathDictionary:
        """Path dictionary for interning paths and tokens against this stage.

        Raises:
            RuntimeError: If the stage has been destroyed.
        """
        if self._paths is None:
            raise RuntimeError("This SharedOvStage has been destroyed.")
        return self._paths

    @property
    def ordinals(self) -> OvStageOrdinalLanes:
        """Ordinal lanes every consumer of this stage allocates from."""
        return self._ordinals

    @property
    def population_ordinal(self) -> int:
        """Ordinal the initial scene is populated and sealed at."""
        return self._ordinals.population_ordinal

    def populate_from_usda(self, usda: str, *, domains: ovstage.PopulationDomain) -> None:
        """Populate the initial scene from USDA text at the population ordinal.

        The population ordinal is left unsealed so the caller can keep authoring setup writes at
        it (render products, scene partitions, cloned environments) before calling :meth:`seal`.

        Args:
            usda: Serialized USD stage to ingest.
            domains: Population domains to ingest. Fixed for the life of the stage; a domain left
                out here cannot be widened later.
        """
        ovstage.population.open_usd_from_string(self.stage, usda, ordinal=self.population_ordinal, domains=domains)

    def seal(self, ordinal: int) -> None:
        """Advance the write floor so every write at ``ordinal`` becomes readable.

        Consumers read sealed data only, so an unsealed ordinal is invisible to them: OVPhysX
        parses an unsealed attach ordinal as an empty scene rather than reporting an error.

        Args:
            ordinal: Ordinal to seal. Sealing clamps to the current floor, so an ordinal at or
                below it is a no-op.
        """
        self.stage.advance_write_floor(ordinal=ordinal).wait()

    def destroy(self) -> None:
        """Destroy the path dictionary and then the stage.

        The dictionary is created against the stage, so it goes first. Idempotent, so a consumer
        may close a stage that is already closed. Consumers holding the stage must release it
        first; destroying a stage a consumer is still attached to leaves that consumer with a
        dangling handle.
        """
        if self._paths is not None:
            self._paths.destroy()
            self._paths = None
        if self._stage is not None:
            self._stage.destroy()
            self._stage = None

    def __enter__(self) -> SharedOvStage:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.destroy()


def xform_tensor_from_numpy(xforms: np.ndarray) -> ovstage.DLTensor:
    """Wrap a ``(N, 4, 4)`` float64 host array as a 16-lane DLTensor for ``omni:xform`` writes.

    Args:
        xforms: Array of shape ``(N, 4, 4)`` with dtype ``float64``.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=16``.
    """
    flat = np.ascontiguousarray(xforms, dtype=np.float64).reshape(-1)
    return ovstage.make_dltensor(flat, dtype=OVSTAGE_XFORM_DTYPE, shape=[xforms.shape[0]])


def xform_tensor_from_warp(xforms: wp.array) -> ovstage.DLTensor:
    """Describe a warp ``mat44d`` array as a 16-lane DLTensor for ``omni:xform`` writes.

    The array is consumed zero-copy through DLPack: a warp ``mat44d`` exports as ``(N, 4, 4)``
    ``lanes=1``, and ovstage folds the trailing matrix axes into the ``lanes=16`` the column
    expects. A device array therefore reaches ovstage without a host round-trip.

    The caller owns the data: the returned tensor must stay alive until the consuming write
    completes, and that write must be ordered against the kernels that produced
    :paramref:`xforms` — pass their Warp stream as ``write_attribute(cuda_stream=...)``.

    Args:
        xforms: Warp array of shape ``[N]`` and dtype :class:`warp.mat44d`.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=16``.
    """
    return ovstage.make_dltensor(xforms, dtype=OVSTAGE_XFORM_DTYPE)


def points_tensor_from_warp(points: wp.array) -> ovstage.DLTensor:
    """Describe a warp ``vec3f`` array as a 3-lane DLTensor for ``points`` writes.

    The array is consumed zero-copy through DLPack: a warp ``vec3f`` exports as ``(N, 3)``
    ``lanes=1``, and ovstage folds the trailing component axis into the ``lanes=3`` the
    ``point3f[]`` column expects. A device array therefore reaches ovstage without a host
    round-trip.

    The caller owns the data: the returned tensor must stay alive until the consuming write
    completes, and that write must be ordered against the kernels that produced
    :paramref:`points` — pass their Warp stream as ``write_attribute(cuda_stream=...)``.

    Args:
        points: Warp array of shape ``[N]`` and dtype :class:`warp.vec3f`.

    Returns:
        A :class:`ovstage.DLTensor` with shape ``[N]`` and ``lanes=3``.
    """
    return ovstage.make_dltensor(points, dtype=OVSTAGE_POINT_DTYPE)
