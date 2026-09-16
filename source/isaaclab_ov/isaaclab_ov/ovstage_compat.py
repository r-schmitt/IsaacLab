# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version compatibility between the public OVStage 0.1 API and OVStage 0.2 and later.

OVStage computes world transforms from the prim hierarchy either on the host or on the
device. The host model costs CPU work proportional to the number of prims, which the
barrier in the OVRTX ovstage write path then waits on, so the device model is preferred
wherever it is available. ``GPU_INCREMENTAL`` leaves objects out of place on OVStage 0.1
and is corrected in 0.2, so the model is chosen from the installed version rather than
hard-coded.

Where the hierarchy is computed also decides how the OVRTX ovstage path completes its
per-frame writes, so both policies are published from here and share one version boundary.

The installed version cannot change while the process runs, so both are resolved once at
import and published as :data:`HIERARCHY_COMPUTATION_MODEL` and
:data:`DEFER_PER_FRAME_WRITE_COMPLETION`.

The published name is resolved against :class:`ovstage.HierarchyComputationModel` by the
caller, which keeps this module free of an ``ovstage`` import and therefore importable
wherever the version policy needs to be inspected or tested.

The public extras stay pinned to ``ovstage==0.1.1.355824``; a missing or unparsable
install keeps the OVStage 0.1 behavior for both policies.
"""

from __future__ import annotations

import importlib.metadata
import logging

from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

# First OVStage version whose GPU_INCREMENTAL hierarchy model places objects correctly.
_GPU_HIERARCHY_VERSION = Version("0.2")


def detect_ovstage_version() -> Version | None:
    """Return the installed ``ovstage`` version.

    Read from distribution metadata rather than importing ``ovstage`` so the version
    policy can be inspected without loading the runtime. An unparsable version is logged
    and reported as missing, which keeps the OVStage 0.1 behavior.

    Returns:
        The installed version, or ``None`` when ``ovstage`` is absent or its version
        string cannot be parsed.
    """
    try:
        raw = importlib.metadata.version("ovstage")
    except importlib.metadata.PackageNotFoundError:
        return None
    try:
        return Version(raw)
    except InvalidVersion:
        logger.warning("Could not parse ovstage version %r; assuming the OVStage 0.1 hierarchy model.", raw)
        return None


def supports_gpu_hierarchy_computation(version: Version | None) -> bool:
    """Return whether ``version`` computes the prim hierarchy correctly on the device.

    Args:
        version: OVStage version to classify, or ``None`` when OVStage is unavailable.

    Returns:
        Whether ``version`` is OVStage 0.2 or newer.
    """
    return version is not None and version >= _GPU_HIERARCHY_VERSION


def resolve_hierarchy_computation_model(version: Version | None) -> str:
    """Return the :class:`ovstage.HierarchyComputationModel` member name for ``version``.

    Args:
        version: OVStage version the model is chosen for, or ``None`` when OVStage is
            unavailable.

    Returns:
        ``"GPU_INCREMENTAL"`` on OVStage 0.2 and later, otherwise ``"CPU_INCREMENTAL"``.
    """
    if supports_gpu_hierarchy_computation(version):
        return "GPU_INCREMENTAL"
    return "CPU_INCREMENTAL"


def defers_per_frame_write_completion(version: Version | None) -> bool:
    """Return whether per-frame attribute writes should complete at the ordinal barrier.

    Completing the frame's writes together at the barrier removes a host round-trip per write, but
    it only pays off once the hierarchy is computed on the device. Under the host model, waiting on
    each write overlaps ovstage's per-write hierarchy work with the caller's preparation of the next
    write; deferring them serializes that work at the barrier with nothing left to overlap, which
    measures slower. The boundary is therefore the same as
    :func:`supports_gpu_hierarchy_computation`.

    Args:
        version: OVStage version to classify, or ``None`` when OVStage is unavailable.

    Returns:
        Whether ``version`` is OVStage 0.2 or newer.
    """
    return supports_gpu_hierarchy_computation(version)


OVSTAGE_VERSION: Version | None = detect_ovstage_version()
"""Installed OVStage version, or ``None`` when it is unavailable or unparsable."""

HIERARCHY_COMPUTATION_MODEL: str = resolve_hierarchy_computation_model(OVSTAGE_VERSION)
"""Name of the hierarchy computation model to request for the installed OVStage."""

DEFER_PER_FRAME_WRITE_COMPLETION: bool = defers_per_frame_write_completion(OVSTAGE_VERSION)
"""Whether per-frame ovstage writes complete at the ordinal barrier for the installed OVStage."""
