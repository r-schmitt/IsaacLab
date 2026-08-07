# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Nsight Systems (nsys) capture-range helpers.

These helpers let a targeted nsys profile capture a specific, named code region
(for example a startup phase such as scene creation or simulation start) instead
of the whole run. Two nsys capture-range mechanisms are supported and can be
combined:

* **NVTX range** -- every named region is emitted as an NVTX range, so
  ``nsys profile --capture-range=nvtx --nvtx-capture=<name>`` starts and stops
  the capture around it. Named regions also show up as swim-lanes in a
  full-capture profile, which otherwise leaves scene construction and
  simulation start as dark, un-annotated time.
* **cudaProfilerApi** -- set the :data:`NSYS_CAPTURE_ENV_VAR` environment
  variable to a comma-separated list of region names. Entering a listed region
  calls ``cudaProfilerStart`` and exiting calls ``cudaProfilerStop``, so
  ``nsys profile --capture-range=cudaProfilerApi`` records exactly that region.

Both mechanisms are no-ops when ``torch`` or a CUDA device is unavailable, so
importing and calling these helpers is safe in CPU-only and unit-test contexts.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

NSYS_CAPTURE_ENV_VAR = "ISAACLAB_NSYS_CAPTURE"
"""Name of the environment variable that selects region(s) for a ``cudaProfilerApi`` capture window.

Set it to a comma-separated list of region names (e.g. ``"scene_creation,simulation_start"``)
and launch nsys with ``--capture-range=cudaProfilerApi`` to record only those regions.
"""


def nsys_capture_regions() -> set[str]:
    """Return the region names selected for a ``cudaProfilerApi`` capture window.

    Returns:
        The set of region names parsed from :data:`NSYS_CAPTURE_ENV_VAR`; empty
        when the variable is unset or blank.
    """
    raw = os.environ.get(NSYS_CAPTURE_ENV_VAR, "")
    return {name.strip() for name in raw.split(",") if name.strip()}


def _torch_with_cuda():
    """Return the ``torch`` module when a CUDA device is available, else ``None``."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch


@dataclass
class _RegionToken:
    """Opaque handle returned by :func:`nsys_region_enter` and consumed by :func:`nsys_region_exit`."""

    capture: bool
    """Whether entering the region opened a ``cudaProfilerApi`` capture window that must be closed."""


def nsys_region_enter(name: str) -> _RegionToken | None:
    """Open an nsys region: push an NVTX range and optionally start a capture window.

    Args:
        name: Region label. Used as the NVTX range name and matched against
            :func:`nsys_capture_regions` to decide whether to start a
            ``cudaProfilerApi`` capture window.

    Returns:
        A token to pass to :func:`nsys_region_exit`, or ``None`` when CUDA is
        unavailable (in which case :func:`nsys_region_exit` is a no-op).
    """
    torch = _torch_with_cuda()
    if torch is None:
        return None
    capture = name in nsys_capture_regions()
    torch.cuda.nvtx.range_push(name)
    if capture:
        # Flush pending GPU work so the capture window opens on a clean boundary.
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
    return _RegionToken(capture=capture)


def nsys_region_exit(token: _RegionToken | None) -> None:
    """Close an nsys region opened by :func:`nsys_region_enter`.

    Args:
        token: The token returned by :func:`nsys_region_enter`; ``None`` is ignored.
    """
    if token is None:
        return
    torch = _torch_with_cuda()
    if torch is None:
        return
    if token.capture:
        # Flush the region's GPU work so it lands inside the capture window.
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    torch.cuda.nvtx.range_pop()


@contextmanager
def nsys_capture_range(name: str) -> Iterator[None]:
    """Context manager that brackets a code region for nsys.

    Pushes an NVTX range for the duration of the ``with`` block and, when *name*
    is listed in :data:`NSYS_CAPTURE_ENV_VAR`, wraps it in a ``cudaProfilerApi``
    capture window.

    Args:
        name: Region label (see :func:`nsys_region_enter`).
    """
    token = nsys_region_enter(name)
    try:
        yield
    finally:
        nsys_region_exit(token)
