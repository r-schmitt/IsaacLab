# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorder utilities for the IsaacLab benchmark suite.

On Isaac Lab 2.x there is no ``isaaclab.utils.module.lazy_export`` (a 3.x helper
built on the external ``lazy_loader`` package), so this backport resolves the
names declared in ``__init__.pyi`` on first attribute access via a PEP 562
module-level ``__getattr__`` — the same approach as the parent
``isaaclab.test.benchmark`` package. Keeps import lazy so pulling in a single
recorder does not eagerly import the others (e.g. the GPU/pynvml stack).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Name -> submodule mapping, kept in lockstep with ``__init__.pyi``.
_LAZY_SUBMODULE_BY_NAME: dict[str, str] = {
    "CPUInfoRecorder": "record_cpu_info",
    "GPUInfoRecorder": "record_gpu_info",
    "MemoryInfoRecorder": "record_memory_info",
    "VersionInfoRecorder": "record_version_info",
}

__all__ = list(_LAZY_SUBMODULE_BY_NAME.keys())


def __getattr__(name: str):
    """Lazily import a public recorder from its owning submodule (PEP 562)."""
    module_name = _LAZY_SUBMODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent access skips the import machinery
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    # Give static analyzers the real symbols without triggering eager imports.
    from .record_cpu_info import CPUInfoRecorder
    from .record_gpu_info import GPUInfoRecorder
    from .record_memory_info import MemoryInfoRecorder
    from .record_version_info import VersionInfoRecorder
