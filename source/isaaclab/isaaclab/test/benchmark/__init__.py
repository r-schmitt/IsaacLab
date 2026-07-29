# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmarking utilities for IsaacLab.

This package provides benchmarking utilities used across different test modules.

On Isaac Lab 2.x there is no ``isaaclab.utils.module.lazy_export`` (a 3.x helper
built on the external ``lazy_loader`` package). To keep the public surface lazy —
so ``import isaaclab.test.benchmark`` does not eagerly pull in ``torch`` or the
Isaac Sim runtime — this backport resolves the names declared in ``__init__.pyi``
on first attribute access via a PEP 562 module-level ``__getattr__``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Name -> submodule mapping, kept in lockstep with ``__init__.pyi`` (the stable
# public surface). Grouped by owning module for readability.
_LAZY_SUBMODULE_BY_NAME: dict[str, str] = {
    # benchmark_core
    "BaseIsaacLabBenchmark": "benchmark_core",
    "get_default_output_filename": "benchmark_core",
    # benchmark_monitor
    "BenchmarkMonitor": "benchmark_monitor",
    # method_benchmark
    "MethodBenchmarkDefinition": "method_benchmark",
    "MethodBenchmarkRunner": "method_benchmark",
    "MethodBenchmarkRunnerConfig": "method_benchmark",
    # measurements
    "BooleanMeasurement": "measurements",
    "DictMeasurement": "measurements",
    "DictMetadata": "measurements",
    "FloatMetadata": "measurements",
    "IntMetadata": "measurements",
    "ListMeasurement": "measurements",
    "Measurement": "measurements",
    "MetadataBase": "measurements",
    "SingleMeasurement": "measurements",
    "StatisticalMeasurement": "measurements",
    "StringMetadata": "measurements",
    "TestPhase": "measurements",
    # schema
    "SCHEMA_VERSION": "schema",
    "CProfileFunction": "schema",
    "Framework": "schema",
    "GpuDeviceInfo": "schema",
    "Hardware": "schema",
    "Learning": "schema",
    "LearningCurve": "schema",
    "MeanStd": "schema",
    "PhysicsBackend": "schema",
    "PlayBundle": "schema",
    "RenderingBackend": "schema",
    "Resources": "schema",
    "RunConfig": "schema",
    "RunIdentity": "schema",
    "RunStatus": "schema",
    "Runtime": "schema",
    "RuntimeBundle": "schema",
    "StartupBundle": "schema",
    "StartupConfig": "schema",
    "StartupPhase": "schema",
    "StartupTime": "schema",
    "TrainingBundle": "schema",
    "Versions": "schema",
    # serialize
    "write_bundle_file": "serialize",
}

__all__ = list(_LAZY_SUBMODULE_BY_NAME.keys())


def __getattr__(name: str):
    """Lazily import a public symbol from its owning submodule (PEP 562)."""
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
    from .benchmark_core import BaseIsaacLabBenchmark, get_default_output_filename
    from .benchmark_monitor import BenchmarkMonitor
    from .measurements import (
        BooleanMeasurement,
        DictMeasurement,
        DictMetadata,
        FloatMetadata,
        IntMetadata,
        ListMeasurement,
        Measurement,
        MetadataBase,
        SingleMeasurement,
        StatisticalMeasurement,
        StringMetadata,
        TestPhase,
    )
    from .method_benchmark import (
        MethodBenchmarkDefinition,
        MethodBenchmarkRunner,
        MethodBenchmarkRunnerConfig,
    )
    from .schema import (
        SCHEMA_VERSION,
        CProfileFunction,
        Framework,
        GpuDeviceInfo,
        Hardware,
        Learning,
        LearningCurve,
        MeanStd,
        PhysicsBackend,
        PlayBundle,
        RenderingBackend,
        Resources,
        RunConfig,
        RunIdentity,
        RunStatus,
        Runtime,
        RuntimeBundle,
        StartupBundle,
        StartupConfig,
        StartupPhase,
        StartupTime,
        TrainingBundle,
        Versions,
    )
    from .serialize import write_bundle_file
