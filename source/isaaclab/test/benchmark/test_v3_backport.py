# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static acceptance gate for the v3-schema benchmark backport (SPEC §7.1).

Kit-free: exercises only the pure-Python schema/serialize layer. Verifies the
schema-bundle round-trips losslessly through ``serialize`` and that the field
paths / units the omniperf runner's ``_schema_bundle_to_legacy_result`` reads
(SPEC §3.5) are present. Run headless via::

    PYTHONPATH=source/isaaclab uv run --no-project --with toml python \
        source/isaaclab/test/benchmark/test_v3_backport.py

or under pytest if available.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Make ``scripts/benchmarks/_compat.py`` importable (repo-root-relative).
# This file lives at ``source/isaaclab/test/benchmark/`` -> repo root is 4 up.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "benchmarks"))

from isaaclab.test.benchmark.schema import (
    SCHEMA_VERSION,
    CProfileFunction,
    GpuDeviceInfo,
    Hardware,
    Learning,
    LearningCurve,
    MeanStd,
    Resources,
    RunConfig,
    RunIdentity,
    Runtime,
    RuntimeBundle,
    StartupBundle,
    StartupConfig,
    StartupPhase,
    StartupTime,
    TrainingBundle,
    Versions,
)
from isaaclab.test.benchmark.serialize import _to_plain, write_bundle_file


def _versions() -> Versions:
    return Versions(
        isaaclab="2.3.2",
        isaacsim="5.1.0",
        kit="107.3.3",
        newton=None,
        warp=None,
        mjwarp=None,
        torch="2.7.0",
        rsl_rl="2.3.1",
        rl_games=None,
        skrl=None,
        sb3=None,
        git_commit="dee71b5",
        git_branch="perf/benchmark-v3-backport-2.3.2",
        git_dirty=False,
    )


def _hardware() -> Hardware:
    return Hardware(
        hostname="test-host",
        gpu_devices=[GpuDeviceInfo(name="NVIDIA RTX", mem_gb=48.0, compute_cap="8.9")],
        cpu_name="Test CPU",
        cpu_count=32,
        ram_gb=128.0,
    )


def _runtime() -> Runtime:
    return Runtime(
        startup_time_s=StartupTime(
            app_launch=1.5,
            env_creation=2.0,
            first_step=0.5,
            python_imports=3.0,
            task_config=0.25,
        ),
        iterations_completed=100,
        total_wall_time_s=12.5,
        steps_per_iteration=256,
        iteration_time_s=MeanStd(mean=0.125, std=0.01, peak=0.2),
        collection_fps=MeanStd(mean=2048.0, std=64.0, peak=2200.0),
        total_fps=MeanStd(mean=2048.0, std=64.0, peak=2200.0),
        iterations_per_s=MeanStd(mean=8.0, std=0.5, peak=9.0),
    )


def _resources() -> Resources:
    return Resources(
        gpu_util_pct=MeanStd(mean=85.0, std=5.0, peak=None),
        gpu_mem_gb=MeanStd(mean=4.0, std=0.1, peak=4.2),
        cpu_util_pct=MeanStd(mean=40.0, std=3.0, peak=None),
        ram_gb=MeanStd(mean=8.0, std=0.2, peak=8.5),
    )


def _runtime_bundle() -> RuntimeBundle:
    return RuntimeBundle(
        run=RunIdentity(
            run_id="runtime-test",
            framework=None,
            config=RunConfig(physics_backend="physx", rendering_backend="isaacsim_rtx", presets=["physx"]),
            task="Isaac-Cartpole-Direct-v0",
            seed=42,
            start_time_utc="2026-07-29T12:00:00+00:00",
            end_time_utc="2026-07-29T12:00:12+00:00",
            duration_s=12.5,
            status="completed",
            num_envs=256,
            max_iterations=None,
        ),
        versions=_versions(),
        hardware=_hardware(),
        runtime=_runtime(),
        resources=_resources(),
    )


def _training_bundle() -> TrainingBundle:
    return TrainingBundle(
        run=RunIdentity(
            run_id="training-test",
            framework="rsl_rl",
            config=RunConfig(physics_backend="physx", rendering_backend="isaacsim_rtx", presets=["physx"]),
            task="Isaac-Cartpole-Direct-v0",
            seed=42,
            start_time_utc="2026-07-29T12:00:00+00:00",
            end_time_utc="2026-07-29T12:00:30+00:00",
            duration_s=30.0,
            status="completed",
            num_envs=256,
            max_iterations=100,
        ),
        versions=_versions(),
        hardware=_hardware(),
        runtime=_runtime(),
        resources=_resources(),
        learning=Learning(
            ema_alpha=0.1,
            reward=LearningCurve(final_raw=480.0, final_ema=475.0, series_per_iter=[100.0, 300.0, 480.0]),
            ep_length=LearningCurve(final_raw=500.0, final_ema=498.0, series_per_iter=[200.0, 400.0, 500.0]),
        ),
        success_rate=0.95,
    )


def _roundtrip(bundle, name: str) -> dict:
    """Write via serialize, reload, and assert lossless JSON round-trip."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"benchmark_{name}_2026_schema.json"
        write_bundle_file(bundle, str(path))
        assert path.exists(), f"{name}: schema file not written"
        with open(path) as f:
            loaded = json.load(f)
    assert loaded == _to_plain(bundle), f"{name}: JSON round-trip is not lossless"
    return loaded


def test_schema_version_is_v1_0() -> None:
    assert SCHEMA_VERSION == "1.0", f"expected schema v1.0 (pinned c587429e), got {SCHEMA_VERSION!r}"


def test_runtime_bundle_roundtrip_and_contract() -> None:
    loaded = _roundtrip(_runtime_bundle(), "runtime")
    # SPEC §3.5 / §7.1: fields the runner adapter reads.
    assert loaded["run"]["num_envs"] == 256
    st = loaded["runtime"]["startup_time_s"]
    for phase in ("app_launch", "python_imports", "task_config", "env_creation", "first_step"):
        assert phase in st, f"runtime.startup_time_s missing {phase}"
    for fps in ("collection_fps", "total_fps", "iterations_per_s", "iteration_time_s"):
        node = loaded["runtime"][fps]
        assert set(("mean", "std", "peak")).issubset(node.keys()), f"{fps} not a MeanStd"
    assert loaded["schema_version"] == "1.0"


def test_training_bundle_roundtrip_and_contract() -> None:
    loaded = _roundtrip(_training_bundle(), "training")
    # SPEC §3.5: training adds a top-level success_rate and a learning block.
    assert loaded["success_rate"] == 0.95
    assert "learning" in loaded and "reward" in loaded["learning"]
    assert loaded["run"]["num_envs"] == 256


def _startup_bundle() -> StartupBundle:
    phase = StartupPhase(
        total_time_s=1.5,
        top_functions=[CProfileFunction(name="mod:1(foo)", own_time_s=0.5, cum_time_s=1.0, calls=3)],
    )
    return StartupBundle(
        run=RunIdentity(
            run_id="startup-test",
            framework=None,
            config=RunConfig(physics_backend="physx", rendering_backend="none", presets=["physx"]),
            task="Isaac-Cartpole-Direct-v0",
            seed=0,
            start_time_utc="2026-07-29T12:00:00+00:00",
            end_time_utc="2026-07-29T12:00:05+00:00",
            duration_s=5.0,
            status="completed",
        ),
        versions=_versions(),
        hardware=_hardware(),
        phases={
            "app_launch": phase,
            "python_imports": phase,
            "task_config": phase,
            "env_creation": phase,
            "first_step": phase,
        },
        config=StartupConfig(top_n=30, whitelist=None),
    )


def test_startup_bundle_roundtrip() -> None:
    loaded = _roundtrip(_startup_bundle(), "startup")
    assert set(("app_launch", "python_imports", "task_config", "env_creation", "first_step")).issubset(
        loaded["phases"].keys()
    )
    assert loaded["schema_version"] == "1.0"


def test_compat_physx_enforcement() -> None:
    compat = importlib.import_module("_compat")
    # Accepted: PhysX / default / opaque Hydra overrides.
    compat.enforce_physx_only(["presets=physx"])
    compat.enforce_physx_only(["physx", "default", "env.scene.num_envs=8"])
    # Rejected: any Newton / OVPhysX backend, however expressed.
    for bad in (["presets=newton"], ["physics=newton_mjwarp"], ["ovphysx"], ["presets=physx,newton_kamino"]):
        try:
            compat.enforce_physx_only(bad)
            raise AssertionError(f"expected rejection for {bad}")
        except compat.NonPhysXPresetError:
            pass


def test_compat_partition_and_run_config() -> None:
    compat = importlib.import_module("_compat")
    hydra, preset = compat.partition_tokens(["presets=physx", "env.scene.num_envs=8", "physx", "seed=1"])
    assert "env.scene.num_envs=8" in hydra and "seed=1" in hydra
    assert "presets=physx" in preset and "physx" in preset
    rc = compat.run_config(["presets=physx"], enable_cameras=True)
    assert rc.physics_backend == "physx" and rc.rendering_backend == "isaacsim_rtx"
    assert compat.run_config([], enable_cameras=False).rendering_backend == "none"


def test_compat_extract_rsl_rl_series() -> None:
    """rsl_rl tag->series mapping, incl. the 2.3.2 ``"Perf/collection time"`` spelling (SPEC §5)."""
    compat = importlib.import_module("_compat")
    # 2.3.2 rsl-rl-lib logs the space-spelled collection-time tag.
    log_data = {
        "Perf/collection time": [0.1, 0.2],
        "Perf/learning_time": [0.05, 0.05],
        "Perf/total_fps": [5000.0, 5200.0],
        "Train/mean_reward": [10.0, 20.0],
        "Train/mean_episode_length": [100.0, 120.0],
    }
    series = compat.extract_rsl_rl_series(log_data, num_envs=1000, num_steps_per_env=24)
    assert series.steps_per_iteration == 24000
    assert series.iteration_times_s == [0.15000000000000002, 0.25]
    # collection_fps = steps / collection_time
    assert series.collection_fps == [24000 / 0.1, 24000 / 0.2]
    assert series.total_fps == [5000.0, 5200.0]
    assert series.reward == [10.0, 20.0] and series.ep_length == [100.0, 120.0]
    # Underscore spelling (3.x-era rsl-rl-lib) must also be honored.
    series_u = compat.extract_rsl_rl_series(
        {"Perf/collection_time": [0.5]}, num_envs=10, num_steps_per_env=2
    )
    assert series_u.collection_fps == [20 / 0.5]


def test_compat_extract_rl_games_series() -> None:
    """rl_games FPS tags -> series; iteration time derived as steps/total_fps (SPEC §5)."""
    compat = importlib.import_module("_compat")
    log_data = {
        "performance/step_inference_rl_update_fps": [8000.0, 10000.0],
        "performance/step_inference_fps": [12000.0, 15000.0],
        "rewards/iter": [1.0, 2.0],
        "episode_lengths/iter": [50.0, 60.0],
    }
    series = compat.extract_rl_games_series(log_data, num_envs=512, horizon_length=16)
    assert series.steps_per_iteration == 8192
    assert series.total_fps == [8000.0, 10000.0]
    assert series.collection_fps == [12000.0, 15000.0]
    assert series.iteration_times_s == [8192 / 8000.0, 8192 / 10000.0]
    assert series.reward == [1.0, 2.0] and series.ep_length == [50.0, 60.0]


def test_compat_success_tail_mean() -> None:
    """Trailing-window success mean; ``None`` when no success tag present (SPEC §5)."""
    compat = importlib.import_module("_compat")
    assert compat.success_tail_mean({}) is None
    assert compat.success_tail_mean({"Metrics/success_rate": [0.0, 0.5, 1.0]}, window=2) == 0.75
    # Episode-namespaced tag is honored when the top-level one is absent.
    assert compat.success_tail_mean({"Episode/Metrics/success_rate": [0.2, 0.4]}) == 0.3


def test_compat_extract_feeds_builders() -> None:
    """The extracted series drive the builders into a valid, round-trippable TrainingBundle."""
    compat = importlib.import_module("_compat")
    try:
        # builders -> metrics -> tensorboard; skip cleanly in a bare env that
        # lacks tensorboard (the real Isaac Lab env / farm always has it).
        builders = importlib.import_module("isaaclab.test.benchmark.builders")
    except ImportError as exc:
        print(f"SKIP test_compat_extract_feeds_builders (builders import unavailable: {exc})")
        return
    log_data = {
        "Perf/collection time": [0.1, 0.2],
        "Perf/learning_time": [0.05, 0.05],
        "Perf/total_fps": [5000.0, 5200.0],
        "Train/mean_reward": [10.0, 20.0],
        "Train/mean_episode_length": [100.0, 120.0],
        "Metrics/success_rate": [0.4, 0.6],
    }
    series = compat.extract_rsl_rl_series(log_data, num_envs=1000, num_steps_per_env=24)
    runtime = builders.build_runtime(
        startup_time_s=StartupTime(app_launch=1.0, env_creation=1.0, first_step=0.15),
        iteration_times_s=series.iteration_times_s,
        collection_fps=series.collection_fps,
        total_fps=series.total_fps,
        steps_per_iteration=series.steps_per_iteration,
    )
    assert runtime.iterations_completed == 2
    assert runtime.steps_per_iteration == 24000
    learning = builders.build_learning(
        reward_series=series.reward, ep_length_series=series.ep_length, ema_alpha=0.1
    )
    assert learning.reward.final_raw == 20.0
    bundle = builders.build_training_bundle(
        run=_training_bundle().run,
        versions=_versions(),
        hardware=_hardware(),
        runtime=runtime,
        resources=_resources(),
        learning=learning,
        success_rate=compat.success_tail_mean(log_data),
    )
    loaded = _roundtrip(bundle, "training_extracted")
    assert loaded["success_rate"] == 0.5
    assert loaded["runtime"]["iterations_completed"] == 2


def _run_all() -> int:
    tests = [
        test_schema_version_is_v1_0,
        test_runtime_bundle_roundtrip_and_contract,
        test_training_bundle_roundtrip_and_contract,
        test_startup_bundle_roundtrip,
        test_compat_physx_enforcement,
        test_compat_partition_and_run_config,
        test_compat_extract_rsl_rl_series,
        test_compat_extract_rl_games_series,
        test_compat_success_tail_mean,
        test_compat_extract_feeds_builders,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
