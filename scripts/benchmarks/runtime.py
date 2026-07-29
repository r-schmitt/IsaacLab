# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Benchmark environment runtime (random actions, no policy) — Isaac Lab 2.3.2 backport.

Steps an Isaac Lab environment with random actions and emits a v1.0
:class:`~isaaclab.test.benchmark.RuntimeBundle` (schema formatter) paired with
the legacy KPI file (omniperf formatter). Backported from Isaac Lab
``c587429e`` and adapted to 2.3.2 through :mod:`scripts.benchmarks._compat`:
the app is launched first, then task config is resolved through the 2.x
``hydra_task_config`` decorator. PhysX only (see ``_compat``).

Usage example::

    ./isaaclab.sh -p scripts/benchmarks/runtime.py \
        --task Isaac-Cartpole-Direct-v0 --num_envs 256 --num_frames 100 \
        --output_path /tmp/out --benchmark_formatter schema,omniperf --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ``_compat`` lives alongside this script; ensure it is importable when the
# script is launched by absolute path (as the omniperf runner does).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _compat  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark environment runtime (random actions, no policy).")
    parser.add_argument("--task", type=str, required=True, help="Gym task id to benchmark.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of environment steps to benchmark.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--benchmark_formatter",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'. Example: 'schema,omniperf'."
        ),
    )
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run with multiple GPUs or nodes."
    )
    _compat.add_launcher_args(parser)
    return parser


def run(argv: list[str]) -> None:
    """Run the runtime benchmark and write the selected formatter outputs.

    Args:
        argv: Command-line arguments excluding the script path.
    """
    parser = _build_parser()
    # Enforces PhysX-only and moves Hydra overrides onto ``sys.argv`` before Kit launches.
    args, preset_tokens = _compat.parse_benchmark_cli(parser, argv)
    # Enable cameras before launch for vision tasks / explicit signals (the runner
    # does not inject --enable_cameras; AppLauncher reads it at construction).
    _compat.resolve_enable_cameras(args, args.task)

    app_t0 = time.perf_counter_ns()
    with _compat.launch_kit(args):
        app_t1 = time.perf_counter_ns()

        import contextlib

        import gymnasium as gym

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import hydra_task_config

        from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
        from isaaclab.test.benchmark.schema import StartupTime

        # PLACEHOLDER: Extension template (do not remove this comment)
        with contextlib.suppress(ImportError):
            import isaaclab_tasks_experimental  # noqa: F401

        task_id = _compat.resolve_task_id(args.task)
        start_utc = capture.now_utc_iso()
        cfg = _compat.run_config(preset_tokens, enable_cameras=bool(getattr(args, "enable_cameras", False)))

        formatter_types = [t.strip() for t in args.benchmark_formatter.split(",") if t.strip()] or ["omniperf"]

        @hydra_task_config(task_id, None)
        def _body(env_cfg, agent_cfg) -> None:
            if args.num_envs is not None:
                env_cfg.scene.num_envs = args.num_envs
            if args.device is not None:
                env_cfg.sim.device = args.device
            if args.seed is not None:
                env_cfg.seed = args.seed

            benchmark = BaseIsaacLabBenchmark(
                benchmark_name="benchmark_runtime",
                formatter_type=args.benchmark_formatter,
                output_path=args.output_path,
                use_recorders=True,
                frametime_recorders=any(t in ("summary", "omniperf") for t in formatter_types),
                output_prefix=f"benchmark_runtime_{task_id}",
                workflow_metadata={
                    "metadata": [
                        {"name": "task", "data": task_id},
                        {"name": "num_envs", "data": args.num_envs},
                        {"name": "num_frames", "data": args.num_frames},
                        {"name": "presets", "data": ",".join(cfg.presets)},
                    ]
                },
            )

            env_t0 = time.perf_counter_ns()
            with contextlib.closing(gym.make(task_id, cfg=env_cfg)) as env:
                env_t1 = time.perf_counter_ns()

                num_envs = env.unwrapped.num_envs

                with BenchmarkMonitor(benchmark, interval=1.0):
                    step_times_s = stepping.run_runtime_loop(env, args.num_frames)

                benchmark.update_manual_recorders()

                startup = StartupTime(
                    app_launch=(app_t1 - app_t0) / 1e9,
                    env_creation=(env_t1 - env_t0) / 1e9,
                    first_step=(step_times_s[0] if step_times_s else 0.0),
                )

                # Pure runtime: one env.step() call steps all envs, so per-call
                # throughput is num_envs / step_time; total FPS equals collection FPS.
                fps = [num_envs / t for t in step_times_s if t > 0]
                runtime = builders.build_runtime(
                    startup_time_s=startup,
                    iteration_times_s=step_times_s,
                    collection_fps=fps,
                    total_fps=fps,
                    steps_per_iteration=num_envs,
                )

                versions = capture.capture_versions(benchmark)
                hardware = capture.capture_hardware(benchmark)
                resources = capture.capture_resources(benchmark)

                end_utc = capture.now_utc_iso()
                stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
                seed = args.seed if args.seed is not None else 0
                run_id = capture.synth_run_id(None, cfg.physics_backend, task_id, seed, stamp)

                run = builders.build_run_identity(
                    run_id=run_id,
                    framework=None,
                    config=cfg,
                    task=task_id,
                    seed=seed,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    num_envs=num_envs,
                )

                bundle = builders.build_runtime_bundle(
                    run=run,
                    versions=versions,
                    hardware=hardware,
                    runtime=runtime,
                    resources=resources,
                )

                benchmark.attach_bundle(bundle)
                benchmark._finalize_impl()

        _body()


if __name__ == "__main__":
    run(sys.argv[1:])
