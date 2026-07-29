# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Profile Isaac Lab startup phases — Isaac Lab 2.3.2 backport.

Each phase runs in an independent ``cProfile`` session and is emitted as a v1.0
:class:`~isaaclab.test.benchmark.StartupBundle` (schema formatter) paired with
the legacy KPI file (omniperf formatter).

Profiled phases (``app_launch``, ``python_imports``, ``task_config``,
``env_creation``, ``first_step``).

2.x ordering note
-----------------
Upstream ``c587429e`` resolves the task config *before* launching Kit
(``resolve_task_config`` then ``launch_simulation``). On 2.3.2 task-config
resolution goes through the ``hydra_task_config`` decorator, which requires the
app to be running. This backport therefore launches Kit first and profiles the
deferred imports + ``hydra_task_config`` resolution afterwards: the same five
phase wall-times are captured, but ``app_launch`` precedes ``python_imports`` /
``task_config`` rather than following them. Flagged for farm validation (§7.2).

Usage example::

    ./isaaclab.sh -p scripts/benchmarks/startup.py \
        --task Isaac-Cartpole-Direct-v0 --num_envs 256 \
        --output_path /tmp/out --benchmark_formatter schema,omniperf --headless
"""

from __future__ import annotations

import argparse
import cProfile
import contextlib
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _compat  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Isaac Lab startup phases.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id to profile.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Number of top cProfile functions per phase (default: 5 with whitelist, 30 otherwise).",
    )
    parser.add_argument(
        "--benchmark_formatter",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'. Example: 'schema,omniperf'."
        ),
    )
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--whitelist_config",
        type=str,
        default=None,
        help="Path to YAML file with per-phase fnmatch patterns. Overrides --top_n for listed phases.",
    )
    _compat.add_launcher_args(parser)
    return parser


def _load_whitelist(path: str | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    import yaml

    try:
        with open(path) as wf:
            raw = yaml.safe_load(wf)
    except OSError as e:
        print(f"[ERROR] Cannot read whitelist config '{path}': {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Invalid YAML in whitelist config '{path}': {e}")
        sys.exit(1)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        print(f"[ERROR] Whitelist config must be a YAML mapping (got {type(raw).__name__}).")
        sys.exit(1)
    valid_phases = {"app_launch", "python_imports", "task_config", "env_creation", "first_step"}
    unknown = set(raw.keys()) - valid_phases
    if unknown:
        print(f"[WARNING] Whitelist config contains unknown phase(s): {unknown}. Valid: {valid_phases}.")
    for phase_name, patterns in raw.items():
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            print(f"[ERROR] Whitelist phase '{phase_name}' must be a list of strings.")
            sys.exit(1)
    return raw


def run(argv: list[str]) -> None:
    """Profile startup and write the selected formatter outputs.

    Args:
        argv: Command-line arguments excluding the script path.
    """
    start_utc = datetime.now(timezone.utc).isoformat()
    parser = _build_parser()
    args, preset_tokens = _compat.parse_benchmark_cli(parser, argv)
    # Enable cameras before launch for vision tasks / explicit signals (the runner
    # does not inject --enable_cameras; AppLauncher reads it at construction).
    _compat.resolve_enable_cameras(args, args.task)

    whitelist = _load_whitelist(args.whitelist_config)
    top_n = args.top_n if args.top_n is not None else (5 if whitelist else 30)

    app_launch_profile = cProfile.Profile()
    app_t0 = time.perf_counter_ns()
    app_launch_profile.enable()
    with _compat.launch_kit(args):
        app_launch_profile.disable()
        app_launch_wall_ms = (time.perf_counter_ns() - app_t0) / 1e6

        # --- python_imports phase (deferred imports, profiled) ---
        imports_profile = cProfile.Profile()
        imports_t0 = time.perf_counter_ns()
        imports_profile.enable()

        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import hydra_task_config

        from isaaclab.test.benchmark import BaseIsaacLabBenchmark, builders, capture, stepping
        from isaaclab.test.benchmark.profiling import parse_cprofile_stats
        from isaaclab.test.benchmark.schema import CProfileFunction, StartupPhase

        with contextlib.suppress(ImportError):
            import isaaclab_tasks_experimental  # noqa: F401

        imports_profile.disable()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        imports_wall_ms = (time.perf_counter_ns() - imports_t0) / 1e6

        task_id = _compat.resolve_task_id(args.task)
        cfg = _compat.run_config(preset_tokens, enable_cameras=bool(getattr(args, "enable_cameras", False)))

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        source_dir = os.path.join(repo_root, "source")
        isaaclab_prefixes = (
            [os.path.join(source_dir, d) for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
            if os.path.isdir(source_dir)
            else []
        )

        # --- task_config phase (hydra_task_config resolution, profiled) ---
        task_cfg_profile = cProfile.Profile()
        task_cfg_t0 = time.perf_counter_ns()
        task_cfg_profile.enable()

        @hydra_task_config(task_id, None)
        def _resolve(env_cfg, agent_cfg):
            return env_cfg

        resolved_env_cfg = _resolve()
        task_cfg_profile.disable()
        task_config_wall_ms = (time.perf_counter_ns() - task_cfg_t0) / 1e6

        if args.num_envs is not None:
            resolved_env_cfg.scene.num_envs = args.num_envs
        if args.device is not None:
            resolved_env_cfg.sim.device = args.device
        if args.seed is not None:
            resolved_env_cfg.seed = args.seed

        env = None
        try:
            # --- env_creation phase ---
            env_creation_profile = cProfile.Profile()
            env_t0 = time.perf_counter_ns()
            env_creation_profile.enable()
            try:
                env = gym.make(task_id, cfg=resolved_env_cfg)
                env.reset()
            finally:
                env_creation_profile.disable()
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            env_creation_wall_ms = (time.perf_counter_ns() - env_t0) / 1e6

            # --- first_step phase ---
            actions = stepping.sample_random_actions(env)
            first_step_profile = cProfile.Profile()
            step_t0 = time.perf_counter_ns()
            first_step_profile.enable()
            try:
                env.step(actions)
            finally:
                first_step_profile.disable()
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            first_step_wall_ms = (time.perf_counter_ns() - step_t0) / 1e6

            phase_raw = {
                "app_launch": (app_launch_profile, app_launch_wall_ms),
                "python_imports": (imports_profile, imports_wall_ms),
                "task_config": (task_cfg_profile, task_config_wall_ms),
                "env_creation": (env_creation_profile, env_creation_wall_ms),
                "first_step": (first_step_profile, first_step_wall_ms),
            }
            phases: dict[str, StartupPhase] = {}
            for phase_name, (profile, wall_ms) in phase_raw.items():
                functions = parse_cprofile_stats(
                    profile, isaaclab_prefixes, top_n=top_n, whitelist=whitelist.get(phase_name)
                )
                phases[phase_name] = StartupPhase(
                    total_time_s=wall_ms / 1000.0,
                    top_functions=[
                        CProfileFunction(
                            name=lbl, own_time_s=tot_ms / 1000.0, cum_time_s=cum_ms / 1000.0, calls=ncalls
                        )
                        for (lbl, tot_ms, cum_ms, ncalls) in functions
                    ],
                )

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
                num_envs=None,
                max_iterations=None,
            )

            benchmark = BaseIsaacLabBenchmark(
                benchmark_name="benchmark_startup",
                formatter_type=args.benchmark_formatter,
                output_path=args.output_path,
                use_recorders=True,
                output_prefix=f"benchmark_startup_{task_id}",
                workflow_metadata={
                    "metadata": [
                        {"name": "task", "data": task_id},
                        {"name": "seed", "data": args.seed},
                        {"name": "num_envs", "data": args.num_envs},
                        {"name": "top_n", "data": top_n},
                        {"name": "presets", "data": ",".join(cfg.presets)},
                    ]
                },
            )
            benchmark.update_manual_recorders()

            versions = capture.capture_versions(benchmark)
            hardware = capture.capture_hardware(benchmark)

            bundle = builders.build_startup_bundle(
                run=run,
                versions=versions,
                hardware=hardware,
                phases=phases,
                top_n=top_n,
                whitelist=args.whitelist_config,
            )

            benchmark.attach_bundle(bundle)
            benchmark._finalize_impl()
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
