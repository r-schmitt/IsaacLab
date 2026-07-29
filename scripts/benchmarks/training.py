# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Benchmark RL training (rsl_rl / rl_games) — Isaac Lab 2.3.2 backport.

Emits a v1.0 :class:`~isaaclab.test.benchmark.TrainingBundle` (schema formatter)
paired with the legacy KPI file (omniperf formatter), consumable unchanged by
the ``omniperf-benchmark`` runner.

Rather than porting the 3.x training-loop instrumentation, this entry point
**reuses the proven Isaac Lab 2.3.2 training path** — the same ``OnPolicyRunner``
/ rl_games ``Runner`` invocation and the same post-run TensorBoard parsing used
by ``benchmark_rsl_rl.py`` / ``benchmark_rlgames.py`` — and only redirects the
*output* into the v1.0 schema. The 3.x-vs-2.3.2 differences (RL-library API
surface, TensorBoard tag spellings, PhysX-only enforcement, ``-v0`` task ids,
launcher/CLI APIs) are handled through :mod:`scripts.benchmarks._compat`:

* App is launched first, then task + agent config are resolved through the 2.x
  ``hydra_task_config`` decorator (3.x used a pre-launch ``resolve_task_config``).
* ``RslRlVecEnvWrapper(env)`` — no 3.x ``clip_actions`` kwarg / ``DistillationRunner``.
* Per-iteration FPS / reward / episode-length / success series are extracted
  from the run's TensorBoard event file via ``_compat.extract_*_series`` (which
  tolerate the 2.3.2 ``"Perf/collection time"`` spelling) — no live step hooks.

Usage example::

    ./isaaclab.sh -p scripts/benchmarks/training.py --rl_library rsl_rl \
        --task Isaac-Cartpole-v0 --num_envs 4096 --max_iterations 10 \
        --output_path /tmp/out --benchmark_formatter schema,omniperf --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

# ``_compat`` lives alongside this script; ensure it is importable when the
# script is launched by absolute path (as the omniperf runner does).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _compat  # noqa: E402

_AGENT_ENTRY_POINT = {
    "rsl_rl": "rsl_rl_cfg_entry_point",
    "rl_games": "rl_games_cfg_entry_point",
}


def _import_module_from_path(module_name: str, module_path: Path):
    """Import a module from an explicit file path (used for ``rsl_rl/cli_args.py``)."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_parser(rl_library: str):
    """Build the training-benchmark parser for *rl_library*.

    Args:
        rl_library: Selected training library (``"rsl_rl"`` or ``"rl_games"``).

    Returns:
        ``(parser, rsl_rl_cli)`` where *rsl_rl_cli* is the loaded rsl_rl
        ``cli_args`` module (or ``None`` for rl_games).
    """
    parser = argparse.ArgumentParser(description="Benchmark RL training (Isaac Lab 2.3.2 backport).")
    parser.add_argument("--rl_library", choices=sorted(_AGENT_ENTRY_POINT), required=True)
    parser.add_argument("--task", type=str, required=True, help="Gym task id to benchmark.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Environment/agent seed (-1 to sample).")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL training iterations.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
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
        "--ema_alpha",
        type=float,
        default=0.1,
        help="EMA smoothing factor for learning curves (higher = more recent weight).",
    )
    parser.add_argument(
        "--no_series",
        action="store_true",
        default=False,
        help="Omit per-iteration series data from the bundle to reduce file size.",
    )

    rsl_rl_cli = None
    if rl_library == "rsl_rl":
        # rsl_rl adds experiment/run-name/resume/logger args consumed by update_rsl_rl_cfg.
        # cli_args.py is import-light (heavy imports are lazy), so this is Kit-free.
        rsl_rl_cli = _import_module_from_path(
            "_bench_rsl_rl_cli_args",
            Path(__file__).resolve().parents[1] / "reinforcement_learning" / "rsl_rl" / "cli_args.py",
        )
        rsl_rl_cli.add_rsl_rl_args(parser)

    _compat.add_launcher_args(parser)
    return parser, rsl_rl_cli


def run(argv: list[str]) -> None:
    """Run the training benchmark and write the selected formatter outputs.

    Args:
        argv: Command-line arguments excluding the script path.
    """
    # Pre-parse ``--rl_library`` so the parser can add library-specific args.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--rl_library", choices=sorted(_AGENT_ENTRY_POINT), required=True)
    pre_args, _ = pre.parse_known_args(argv)
    rl_library = pre_args.rl_library

    parser, rsl_rl_cli = _build_parser(rl_library)
    # Enforces PhysX-only and moves Hydra overrides onto ``sys.argv`` before Kit launches.
    args, preset_tokens = _compat.parse_benchmark_cli(parser, argv)
    # Enable cameras before launch for vision tasks / --video / explicit signals
    # (AppLauncher picks the RTX experience file from this at construction).
    _compat.resolve_enable_cameras(args, args.task)

    app_t0 = time.perf_counter_ns()
    with _compat.launch_kit(args):
        app_t1 = time.perf_counter_ns()

        imports_t0 = time.perf_counter_ns()
        import contextlib

        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import hydra_task_config

        from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, metrics
        from isaaclab.test.benchmark.schema import StartupTime

        if rl_library == "rsl_rl":
            from rsl_rl.runners import OnPolicyRunner

            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        else:
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.algo_observer import IsaacAlgoObserver
            from rl_games.torch_runner import Runner

            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        # PLACEHOLDER: Extension template (do not remove this comment)
        with contextlib.suppress(ImportError):
            import isaaclab_tasks_experimental  # noqa: F401
        imports_t1 = time.perf_counter_ns()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        task_id = _compat.resolve_task_id(args.task)
        start_utc = capture.now_utc_iso()
        cfg = _compat.run_config(preset_tokens, enable_cameras=bool(getattr(args, "enable_cameras", False)))
        formatter_types = [t.strip() for t in args.benchmark_formatter.split(",") if t.strip()] or ["omniperf"]

        def _make_benchmark(seed: int | None, num_envs: int | None, max_iterations: int | None):
            return BaseIsaacLabBenchmark(
                benchmark_name="benchmark_training",
                formatter_type=args.benchmark_formatter,
                output_path=args.output_path,
                use_recorders=True,
                frametime_recorders=any(t in ("summary", "omniperf") for t in formatter_types),
                output_prefix=f"benchmark_training_{task_id}",
                workflow_metadata={
                    "metadata": [
                        {"name": "task", "data": task_id},
                        {"name": "rl_library", "data": rl_library},
                        {"name": "seed", "data": seed},
                        {"name": "num_envs", "data": num_envs},
                        {"name": "max_iterations", "data": max_iterations},
                        {"name": "presets", "data": ",".join(cfg.presets)},
                    ]
                },
            )

        def _wrap_video(env, log_dir: str):
            if not args.video:
                return env
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos"),
                "step_trigger": lambda step: step % args.video_interval == 0,
                "video_length": args.video_length,
                "disable_logger": True,
            }
            return gym.wrappers.RecordVideo(env, **video_kwargs)

        def _emit(
            benchmark,
            *,
            series: _compat.TrainingSeries,
            log_data: dict,
            startup: StartupTime,
            seed: int,
            num_envs: int,
            max_iterations: int | None,
            video_path: str | None,
        ) -> None:
            benchmark.update_manual_recorders()

            runtime = builders.build_runtime(
                startup_time_s=startup,
                iteration_times_s=series.iteration_times_s,
                collection_fps=series.collection_fps,
                total_fps=series.total_fps,
                steps_per_iteration=series.steps_per_iteration,
            )
            learning = builders.build_learning(
                reward_series=series.reward,
                ep_length_series=series.ep_length,
                ema_alpha=args.ema_alpha,
                keep_series=not args.no_series,
            )
            success_rate = _compat.success_tail_mean(log_data)

            versions = capture.capture_versions(benchmark)
            hardware = capture.capture_hardware(benchmark)
            resources = capture.capture_resources(benchmark)

            end_utc = capture.now_utc_iso()
            stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
            run_identity = builders.build_run_identity(
                run_id=capture.synth_run_id(rl_library, cfg.physics_backend, task_id, seed, stamp),
                framework=rl_library,
                config=cfg,
                task=task_id,
                seed=seed,
                start_utc=start_utc,
                end_utc=end_utc,
                num_envs=num_envs,
                max_iterations=max_iterations,
            )
            bundle = builders.build_training_bundle(
                run=run_identity,
                versions=versions,
                hardware=hardware,
                runtime=runtime,
                resources=resources,
                learning=learning,
                success_rate=success_rate,
                video_path=video_path,
            )
            benchmark.attach_bundle(bundle)
            benchmark._finalize_impl()

        def _warn_empty(log_data: dict, reward_tag: str, log_dir: str, max_iterations: int | None) -> None:
            if not log_data or (not log_data.get(reward_tag) and (max_iterations or 0) >= 1):
                print(
                    f"[WARNING] No TensorBoard data parsed from {log_dir!r}; the emitted bundle will report"
                    " zero metrics. Check the log directory and the rsl_rl/rl_games TensorBoard tag names.",
                    file=sys.stderr,
                )

        cfg_t0 = time.perf_counter_ns()

        @hydra_task_config(task_id, _AGENT_ENTRY_POINT[rl_library])
        def _body(env_cfg, agent_cfg) -> None:
            cfg_t1 = time.perf_counter_ns()

            if rl_library == "rsl_rl":
                from datetime import datetime

                agent_cfg = rsl_rl_cli.update_rsl_rl_cfg(agent_cfg, args)
                if args.num_envs is not None:
                    env_cfg.scene.num_envs = args.num_envs
                if args.max_iterations is not None:
                    agent_cfg.max_iterations = args.max_iterations
                if args.device is not None:
                    env_cfg.sim.device = args.device
                env_cfg.seed = agent_cfg.seed

                log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
                log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                if agent_cfg.run_name:
                    log_dir += f"_{agent_cfg.run_name}"
                log_dir = os.path.join(log_root_path, log_dir)

                benchmark = _make_benchmark(agent_cfg.seed, env_cfg.scene.num_envs, agent_cfg.max_iterations)

                env_t0 = time.perf_counter_ns()
                env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
                env = _wrap_video(env, log_dir)
                env = RslRlVecEnvWrapper(env)
                env_t1 = time.perf_counter_ns()

                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
                env.seed(agent_cfg.seed)

                with BenchmarkMonitor(benchmark, interval=1.0):
                    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

                log_data = metrics.parse_tf_logs(log_dir)
                _warn_empty(log_data, "Train/mean_reward", log_dir, agent_cfg.max_iterations)
                num_envs = env.unwrapped.num_envs
                series = _compat.extract_rsl_rl_series(
                    log_data, num_envs=num_envs, num_steps_per_env=agent_cfg.num_steps_per_env
                )
                seed = agent_cfg.seed if agent_cfg.seed is not None else 0
                max_iterations = agent_cfg.max_iterations
            else:
                import math
                import random
                from datetime import datetime

                if args.num_envs is not None:
                    env_cfg.scene.num_envs = args.num_envs
                if args.device is not None:
                    env_cfg.sim.device = args.device
                    agent_cfg["params"]["config"]["device"] = args.device
                    agent_cfg["params"]["config"]["device_name"] = args.device
                if args.seed == -1:
                    args.seed = random.randint(0, 10000)
                agent_cfg["params"]["seed"] = args.seed if args.seed is not None else agent_cfg["params"]["seed"]
                if args.max_iterations is not None:
                    agent_cfg["params"]["config"]["max_epochs"] = args.max_iterations
                env_cfg.seed = agent_cfg["params"]["seed"]

                log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
                log_dir = agent_cfg["params"]["config"].get(
                    "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
                agent_cfg["params"]["config"]["train_dir"] = log_root_path
                agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
                run_log_dir = os.path.join(log_root_path, log_dir)

                rl_device = agent_cfg["params"]["config"]["device"]
                clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
                clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

                benchmark = _make_benchmark(
                    agent_cfg["params"]["seed"],
                    env_cfg.scene.num_envs,
                    agent_cfg["params"]["config"].get("max_epochs"),
                )

                env_t0 = time.perf_counter_ns()
                env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
                env = _wrap_video(env, run_log_dir)
                env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
                env_t1 = time.perf_counter_ns()

                vecenv.register(
                    "IsaacRlgWrapper",
                    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
                )
                env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
                agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

                runner = Runner(IsaacAlgoObserver())
                runner.load(agent_cfg)
                env.seed(agent_cfg["params"]["seed"])
                runner.reset()

                with BenchmarkMonitor(benchmark, interval=1.0):
                    runner.run({"train": True, "play": False, "sigma": None})

                log_data = metrics.parse_tf_logs(run_log_dir, "summaries/events*")
                max_iterations = agent_cfg["params"]["config"].get("max_epochs")
                _warn_empty(log_data, "rewards/iter", run_log_dir, max_iterations)
                num_envs = env.unwrapped.num_envs
                horizon_length = agent_cfg["params"]["config"].get("horizon_length", 16)
                series = _compat.extract_rl_games_series(log_data, num_envs=num_envs, horizon_length=horizon_length)
                seed = agent_cfg["params"]["seed"] if agent_cfg["params"]["seed"] is not None else 0
                log_dir = run_log_dir

            startup = StartupTime(
                app_launch=(app_t1 - app_t0) / 1e9,
                env_creation=(env_t1 - env_t0) / 1e9,
                first_step=(series.iteration_times_s[0] if series.iteration_times_s else 0.0),
                python_imports=(imports_t1 - imports_t0) / 1e9,
                task_config=(cfg_t1 - cfg_t0) / 1e9,
            )
            video_path = os.path.join(log_dir, "videos") if args.video else None

            _emit(
                benchmark,
                series=series,
                log_data=log_data,
                startup=startup,
                seed=seed,
                num_envs=num_envs,
                max_iterations=max_iterations,
                video_path=video_path,
            )
            env.close()

        _body()


if __name__ == "__main__":
    run(sys.argv[1:])
