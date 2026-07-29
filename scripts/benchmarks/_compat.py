# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 2.x compatibility shim for the v3-schema benchmark backport.

The unified benchmark entry points (``runtime.py`` / ``startup.py`` /
``training.py``) were backported verbatim from Isaac Lab ``c587429e`` (schema
v1.0). That 3.x-era code reaches for APIs that do not exist on 2.3.2:

* ``isaaclab.app.add_launcher_args`` / ``launch_simulation`` — 2.3.2 exposes
  ``AppLauncher.add_app_launcher_args(parser)`` and ``AppLauncher(args).app``.
* ``isaaclab_tasks.utils.setup_preset_cli`` — 2.3.2 has no preset system; the
  omniperf runner passes a ``presets=`` Hydra token that must be accepted
  (PhysX) or rejected (Newton/OVPhysX) rather than resolved.
* ``isaaclab_tasks.utils.resolve_task_config`` — 2.3.2 resolves task + agent
  config through the ``isaaclab_tasks.utils.hydra.hydra_task_config`` decorator.

This module centralises those adaptations. Everything that touches Isaac Sim /
Gymnasium / Isaac Lab is imported lazily *inside* the functions, so importing
``_compat`` (for its pure PhysX-enforcement and task-id helpers) never launches
Kit and stays unit-testable in the static gate (SPEC §7.1).
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
from collections.abc import Iterator, Sequence
from typing import NamedTuple

# ---------------------------------------------------------------------------
# PhysX-only enforcement (SPEC §4.3 / §8) — pure, Kit-free.
# ---------------------------------------------------------------------------

# Tokens that select PhysX (or the 2.3.2 default, which *is* PhysX). Accepted as
# no-ops: 2.3.2 has exactly one physics backend.
_PHYSX_TOKENS = frozenset({"physx", "default", ""})

# Non-PhysX backends the schema types but 2.3.2 cannot run. Rejected up front.
_REJECTED_PHYSICS_TOKENS = frozenset({
    "newton",
    "newton_mjwarp",
    "newton_kamino",
    "kamino",
    "ovphysx",
})

# Hydra selector keys that carry a physics-backend choice.
_PHYSICS_SELECTORS = frozenset({"presets", "physics", "physics_backend"})


class NonPhysXPresetError(ValueError):
    """Raised when a non-PhysX physics backend is requested on the 2.3.2 fork."""


def _iter_physics_values(tokens: Sequence[str]) -> Iterator[str]:
    """Yield lowercased physics-backend values implied by Hydra ``tokens``.

    Recognises ``presets=<v>`` / ``physics=<v>`` (comma-separated values are
    split) and bare backend tokens such as ``newton_mjwarp``.
    """
    for token in tokens:
        selector, sep, raw_value = token.partition("=")
        selector = selector.strip().lower()
        if sep:
            if selector in _PHYSICS_SELECTORS:
                for value in raw_value.split(","):
                    value = value.strip().lower()
                    if value:
                        yield value
            # Other selectors (e.g. ``env.foo=1``) are opaque Hydra overrides.
            continue
        bare = selector
        if bare:
            yield bare


def _is_consumed_preset_token(token: str) -> bool:
    """Whether ``token`` is a physics/preset selector this shim consumes itself.

    2.3.2 has no preset system and no ``presets``/``physics`` config field, so a
    ``presets=physx`` / ``physics=physx`` selector — or a bare ``physx`` /
    ``default`` token — must not reach the ``hydra_task_config`` override parser
    (Hydra would reject the unknown key). Genuine Hydra overrides such as
    ``env.scene.num_envs=8`` are left untouched.
    """
    selector, sep, _ = token.partition("=")
    selector = selector.strip().lower()
    if sep:
        return selector in _PHYSICS_SELECTORS
    return selector in _PHYSX_TOKENS


def partition_tokens(tokens: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split leftover CLI tokens into (hydra_overrides, consumed_preset_tokens).

    Args:
        tokens: Leftover tokens from ``parse_known_args``.

    Returns:
        ``(hydra_overrides, preset_tokens)`` where ``hydra_overrides`` are safe
        to feed to ``hydra_task_config`` and ``preset_tokens`` are the
        physics/preset selectors this shim consumes (recorded for provenance).
    """
    hydra_overrides: list[str] = []
    preset_tokens: list[str] = []
    for token in tokens:
        if not token.strip():
            continue
        (preset_tokens if _is_consumed_preset_token(token) else hydra_overrides).append(token)
    return hydra_overrides, preset_tokens


def enforce_physx_only(tokens: Sequence[str]) -> None:
    """Reject any non-PhysX physics backend requested via Hydra ``tokens``.

    PhysX / default tokens and opaque Hydra overrides are accepted silently.

    Args:
        tokens: Leftover Hydra override tokens (e.g. ``["presets=physx"]``).

    Raises:
        NonPhysXPresetError: If a Newton / OVPhysX backend is requested.
    """
    for value in _iter_physics_values(tokens):
        # Only known non-PhysX backends are hard-rejected; unknown bare tokens
        # are left for Hydra to interpret (they may be ordinary config overrides).
        if value in _REJECTED_PHYSICS_TOKENS:
            raise NonPhysXPresetError(
                f"Physics backend {value!r} is not available on the Isaac Lab 2.3.2 fork "
                "(PhysX only). Remove the non-PhysX preset token; accepted values are "
                "'physx' / 'default'."
            )


# ---------------------------------------------------------------------------
# Task-id suffix handling (SPEC §6) — pure, Kit-free apart from the registry read.
# ---------------------------------------------------------------------------


def resolve_task_id(task: str) -> str:
    """Return the registered Gym id for ``task``, adding the 2.3.2 ``-v0`` suffix if needed.

    2.3.2 registers ids with a ``-v0`` suffix (e.g. ``Isaac-Cartpole-Direct-v0``).
    ``gym.make`` requires the exact registered id. If ``task`` is already
    registered it is returned unchanged; otherwise ``<task>-v0`` is tried.

    Args:
        task: Task id as passed on the CLI (with or without a version suffix).

    Returns:
        A registered task id when one can be found, else ``task`` unchanged (so
        the caller still gets Gym's native "No registered env" error).
    """
    import gymnasium as gym

    registry = gym.registry
    if task in registry:
        return task
    candidate = f"{task}-v0"
    if candidate in registry:
        return candidate
    return task


# ---------------------------------------------------------------------------
# CLI + launcher shims (Kit).
# ---------------------------------------------------------------------------


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """2.x replacement for ``isaaclab.app.add_launcher_args``.

    Appends the ``AppLauncher`` CLI arguments (``--headless``, ``--device``,
    ``--enable_cameras``, ``--distributed``, ...) to ``parser``.
    """
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)


def parse_benchmark_cli(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> tuple[argparse.Namespace, list[str]]:
    """2.x replacement for ``isaaclab_tasks.utils.setup_preset_cli``.

    Parses known args, retains the remaining Hydra override tokens in
    ``sys.argv`` (so the ``hydra_task_config`` decorator can consume them), and
    enforces the PhysX-only contract at parse time — before any Kit launch.

    Args:
        parser: Parser already populated with the script + launcher arguments.
        argv: Command-line arguments excluding the script path.

    Returns:
        Parsed namespace and the consumed physics/preset tokens (for provenance;
        the Hydra-safe overrides are placed on ``sys.argv``).

    Raises:
        NonPhysXPresetError: If a non-PhysX physics backend is requested.
    """
    args, remaining = parser.parse_known_args(list(argv))
    enforce_physx_only(remaining)
    hydra_overrides, preset_tokens = partition_tokens(remaining)
    # Only genuine Hydra overrides reach the hydra_task_config parser; the
    # consumed preset selectors (e.g. ``presets=physx``) are dropped so 2.3.2
    # Hydra does not reject an unknown key.
    sys.argv = [sys.argv[0]] + hydra_overrides
    return args, preset_tokens


@contextlib.contextmanager
def launch_kit(args: argparse.Namespace):
    """2.x replacement for ``isaaclab.app.launch_simulation``.

    Launches the Omniverse app via ``AppLauncher`` and guarantees it is closed
    on exit. Yields the ``AppLauncher`` instance (callers use ``.local_rank`` /
    ``.global_rank`` for distributed runs).

    Args:
        args: Parsed namespace carrying the ``AppLauncher`` arguments.
    """
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    try:
        yield app_launcher
    finally:
        app_launcher.app.close()


def run_config(tokens: Sequence[str], *, enable_cameras: bool):
    """Build a PhysX ``RunConfig`` without touching the 3.x preset layer.

    2.3.2 has a single physics backend (PhysX) and, for camera tasks, the
    IsaacSim RTX renderer. ``capture.run_config_from_presets`` inspects 3.x
    preset modules that do not exist here, so the entry points call this instead.

    Args:
        tokens: Leftover Hydra tokens (recorded verbatim in ``RunConfig.presets``).
        enable_cameras: Whether the run enabled RTX cameras.

    Returns:
        A :class:`~isaaclab.test.benchmark.schema.RunConfig` pinned to PhysX.
    """
    from isaaclab.test.benchmark.schema import RunConfig

    enforce_physx_only(tokens)
    return RunConfig(
        physics_backend="physx",
        rendering_backend="isaacsim_rtx" if enable_cameras else "none",
        presets=[t for t in tokens if t.strip()],
    )


# ---------------------------------------------------------------------------
# Training TensorBoard extraction (SPEC §5) — pure, Kit-free.
#
# ``training.py`` reuses the *proven* 2.3.2 training loop (``OnPolicyRunner`` /
# rl_games ``Runner``) and, exactly like the native 2.3.2 benchmark scripts,
# recovers per-iteration KPIs from the run's TensorBoard event file after
# training. These helpers map the 2.3.2 scalar tags onto the v1.0 builder
# inputs. They take a plain ``{tag: [values]}`` dict (from
# ``isaaclab.test.benchmark.metrics.parse_tf_logs``) and return plain Python —
# no torch / Gym / Kit — so the tag→series mapping is unit-tested in the static
# acceptance gate against synthetic log data.
# ---------------------------------------------------------------------------

# Success-rate scalar tags, mirrored from
# ``isaaclab.test.benchmark.metrics.SUCCESS_RATE_LOG_TAGS`` so this shim stays
# import-light (no tensorboard) and remains unit-testable in the static gate.
SUCCESS_RATE_LOG_TAGS = ("Metrics/success_rate", "Episode/Metrics/success_rate")

# Trailing-window size for the post-hoc success tail mean; matches the 3.x
# ``early_stop.DEFAULT_SUCCESS_WINDOW`` the backport source used.
DEFAULT_SUCCESS_WINDOW = 20

# rsl-rl-lib changed the collection-time scalar tag spelling between the version
# shipped with 2.3.2 (``"Perf/collection time"``, space) and the 3.x-era code
# the suite was backported from (``"Perf/collection_time"``, underscore). Accept
# whichever the installed rsl-rl-lib actually logged.
_RSL_RL_COLLECTION_TIME_TAGS = ("Perf/collection_time", "Perf/collection time")
_RSL_RL_LEARNING_TIME_TAG = "Perf/learning_time"
_RSL_RL_TOTAL_FPS_TAG = "Perf/total_fps"
_RSL_RL_REWARD_TAG = "Train/mean_reward"
_RSL_RL_EP_LENGTH_TAG = "Train/mean_episode_length"

# rl_games logs FPS directly (frames/s) and its reward / episode-length curves
# under these tags; iteration time is derived as steps / total-fps.
_RL_GAMES_TOTAL_FPS_TAG = "performance/step_inference_rl_update_fps"
_RL_GAMES_COLLECTION_FPS_TAG = "performance/step_inference_fps"
_RL_GAMES_REWARD_TAG = "rewards/iter"
_RL_GAMES_EP_LENGTH_TAG = "episode_lengths/iter"


class TrainingSeries(NamedTuple):
    """Per-iteration training series extracted from TensorBoard, ready for :mod:`builders`.

    Attributes:
        iteration_times_s: Per-iteration wall-clock time [s].
        collection_fps: Per-iteration environment-stepping throughput [frames/s].
        total_fps: Per-iteration end-to-end throughput [frames/s].
        steps_per_iteration: Environment steps collected per iteration.
        reward: Per-iteration mean reward.
        ep_length: Per-iteration mean episode length.
    """

    iteration_times_s: list[float]
    collection_fps: list[float]
    total_fps: list[float]
    steps_per_iteration: int
    reward: list[float]
    ep_length: list[float]


def _first_present(log_data: dict[str, list[float]], tags: Sequence[str]) -> list[float]:
    """Return the value list for the first non-empty tag in *tags*, else ``[]``."""
    for tag in tags:
        values = log_data.get(tag)
        if values:
            return list(values)
    return []


def extract_rsl_rl_series(
    log_data: dict[str, list[float]], *, num_envs: int, num_steps_per_env: int
) -> TrainingSeries:
    """Map parsed rsl_rl TensorBoard scalars to a :class:`TrainingSeries`.

    rsl_rl reports collection and learning durations separately (seconds), so
    iteration time is their per-iteration sum and collection FPS is
    ``steps / collection_time`` — matching the native 2.3.2 benchmark script.

    Args:
        log_data: ``{tag: [values]}`` mapping from ``metrics.parse_tf_logs``.
        num_envs: Number of parallel environments.
        num_steps_per_env: rsl_rl rollout length per iteration.

    Returns:
        The extracted :class:`TrainingSeries`.
    """
    coll = _first_present(log_data, _RSL_RL_COLLECTION_TIME_TAGS)
    learn = list(log_data.get(_RSL_RL_LEARNING_TIME_TAG, []))
    iteration_times_s = [c + lrn for c, lrn in zip(coll, learn)]
    steps = int(num_envs) * int(num_steps_per_env)
    collection_fps = [steps / c for c in coll if c > 0]
    total_fps = list(log_data.get(_RSL_RL_TOTAL_FPS_TAG, []))
    return TrainingSeries(
        iteration_times_s=iteration_times_s,
        collection_fps=collection_fps,
        total_fps=total_fps,
        steps_per_iteration=steps,
        reward=list(log_data.get(_RSL_RL_REWARD_TAG, [])),
        ep_length=list(log_data.get(_RSL_RL_EP_LENGTH_TAG, [])),
    )


def extract_rl_games_series(
    log_data: dict[str, list[float]], *, num_envs: int, horizon_length: int
) -> TrainingSeries:
    """Map parsed rl_games TensorBoard scalars to a :class:`TrainingSeries`.

    rl_games logs FPS directly; iteration time is derived as
    ``steps_per_iteration / total_fps`` — matching the backported adapter.

    Args:
        log_data: ``{tag: [values]}`` mapping from ``metrics.parse_tf_logs``.
        num_envs: Number of parallel environments.
        horizon_length: rl_games rollout ``horizon_length`` per iteration.

    Returns:
        The extracted :class:`TrainingSeries`.
    """
    steps = int(num_envs) * int(horizon_length)
    total_fps = list(log_data.get(_RL_GAMES_TOTAL_FPS_TAG, []))
    collection_fps = list(log_data.get(_RL_GAMES_COLLECTION_FPS_TAG, []))
    iteration_times_s = [steps / f for f in total_fps if f > 0]
    return TrainingSeries(
        iteration_times_s=iteration_times_s,
        collection_fps=collection_fps,
        total_fps=total_fps,
        steps_per_iteration=steps,
        reward=list(log_data.get(_RL_GAMES_REWARD_TAG, [])),
        ep_length=list(log_data.get(_RL_GAMES_EP_LENGTH_TAG, [])),
    )


def success_tail_mean(log_data: dict[str, list[float]], window: int = DEFAULT_SUCCESS_WINDOW) -> float | None:
    """Post-hoc success rate: trailing-window mean of the success series in *log_data*.

    Reads the first present tag from :data:`SUCCESS_RATE_LOG_TAGS` (the same
    tags ``metrics.get_success_rate_log`` uses) and averages its trailing
    *window* values. Returns ``None`` when the task logged no success metric.

    Args:
        log_data: ``{tag: [values]}`` mapping from ``metrics.parse_tf_logs``.
        window: Number of trailing iterations to average.

    Returns:
        Rounded tail mean success rate ``[0..1]``, or ``None``.
    """
    series: list[float] | None = None
    for tag in SUCCESS_RATE_LOG_TAGS:
        if log_data.get(tag):
            series = log_data[tag]
            break
    if not series:
        return None
    tail = series[-window:] if len(series) >= window else series
    return round(statistics.mean(tail), 4)
