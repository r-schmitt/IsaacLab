# HANDOFF — v3-schema benchmark KPI export backport (Isaac Lab 2.3.2)

Branch: `perf/benchmark-v3-backport-2.3.2` (off `dee71b53899a615b8f4ecda70bb8cbba310c4a1f`, VERSION 2.3.2).
Backport source: **`c587429eb521a1aaa6d1120b2ff0e82984afbdbf`** (isaac-sim/IsaacLab, 2026-06-30,
"Finish benchmark formatter migration") — schema **v1.0**, `test/benchmark/` layout. See the design
spec `SPEC_ISAACLAB_232_V3_KPI_EXPORT.md` (§10.R for the authoritative resolutions).

This documents what is done, what the agent validated locally, and the farm-only validation and the
remaining training backport that require Isaac Sim 5.1 + an R580 host (unavailable to the agent — the
agent's gate is static-only per spec §7.1).

## What was implemented

Schema/formatter layer (verbatim from `c587429e`, under
`source/isaaclab/isaaclab/test/benchmark/`): `schema.py` (`SCHEMA_VERSION = "1.0"`), `serialize.py`,
`measurements.py`, `formatters.py`, `builders.py`, `metrics.py`, `benchmark_core.py`, `capture.py`,
`stepping.py`, `profiling.py`, recorders, etc.
- **2.x adaptation:** `__init__.py` was rewritten from the 3.x `lazy_export` (needs the external
  `lazy_loader` pkg, absent on 2.3.2) to a dependency-free PEP 562 lazy `__getattr__` driven by the
  same `__init__.pyi` symbol map. Preserves lazy import (no eager `torch`/`isaacsim`). All other
  layer files are byte-identical to `c587429e`.

Compatibility shim `scripts/benchmarks/_compat.py`:
- `add_launcher_args` → `AppLauncher.add_app_launcher_args`; `parse_benchmark_cli` (replaces 3.x
  `setup_preset_cli`); `launch_kit` ctx mgr (replaces `launch_simulation`); `run_config` (replaces
  `capture.run_config_from_presets`, which touches 3.x preset modules absent on 2.3.2).
- **PhysX-only enforcement** at parse time: accepts `presets=physx`/`physics=physx`/bare
  `physx`/`default`; rejects `newton`/`newton_mjwarp`/`newton_kamino`/`kamino`/`ovphysx` with a clear
  error (spec §4.3/§8).
- `partition_tokens`: consumes the `presets=`/`physics=` selector tokens so 2.3.2 `hydra_task_config`
  never sees an unknown `presets` key; genuine Hydra overrides pass through.
- `resolve_task_id`: adds the 2.3.2 `-v0` suffix when the bare id is not registered (spec §6).

Entry points (adapted to the 2.x launch→`hydra_task_config` flow, routed through `_compat`):
- `scripts/benchmarks/runtime.py` — emits `RuntimeBundle`; schema+omniperf pair, frametime recorder
  gated on `{summary,omniperf}`.
- `scripts/benchmarks/startup.py` — emits `StartupBundle`. **2.x ordering deviation:** upstream
  resolves task config before launching Kit; 2.3.2 resolves it via `hydra_task_config` (needs the app
  running), so Kit is launched first and `app_launch` is profiled before `python_imports`/
  `task_config`. Same five phase wall-times; ordering differs. Verify on farm.
- `scripts/benchmarks/training.py` — emits `TrainingBundle` for `--rl_library {rsl_rl,rl_games}`;
  schema+omniperf pair. **Design decision (see below):** rather than port the 3.x training-loop
  instrumentation, this reuses the *proven 2.3.2 training path* and only redirects the output into the
  v1.0 schema.

## Local static gate (PASSING — spec §7.1)

Run headless (no Kit):
```
PYTHONPATH=source/isaaclab uv run --no-project --with toml \
    python source/isaaclab/test/benchmark/test_v3_backport.py
```
10/10 pass: `SCHEMA_VERSION == "1.0"`; `RuntimeBundle`/`TrainingBundle`/`StartupBundle` lossless
round-trip through `serialize`; runtime bundle carries `run.num_envs`, `runtime.startup_time_s.*`,
`runtime.{collection_fps,total_fps,iterations_per_s,iteration_time_s}` as `{mean,std,peak}`; training
carries top-level `success_rate` + `learning`; `_compat` PhysX enforcement + token partition + PhysX
`run_config`; **training TF-tag→series extraction** (`extract_rsl_rl_series` incl. the 2.3.2
`"Perf/collection time"` spelling, `extract_rl_games_series`, `success_tail_mean`) and an end-to-end
`extract_*→builders→TrainingBundle` round-trip. (The builder-feeding test needs `tensorboard`; it
self-skips in the bare `uv` env — add `--with tensorboard --with numpy` to exercise it, or run under
the real Isaac Lab env.)

Byte-stability note (spec §10.2): the omniperf metric names the runner reads come from
`benchmark_core._measurements_from_bundle`/`_runtime_measurements` (`Mean Collection FPS`,
`Mean Total FPS`, `Mean Iteration Time` (ms), `Mean Iterations per Second`; startup ms names incl.
`Total Start Time (Launch to Train)`; `Max Rewards`/`Max Episode Lengths`) plus the GPU/CPU/mem
recorders. These are byte-identical to `c587429e`. A live omniperf-file key assertion is a farm item
(needs recorders under Kit).

## Training backport — implemented (reuses 2.3.2 training logic)

**Rationale.** The 3.x `bench_rsl_rl.py`/`bench_rl_games.py` and 2.3.2's native
`benchmark_rsl_rl.py`/`benchmark_rlgames.py` share the same shape: launch → `hydra_task_config` →
wrap env → run the library trainer (`OnPolicyRunner.learn` / rl_games `Runner.run`) → **parse the
run's TensorBoard event file** for per-iteration series. They diverge only in (a) the output layer and
(b) a few RL-library API/tag details. So `training.py` keeps 2.3.2's proven training body + tag names
and only swaps the *output* into the v1.0 builders + `BaseIsaacLabBenchmark(schema,omniperf)`. This
avoids importing any 3.x RL-loop instrumentation (`early_stop.py` / `SuccessRateTracker` step hooks /
`common.dispatch_library_entrypoint` / the `rsl_rl/`+`rl_games/` subpackages are **not** ported).

**Layout.** A single flat `scripts/benchmarks/training.py` with an internal `--rl_library` dispatch
(matching the flat `runtime.py`/`startup.py` style), plus pure extraction helpers in `_compat.py`.

**Deltas from 3.x that were adapted to 2.3.2:**
- rsl_rl: `RslRlVecEnvWrapper(env)` (no 3.x `clip_actions` kwarg), `OnPolicyRunner` only (no
  `DistillationRunner`/`handle_deprecated_rsl_rl_cfg`); CLI args + `update_rsl_rl_cfg` from
  `scripts/reinforcement_learning/rsl_rl/cli_args.py`.
- rl_games: `RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)` + `Runner(IsaacAlgoObserver())`
  + `runner.run({"train": True, ...})`; agent cfg resolved via `hydra_task_config(task,
  "rl_games_cfg_entry_point")`.
- **TF tags** (`_compat.extract_*_series`, mirror `RL_LIBRARY_DESCRIPTORS`): rsl_rl reward/ep =
  `Train/mean_reward`/`Train/mean_episode_length`; FPS from `Perf/collection_time` **or the 2.3.2
  space-spelled `Perf/collection time`** + `Perf/learning_time` + `Perf/total_fps`. rl_games reward/ep
  = `rewards/iter`/`episode_lengths/iter`; FPS = `performance/step_inference_rl_update_fps` (total) /
  `performance/step_inference_fps` (collection). Success = trailing-window mean of
  `Metrics/success_rate` (`_compat.success_tail_mean`), no live tracker.

**Farm-validate:** the TF tag names against a *real* event file (rsl-rl-lib on 2.3.2 is the main
risk — confirm `Perf/collection time` vs `_time`, and that `Metrics/success_rate` is emitted by
success-tracking tasks). If a tag is absent the bundle reports zeros and `training.py` prints a
`[WARNING] No TensorBoard data parsed ...`.

- **rl_games minibatch (spec §6):** `batch_size (num_envs × horizon_length) % minibatch_size == 0`.
  Vision cfg is `minibatch_size=19600`, `horizon_length=64`, tuned for `num_envs=1225`. `training.py`
  passes `--num_envs` through unmodified (no silent rewrite); pass a divisible `--num_envs` on the farm.

## Farm validation (deferred — spec §7.2–§7.4)

Prereqs: Isaac Sim 5.1 (Kit 107.3.3); **R580 driver** for camera tasks (R590 segfaults in
`librtx.scenedb.plugin.so`, spec §6). Possible `flatdict==4.0.1` build issue → pin `setuptools<82`,
`--no-build-isolation` (spec §6).

§7.2 runtime smoke (per workflow, one PhysX + one camera task on R580):
```
python scripts/benchmarks/runtime.py --task Isaac-Cartpole-Direct-v0 --num_envs 256 \
    --output_path /tmp/out --benchmark_formatter schema,omniperf --kit_args "..." --headless
python scripts/benchmarks/startup.py --task Isaac-Cartpole-Direct-v0 --num_envs 256 \
    --output_path /tmp/out --benchmark_formatter schema,omniperf --kit_args "..." --headless
python scripts/benchmarks/training.py --rl_library rsl_rl  --task Isaac-Cartpole-v0 --num_envs 4096 \
    --max_iterations 10 --output_path /tmp/out --benchmark_formatter schema,omniperf --headless
python scripts/benchmarks/training.py --rl_library rl_games --task <vision-task> --num_envs 1225 \
    --max_iterations 10 --output_path /tmp/out --benchmark_formatter schema,omniperf --headless
```
Pass: exactly one `benchmark_*_schema.json` + one paired `..._omniperf.json` per invocation sharing a
stem in `--output_path`; **`frametime` phase non-empty** (the open `omni.hydra` risk, spec §6 — can
only be confirmed here); training emits `success_rate` + `learning` with **non-zero** FPS/reward
series (i.e. the TF tag names matched — see the training TF-tag caveat above); rl_games does not trip
the minibatch assertion.

§7.3 E2E through the omniperf runner (`local-debug` handler, nothing uploads): confirm
`_copy_benchmark_result` finds the pair, `_schema_bundle_to_legacy_result` yields
`{runtime,startup,sim_runtime,train}` populated, aggregated `kpis_data.json` has a **WARM** phase with
a non-empty bench_key, and no missing-WARM / `kpis_*.json` failure.

§7.4 DB smoke: one job into `omni_runtime_isaac_lab_v3_sandbox`, verify a row, then one real row into
`omni-runtime-isaac-lab-v3`.

## Open provenance / data-quality items to verify on farm

- `Versions.git_commit/branch/dirty` populate only if the `VersionInfoRecorder` emits a `dev` block on
  2.3.2 (`capture.capture_versions` reads `md["dev"]`). Verify; otherwise wire git provenance.
  (`isaaclab_release` from spec §4.4 is a schema-1.2 field and does **not** exist in v1.0 `Versions` —
  intentionally omitted.)
- `frametime` phase population (`omni.hydra` / `HydraEngineStats`), which feeds the v3 `sim_runtime`
  metrics.
- `startup.py` phase-ordering deviation (above) — confirm the omniperf startup KPIs still map as the
  runner expects.
