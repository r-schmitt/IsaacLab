# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Command to run:
# uv run --no-sync python scripts/benchmarks/benchmark_ovstage.py
#

"""A/B profile of the OVRTX renderer's ovstage and legacy scene-ownership paths.

The two paths are selected at renderer construction by ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` (see
:func:`isaaclab_ov.renderers.ovrtx_renderer.ovrtx_use_ovstage_enabled`). Both compute their
per-frame transforms, points and camera matrices in the same Warp kernels, so the interesting
difference is not the arithmetic but how each path hands those device buffers to the renderer:

- The legacy path calls ``binding.write(..., DataAccess.ASYNC, cuda_stream=...)`` and does not wait.
- The ovstage path calls ``Stage.write_attribute(..., cuda_stream=...).wait()`` per attribute column
  and then ``Stage.advance_write_floor(...).wait()`` before ``Renderer.step``.

``Operation.wait`` and ``advance_write_floor`` are host barriers, so time inside them is a stall
rather than work. This script runs the same task twice under ``nsys``, then reports the NVTX ranges
that separate stall from work so the two paths can be compared directly.

Requires ``nsys`` on ``PATH`` and the ``nvtx`` package in the environment being profiled; without
``nvtx`` the capture succeeds but emits no Python ranges. See ``docs/source/how-to/profile_with_nsys.rst``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import site
import statistics
import subprocess
import sys
from pathlib import Path

VARIANTS = [
    {"name": "legacy", "use_ovstage": False},
    {"name": "ovstage", "use_ovstage": True},
]
"""The two OVRTX scene-ownership paths, reported in this order so deltas read legacy -> ovstage."""

TASK_NAME = "Isaac-RenderBenchmark-Franka-Cabinet"
PRESET = "ovrtx_renderer,simple_shading_constant_diffuse"
"""Cheapest OVRTX shading preset, so the attribute-write path dominates rather than shading cost."""

FRAME_PADDING = 5
"""Warm-up frames discarded at both ends of the measured window, matching ``benchmark_renderer.py``."""

# Resolved from this file rather than the working directory, so the script runs from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_SCRIPT = SCRIPT_DIR / "runtime.py"
TRACE_JSON = SCRIPT_DIR / "nsys_trace.json"
OUTPUT_PATH = SCRIPT_DIR.parent.parent / "benchmarks" / "ovstage_ab"

FRAME_RANGE = "OVRTXRenderer.render"
"""Range used as the frame boundary. One instance per rendered frame, on both paths."""

STALL = "stall"
WORK = "work"

NVTX_RANGES = [
    {"name": "ovstage.Operation.wait", "kind": STALL, "label": "ovstage Operation.wait"},
    {"name": "ovstage.Stage.advance_write_floor", "kind": STALL, "label": "ovstage advance_write_floor"},
    {"name": "ovstage.Stage.write_attribute", "kind": WORK, "label": "ovstage write_attribute"},
    {"name": "ovstage.make_dltensor", "kind": WORK, "label": "ovstage make_dltensor"},
    {"name": "ovrtx.Renderer.write_attribute", "kind": WORK, "label": "ovrtx write_attribute"},
    {"name": "ovrtx.Renderer.write_array_attribute", "kind": WORK, "label": "ovrtx write_array_attribute"},
    {"name": "ovrtx.Renderer.step", "kind": WORK, "label": "ovrtx Renderer.step"},
    {"name": "OVRTXRenderer.update_transforms", "kind": WORK, "label": "update_transforms"},
    {"name": "OVRTXRenderer.update_geometries", "kind": WORK, "label": "update_geometries"},
    {"name": "OVRTXRenderer.update_camera", "kind": WORK, "label": "update_camera"},
    {"name": "OVRTXRenderer.render", "kind": WORK, "label": "render (frame)"},
]
"""Ranges to report. ``name`` is the dotted ``module.qualname`` nsys appends after the domain.

Every entry must already be traced in :data:`TRACE_JSON`; :func:`check_traced_ranges` enforces that
so a renamed function fails loudly instead of silently reporting zero.
"""

RENDER_SCOPE = "IsaacLab::Renderer::render"
"""Backend-agnostic timer around ``BaseRenderer.render``, enabled by ``ISAACLAB_RENDER_PROFILE``.

Used for a wall-clock frame time alongside the NVTX breakdown. Unlike the NVTX ranges it
synchronizes the device on both ends, so it covers completed rather than merely submitted work.
"""

RENDER_SCOPE_PATTERN = re.compile(rf"{re.escape(RENDER_SCOPE)} took ([\d.]+) ms")

log_stream = sys.stdout
"""Destination for progress and diagnostics. ``--json`` points it at stderr so stdout holds only JSON."""


def log(message: str = "") -> None:
    """Write one human-readable line to :data:`log_stream`."""
    print(message, file=log_stream)


def matches(range_name: str, spec_name: str) -> bool:
    """Whether a captured range name refers to the function named by a spec.

    nsys names a range with the traced function's full ``module.qualname``, which for
    :data:`NVTX_RANGES` entries owned by Isaac Lab is far longer than the label worth carrying in
    this file. Matching on a trailing dotted component therefore lets a spec abbreviate the module
    (``OVRTXRenderer.render`` for ``isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer.render``)
    while still being an exact, non-substring comparison.

    Args:
        range_name: Dotted name from the capture, with any domain prefix already stripped.
        spec_name: Dotted name or trailing components of one.

    Returns:
        ``True`` if :paramref:`range_name` is, or ends with, :paramref:`spec_name`.
    """
    return range_name == spec_name or range_name.endswith("." + spec_name)


def traced_range_names(trace_json: Path) -> set[str]:
    """Collect the dotted ``module.qualname`` of every function traced by a trace definition file.

    Mirrors how nsys names a range: the domain is prepended at capture time, so the portion this
    script matches on is the module of the owning block (or a per-function override) plus the
    function's qualified name.

    Args:
        trace_json: Path to an nsys ``--python-functions-trace`` definition file.

    Returns:
        Dotted names of every traced function.
    """
    names = set()
    for block in json.loads(trace_json.read_text()):
        for entry in block["functions"]:
            if isinstance(entry, str):
                names.add(f"{block['module']}.{entry}")
            else:
                names.add(f"{entry.get('module', block['module'])}.{entry['function']}")
    return names


def check_traced_ranges(trace_json: Path) -> None:
    """Fail early if a reported range is not actually traced, which would report as a silent zero.

    Args:
        trace_json: Path to the nsys trace definition file.

    Raises:
        RuntimeError: If any entry in :data:`NVTX_RANGES` or :data:`FRAME_RANGE` is untraced.
    """
    traced = traced_range_names(trace_json)
    wanted = {entry["name"] for entry in NVTX_RANGES} | {FRAME_RANGE}
    missing = sorted(name for name in wanted if not any(matches(candidate, name) for candidate in traced))
    if missing:
        raise RuntimeError(
            f"These ranges are not traced in {trace_json}, so they would report as zero: {missing}. Add them to"
            " the trace file or drop them from NVTX_RANGES."
        )


def parse_log(path: Path, num_frames: int) -> dict | None:
    """Summarize per-frame render times [ms] from a run's captured log.

    Args:
        path: Path to the captured run log.
        num_frames: Number of frames to measure, after skipping :data:`FRAME_PADDING` warm-up frames.

    Returns:
        Timing statistics, or ``None`` if the log holds no usable frames.
    """
    with open(path) as file:
        frames = [float(match.group(1)) for line in file if (match := RENDER_SCOPE_PATTERN.search(line))]

    if not frames:
        log(f"  No '{RENDER_SCOPE}' timings in {path}; was ISAACLAB_RENDER_PROFILE set?")
        return None

    measured = frames[FRAME_PADDING : FRAME_PADDING + num_frames]
    if not measured:
        return None
    return {
        "size": len(measured),
        "median_ms": statistics.median(measured),
        "mean_ms": statistics.mean(measured),
        "min_ms": min(measured),
        "max_ms": max(measured),
        "stdev_ms": statistics.stdev(measured) if len(measured) > 1 else 0.0,
    }


def read_nvtx_trace(report: Path) -> list[dict]:
    """Export a capture's NVTX push/pop ranges via ``nsys stats``.

    ``nsys stats`` prefixes the CSV document with its own progress lines on stdout, so the header
    row is located rather than assumed to be first.

    Args:
        report: Path to an ``.nsys-rep`` capture.

    Returns:
        One entry per range with ``name``, ``start_ns``, ``end_ns`` and ``duration_ns``.

    Raises:
        RuntimeError: If ``nsys stats`` fails or emits no CSV header.
    """
    command = [
        "nsys",
        "stats",
        "--report",
        "nvtx_pushpop_trace",
        "--format",
        "csv",
        "--force-export=true",
        str(report),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"nsys stats failed for {report}:\n{completed.stdout}\n{completed.stderr}")

    lines = completed.stdout.splitlines()
    header = next((index for index, line in enumerate(lines) if line.startswith("Start (ns),")), None)
    if header is None:
        raise RuntimeError(
            f"No NVTX ranges in {report}. Is 'nvtx' installed in the profiled environment?\n{completed.stdout}"
        )

    ranges = []
    for row in csv.DictReader(lines[header:]):
        # A range still open when profiling stopped has no end, and cannot be attributed to a frame.
        if not row["End (ns)"] or not row["Duration (ns)"]:
            continue
        # The domain is prepended at capture time ("ovstage:ovstage.Operation.wait"); the dotted
        # name after it is what NVTX_RANGES matches on.
        _, _, name = row["Name"].partition(":")
        ranges.append(
            {
                "name": name or row["Name"],
                "start_ns": int(row["Start (ns)"]),
                "end_ns": int(row["End (ns)"]),
                "duration_ns": int(row["Duration (ns)"]),
            }
        )
    return ranges


def summarize_ranges(ranges: list[dict], num_frames: int) -> dict:
    """Reduce a capture's NVTX ranges to steady-state per-frame statistics.

    Setup does far more attribute writes than a frame does (cloning, scene partitions, binding
    setup), and those calls carry the same range names as the per-frame ones. Totals over a whole
    capture would therefore be dominated by startup, so the window is restricted to
    :data:`FRAME_RANGE` instances with :data:`FRAME_PADDING` frames dropped from each end.

    Per-frame figures are derived from instance counts inside that window rather than from its
    wall-clock span, so a range that straddles the boundary cannot skew the average.

    Args:
        ranges: Ranges from :func:`read_nvtx_trace`.
        num_frames: Maximum number of frames to retain after warm-up.

    Returns:
        ``frames`` (frames in the window) and ``ranges`` keyed by dotted name, each carrying
        ``instances``, ``instances_per_frame``, ``avg_ms`` and ``ms_per_frame``.

    Raises:
        RuntimeError: If the capture holds too few frames to measure.
    """
    frames = sorted((r for r in ranges if matches(r["name"], FRAME_RANGE)), key=lambda r: r["start_ns"])
    kept = frames[FRAME_PADDING : FRAME_PADDING + num_frames]
    if len(kept) < 2:
        raise RuntimeError(
            f"Only {len(frames)} '{FRAME_RANGE}' ranges in the capture; need more than"
            f" {FRAME_PADDING} warm-up frames plus two measured frames. Raise --num_frames."
        )

    window_start, window_end = kept[0]["start_ns"], kept[-1]["end_ns"]
    frame_count = len(kept)

    summary = {}
    for entry in NVTX_RANGES:
        durations = [
            r["duration_ns"]
            for r in ranges
            if matches(r["name"], entry["name"]) and r["start_ns"] >= window_start and r["end_ns"] <= window_end
        ]
        instances = len(durations)
        total_ms = sum(durations) / 1e6
        summary[entry["name"]] = {
            "instances": instances,
            "instances_per_frame": instances / frame_count,
            "avg_ms": (total_ms / instances) if instances else 0.0,
            "ms_per_frame": total_ms / frame_count,
        }

    return {"frames": frame_count, "window_ms": (window_end - window_start) / 1e6, "ranges": summary}


def build_env(variant: dict, args: argparse.Namespace) -> dict:
    """Assemble the environment for one variant's run.

    Args:
        variant: Entry from :data:`VARIANTS`.
        args: Parsed command-line arguments.

    Returns:
        Environment overlay to apply on top of the current environment.
    """
    env = {
        # An ambient ISAAC_LAB_OVRTX_USE_OVSTAGE=1 would otherwise make both variants exercise the
        # ovstage path, silently dropping legacy coverage while still reporting two runs.
        "ISAAC_LAB_OVRTX_USE_OVSTAGE": "1" if variant["use_ovstage"] else "0",
        "ISAACLAB_RENDER_PROFILE": "1",
        "BENCHMARK_RENDER_RESOLUTION": f"{args.resolution}",
        "NEWTON_USE_CUDA_GRAPH": "0",
        "WARP_CACHE_PATH": str(OUTPUT_PATH / "warp-cache"),
        # OVRTX runtime requirements, as in benchmark_renderer.py.
        "LD_PRELOAD": os.path.join(site.getsitepackages()[0], "ovrtx/bin/plugins/omni.client.lib/libomniclient.so"),
        "CUDA_VISIBLE_DEVICES": f"{args.device_id}",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "OVRTX_rtx_post_tonemap_op": "0",
    }
    return env


def build_command(variant: dict, args: argparse.Namespace) -> list[str]:
    """Build the ``nsys profile`` command line for one variant.

    Args:
        variant: Entry from :data:`VARIANTS`.
        args: Parsed command-line arguments.

    Returns:
        Argument vector to execute.
    """
    runtime = [
        sys.executable,
        str(RUNTIME_SCRIPT),
        "--task",
        args.task,
        "--num_envs",
        f"{args.num_envs}",
        "--warmup_steps",
        "0",
        "--num_steps",
        f"{args.num_frames + FRAME_PADDING * 2}",
        "--output_path",
        str(OUTPUT_PATH),
        f"presets={args.preset}",
    ]
    if args.no_nsys:
        return runtime
    return [
        "nsys",
        "profile",
        "-t",
        args.trace,
        f"--python-functions-trace={TRACE_JSON}",
        "-o",
        str(OUTPUT_PATH / variant["name"]),
        "--force-overwrite=true",
        *runtime,
    ]


def run_variant(variant: dict, args: argparse.Namespace) -> dict | None:
    """Profile one variant and summarize its stalls, work and frame times.

    Args:
        variant: Entry from :data:`VARIANTS`.
        args: Parsed command-line arguments.

    Returns:
        The variant's measurements, or ``None`` if the run failed.
    """
    name = variant["name"]
    log_path = OUTPUT_PATH / f"{name}.log"
    command = build_command(variant, args)
    env = build_env(variant, args)

    if args.dry_run:
        log(f"  ISAAC_LAB_OVRTX_USE_OVSTAGE={env['ISAAC_LAB_OVRTX_USE_OVSTAGE']}")
        log("  " + " ".join(command))
        return None

    with open(log_path, "w") as file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ) | env,
            text=True,
        )
        for line in process.stdout:
            file.write(line)
            if args.verbose:
                log_stream.write(f"\x1b[90m{line}\x1b[0m")
                log_stream.flush()

        if process.wait() != 0:
            log(f"  Failed with exit code {process.returncode}, see {log_path} for details.")
            return None

    result = {"name": name, "log": str(log_path), "frame_time": parse_log(log_path, args.num_frames)}
    if not args.no_nsys:
        report = OUTPUT_PATH / f"{name}.nsys-rep"
        result |= summarize_ranges(read_nvtx_trace(report), args.num_frames)
        result["report"] = str(report)
    return result


def stall_total_ms(result: dict) -> float:
    """Total per-frame time [ms] a variant spends in host barriers.

    Args:
        result: One variant's measurements from :func:`run_variant`.

    Returns:
        Summed ``ms_per_frame`` of every range marked :data:`STALL`.
    """
    return sum(result["ranges"][entry["name"]]["ms_per_frame"] for entry in NVTX_RANGES if entry["kind"] == STALL)


def report_table(results: dict, args: argparse.Namespace) -> None:
    """Print the A/B comparison as a table.

    Args:
        results: Variant name to measurements, for the variants that completed.
        args: Parsed command-line arguments.
    """
    names = [variant["name"] for variant in VARIANTS if variant["name"] in results]

    log("")
    log(f"task: {args.task}   num_envs: {args.num_envs}   resolution: {args.resolution}px")
    for name in names:
        frame_time = results[name]["frame_time"]
        median = f"{frame_time['median_ms']:.2f}ms" if frame_time else "n/a"
        frames = results[name].get("frames", "n/a")
        log(f"  {name:<8} frames: {frames:<5} median {RENDER_SCOPE}: {median}")

    if args.no_nsys:
        return

    separator = "|" + "-" * 32 + "|-------|" + ("-" * 13 + "|") * len(names) + "-" * 13 + "|"
    log("")
    log("Per-frame NVTX time. 'stall' rows are host barriers: time there is waiting, not work.")
    log("")
    header = f"| {'RANGE':<30} | KIND  |" + "".join(f" {name.upper():>11} |" for name in names) + f" {'DELTA':>11} |"
    log(header)
    log(separator)

    for entry in NVTX_RANGES:
        cells = ""
        for name in names:
            stats = results[name]["ranges"][entry["name"]]
            cells += f" {stats['ms_per_frame']:>9.3f}ms |" if stats["instances"] else f" {'-':>11} |"
        if len(names) == 2:
            delta = (
                results[names[1]]["ranges"][entry["name"]]["ms_per_frame"]
                - results[names[0]]["ranges"][entry["name"]]["ms_per_frame"]
            )
            cells += f" {delta:>+9.3f}ms |"
        else:
            cells += f" {'-':>11} |"
        log(f"| {entry['label']:<30} | {entry['kind']:<5} |{cells}")

    log(separator)
    stall_cells = "".join(f" {stall_total_ms(results[name]):>9.3f}ms |" for name in names)
    if len(names) == 2:
        stall_cells += f" {stall_total_ms(results[names[1]]) - stall_total_ms(results[names[0]]):>+9.3f}ms |"
    else:
        stall_cells += f" {'-':>11} |"
    log(f"| {'TOTAL HOST STALL':<30} | {STALL:<5} |{stall_cells}")
    log(separator)

    log("")
    log("Calls per frame:")
    for entry in NVTX_RANGES:
        counts = "  ".join(
            f"{name}: {results[name]['ranges'][entry['name']]['instances_per_frame']:>6.2f}" for name in names
        )
        log(f"  {entry['label']:<30} {counts}")
    log("")
    log("nsys monkey-patches every traced function, so absolute times carry tracing overhead.")
    log("Compare the two columns against each other rather than against an untraced run.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, kept separate from module import so tests can import pure helpers."""
    parser = argparse.ArgumentParser("IsaacLab Benchmark: OVRTX ovstage vs legacy write path")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames to measure")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to render")
    parser.add_argument("--resolution", type=int, default=256, help="Render resolution")
    parser.add_argument("--task", default=TASK_NAME, help="Gym task id to profile")
    parser.add_argument("--preset", default=PRESET, help="Hydra preset tokens; must select an OVRTX renderer")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device to render on")
    parser.add_argument("--trace", default="nvtx,cuda", help="nsys trace selection")
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant["name"] for variant in VARIANTS],
        help="Run only this variant; repeatable. Defaults to both.",
    )
    parser.add_argument(
        "--no_nsys",
        action="store_true",
        help="Skip nsys and report only wall-clock frame times",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the commands without running them")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Write results to stdout as JSON instead of a table")
    return parser


def main() -> None:
    """Parse CLI arguments, profile each variant, and report the comparison."""
    global log_stream

    args = _build_arg_parser().parse_args()

    # Keep stdout free of anything but the JSON document.
    if args.json:
        log_stream = sys.stderr

    if not args.no_nsys:
        if shutil.which("nsys") is None:
            print("nsys not found on PATH; install Nsight Systems or pass --no_nsys.", file=sys.stderr)
            exit(1)
        check_traced_ranges(TRACE_JSON)

    selected = [variant for variant in VARIANTS if not args.variant or variant["name"] in args.variant]
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    results = {}
    for variant in selected:
        log(f"variant: {variant['name']} (ISAAC_LAB_OVRTX_USE_OVSTAGE={int(variant['use_ovstage'])})")
        try:
            result = run_variant(variant, args)
        except KeyboardInterrupt:
            break
        if result is not None:
            results[variant["name"]] = result
        log("")

    if args.dry_run:
        exit(0)

    if args.json:
        print(
            json.dumps(
                {
                    "task": args.task,
                    "num_envs": args.num_envs,
                    "num_frames": args.num_frames,
                    "resolution": args.resolution,
                    "preset": args.preset,
                    "variants": results,
                },
                indent=2,
            )
        )
    else:
        report_table(results, args)

    exit(0 if len(results) == len(selected) else 1)


if __name__ == "__main__":
    main()
