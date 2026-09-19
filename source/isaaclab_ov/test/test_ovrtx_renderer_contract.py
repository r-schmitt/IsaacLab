# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX renderer output contract."""

import contextlib
import importlib.util
import sys
import types

import pytest
import torch
import warp as wp

from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind, RenderBufferSpec
from isaaclab.sim import PinholeCameraCfg

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXRendererCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_compat import RENDER_VAR_FRAME_KEYS  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        _DISABLE_LINUX_CUDA_CPU_SYNC_ENV,
        OVRTXRenderData,
        OVRTXRenderer,
        _gpu_side_render_var_sync_enabled,
        ovrtx_use_ovstage_enabled,
    )
else:
    OVRTXRenderData = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    ovrtx_renderer_module = None
    ovrtx_use_ovstage_enabled = None
    _DISABLE_LINUX_CUDA_CPU_SYNC_ENV = None
    _gpu_side_render_var_sync_enabled = None
    RENDER_VAR_FRAME_KEYS = None

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def _make_ovrtx_render_data() -> OVRTXRenderData:
    rd = OVRTXRenderData.__new__(OVRTXRenderData)
    rd.width = 16
    rd.height = 8
    rd.num_envs = 2
    rd.warp_buffers = {}
    rd.renderer_info = {}
    rd.ppisp_pipeline = None
    return rd


def _make_ovrtx_renderer_without_backend() -> OVRTXRenderer:
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg()
    return renderer


def test_ovrtx_renderer_config_enables_supported_runtime_options(monkeypatch: pytest.MonkeyPatch):
    """OVRTX 0.4.1 options are passed directly to ``RendererConfig``."""
    config_kwargs: dict[str, object] = {}

    class RecordingRendererConfig:
        def __init__(self, **kwargs):
            config_kwargs.update(kwargs)

    monkeypatch.setattr(ovrtx_renderer_module, "RendererConfig", RecordingRendererConfig)
    monkeypatch.setattr(ovrtx_renderer_module, "Renderer", lambda config: object())  # noqa: ARG005
    monkeypatch.setattr(ovrtx_renderer_module, "ovrtx_use_ovstage_enabled", lambda: False)

    renderer = OVRTXRenderer(OVRTXRendererCfg())

    assert renderer._renderer is not None
    assert config_kwargs["suppress_deprecation_warnings"] is True
    assert config_kwargs["texture_streaming_mode"] is ovrtx_renderer_module.TextureStreamingMode.SYNCHRONOUS


def test_ovrtx_supported_output_types_key_set():
    """OVRTX publishes the documented key set and per-output spec."""
    renderer = _make_ovrtx_renderer_without_backend()
    specs = renderer.supported_output_types()

    assert set(specs.keys()) == {
        RenderBufferKind.RGB,
        RenderBufferKind.RGBA,
        RenderBufferKind.RGB_HDR,
        RenderBufferKind.ALBEDO,
        RenderBufferKind.SIMPLE_SHADING_CONSTANT_DIFFUSE,
        RenderBufferKind.SIMPLE_SHADING_DIFFUSE_MDL,
        RenderBufferKind.SIMPLE_SHADING_FULL_MDL,
        RenderBufferKind.SEMANTIC_SEGMENTATION,
        RenderBufferKind.INSTANCE_SEGMENTATION,
        RenderBufferKind.DEPTH,
        RenderBufferKind.DISTANCE_TO_IMAGE_PLANE,
        RenderBufferKind.DISTANCE_TO_CAMERA,
        RenderBufferKind.NORMALS,
        RenderBufferKind.MOTION_VECTORS,
    }
    assert specs[RenderBufferKind.RGBA] == RenderBufferSpec(4, wp.uint8)
    assert specs[RenderBufferKind.RGB_HDR] == RenderBufferSpec(3, wp.float32)
    assert specs[RenderBufferKind.DEPTH] == RenderBufferSpec(1, wp.float32)
    assert specs[RenderBufferKind.MOTION_VECTORS] == RenderBufferSpec(2, wp.float32)


def test_ovrtx_set_outputs_wraps_caller_torch_zero_copy():
    """OVRTXRenderer.set_outputs publishes warp views over the caller's warp storage."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb", "rgba", "depth"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert set(render_data.warp_buffers.keys()) >= {"rgba", "depth"}
    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert render_data.warp_buffers["depth"].ptr == data.output["depth"].warp.ptr
    assert "rgb" not in render_data.warp_buffers


def test_ovrtx_set_outputs_wraps_requested_rgb_hdr_output():
    """OVRTXRenderer.set_outputs publishes a zero-copy view for requested RGB_HDR."""
    renderer = _make_ovrtx_renderer_without_backend()

    if not torch.cuda.is_available():
        pytest.skip("OVRTX zero-copy wrapping requires a CUDA device")
    device = "cuda"

    cfg = _make_camera_cfg(["rgb_hdr"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device=device,
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgb_hdr"].ptr == data.output["rgb_hdr"].warp.ptr


def test_ovrtx_set_outputs_routes_ppisp_buffers_through_warp_buffers():
    """OVRTXRenderer.set_outputs stores PPISP source/destination in warp_buffers."""
    renderer = _make_ovrtx_renderer_without_backend()

    cfg = _make_camera_cfg(["rgb"])
    data = CameraData.allocate(
        data_types=cfg.data_types,
        height=8,
        width=16,
        num_views=2,
        device="cpu",
        supported_specs=renderer.supported_output_types(),
    )
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()
    renderer.set_outputs(render_data, data.output)

    assert render_data.warp_buffers["rgba"].ptr == data.output["rgba"].warp.ptr
    assert "rgb_hdr" in render_data.warp_buffers
    assert render_data.warp_buffers["rgb_hdr"].shape == (2, 8, 16, 3)
    assert render_data.warp_buffers["rgb_hdr"].dtype is wp.float32


def test_ovrtx_process_frame_skips_ldr_rgba_when_ppisp_is_active():
    """PPISP owns RGBA output, so OVRTX LdrColor should not pre-fill it."""

    class FailingRenderVar:
        def map(self, *args, **kwargs):
            raise AssertionError("PPISP RGBA output must not read OVRTX LdrColor")

    class Frame:
        render_vars = {RENDER_VAR_FRAME_KEYS["LdrColor"]: FailingRenderVar()}

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()

    renderer._process_render_frame(render_data, Frame(), {"rgba": object()})


@pytest.mark.parametrize("stale_key", ["LdrColor", "/Render/Vars/LdrColor"])
def test_ovrtx_process_frame_reads_only_the_installed_ldr_color_key(monkeypatch: pytest.MonkeyPatch, stale_key: str):
    """Frames are keyed by source name on OVRTX 0.4 and by prim path on 0.5; only one form is read."""
    installed_key = RENDER_VAR_FRAME_KEYS["LdrColor"]

    mapped = []

    @contextlib.contextmanager
    def fake_map(self, render_var):
        mapped.append(render_var)
        yield object()

    monkeypatch.setattr(OVRTXRenderer, "_map_render_var_to_dlpack", fake_map)
    monkeypatch.setattr(OVRTXRenderer, "_extract_rgba_tiles", lambda *args, **kwargs: None)

    class Frame:
        render_vars = {stale_key: "stale", installed_key: "installed"}

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._process_render_frame(_make_ovrtx_render_data(), Frame(), {"rgba": object()})
    assert mapped == ["installed"]


def test_ovrtx_ppisp_hdr_source_is_cloned_to_output_device(monkeypatch):
    """PPISP HdrColor source is moved to the HDR output buffer device."""

    class FakeArray:
        device = "cuda:1"

    class OutputArray:
        device = "cuda:0"

    cloned = object()
    clone_calls = []

    def fake_clone(src, *, device):
        clone_calls.append((src, device))
        return cloned

    monkeypatch.setattr(wp, "clone", fake_clone)

    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    render_data.ppisp_pipeline = object()
    source = FakeArray()

    assert renderer._prepare_ppisp_hdr_source(render_data, source, {"rgb_hdr": OutputArray()}) is cloned
    assert clone_calls == [(source, "cuda:0")]


class _FakeArray:
    def __init__(self, shape):
        self.shape = shape


def test_launch_extract_all_tiles_rejects_wider_output_channels():
    """An output wider than the tiled input would read out of bounds, so it must raise before launching."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_render_data()

    with pytest.raises(ValueError, match="out of bounds"):
        renderer._launch_extract_all_tiles(render_data, _FakeArray((8, 16, 3)), _FakeArray((2, 8, 16, 4)))


def test_launch_extract_all_tiles_launches_kernel_when_channels_are_compatible(monkeypatch):
    """Equal or narrower output channel counts pass validation and reach the kernel launch."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cpu"
    render_data = _make_ovrtx_render_data()
    render_data.num_cols = 2

    launch_calls = []
    monkeypatch.setattr(wp, "launch", lambda **kwargs: launch_calls.append(kwargs))

    tiled_buffer = _FakeArray((8, 16, 4))
    output_buffer = _FakeArray((2, 8, 16, 3))
    renderer._launch_extract_all_tiles(render_data, tiled_buffer, output_buffer)

    assert len(launch_calls) == 1
    assert launch_calls[0]["inputs"][:2] == [tiled_buffer, output_buffer]


def test_ovrtx_read_output_copies_no_pixel_data():
    """OVRTXRenderer.read_output copies no pixel data; with empty renderer_info it leaves info untouched."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    camera_data = CameraData()
    camera_data.info = {}
    camera_data._output = {}

    result = renderer.read_output(render_data, camera_data)
    assert result is None
    assert render_data.warp_buffers == {}
    assert camera_data.info == {}
    assert camera_data.output == {}


def test_ovrtx_read_output_forwards_renderer_info():
    """OVRTXRenderer.read_output forwards render_data.renderer_info (e.g. semantic idToLabels) into info."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": id_to_labels}}

    camera_data = CameraData()
    camera_data.info = {"semantic_segmentation": None}
    camera_data._output = {}

    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}


def test_ovrtx_read_output_clears_stale_metadata_and_keeps_seeded_keys():
    """read_output replaces (not merges): a dropped render var resets its info entry, seeded keys persist."""
    renderer = _make_ovrtx_renderer_without_backend()
    render_data = _make_ovrtx_render_data()

    # ``camera_data.info`` is seeded with one key per output (mirrors ``camera_data.output``); both start None.
    camera_data = CameraData()
    camera_data.info = {"rgb": None, "semantic_segmentation": None}
    camera_data._output = {}

    # Frame 1: the SemanticIdMap render var is present, so its metadata lands in info.
    id_to_labels = {"2": {"class": "cartpole"}}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": id_to_labels}}
    renderer.read_output(render_data, camera_data)
    assert camera_data.info["semantic_segmentation"] == {"idToLabels": id_to_labels}

    # Frame 2: render() rebuilds renderer_info from scratch and the SemanticIdMap is gone this frame.
    render_data.renderer_info = {}
    renderer.read_output(render_data, camera_data)

    # The stale idToLabels must be cleared, and the seeded keys (rgb, semantic_segmentation) must remain.
    assert camera_data.info == {"rgb": None, "semantic_segmentation": None}


def test_ovrtx_semantic_spec_follows_colorize_flag():
    """Semantic segmentation output spec is colorized RGBA (uint8) or raw int32 IDs per the cfg flag."""
    colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    colorized.cfg = OVRTXRendererCfg(colorize_semantic_segmentation=True)
    assert colorized.supported_output_types()[RenderBufferKind.SEMANTIC_SEGMENTATION] == RenderBufferSpec(4, wp.uint8)

    non_colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    non_colorized.cfg = OVRTXRendererCfg(colorize_semantic_segmentation=False)
    assert non_colorized.supported_output_types()[RenderBufferKind.SEMANTIC_SEGMENTATION] == RenderBufferSpec(
        1, wp.int32
    )


def test_ovrtx_instance_segmentation_spec_follows_colorize_flag():
    """Instance segmentation output spec is colorized RGBA (uint8) or raw int32 IDs per the cfg flag."""
    colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    colorized.cfg = OVRTXRendererCfg(colorize_instance_segmentation=True)
    assert colorized.supported_output_types()[RenderBufferKind.INSTANCE_SEGMENTATION] == RenderBufferSpec(4, wp.uint8)

    non_colorized = OVRTXRenderer.__new__(OVRTXRenderer)
    non_colorized.cfg = OVRTXRendererCfg(colorize_instance_segmentation=False)
    assert non_colorized.supported_output_types()[RenderBufferKind.INSTANCE_SEGMENTATION] == RenderBufferSpec(
        1, wp.int32
    )


def test_ovrtx_use_ovstage_defaults_to_disabled(monkeypatch):
    """The ovstage path is off unless explicitly opted into, so existing deployments are unaffected."""
    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    assert ovrtx_use_ovstage_enabled() is False

    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "0")
    assert ovrtx_use_ovstage_enabled() is False


def test_ovrtx_use_ovstage_enabled_when_requested(monkeypatch):
    """Setting the variable to 1 selects the ovstage path."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    assert ovrtx_use_ovstage_enabled() is True


def test_ovrtx_use_ovstage_rejects_non_boolean_values(monkeypatch):
    """Values other than 0/1 are a configuration error, not a silent disable."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "true")

    with pytest.raises(ValueError, match="Expected 0 or 1"):
        ovrtx_use_ovstage_enabled()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_ovrtx_render_var_sync_is_gpu_side_off_linux(monkeypatch, platform):
    """Everywhere but Linux the mapping is ordered by a GPU-side wait on the Warp stream."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, raising=False)
    assert _gpu_side_render_var_sync_enabled() is True


def test_ovrtx_render_var_sync_waits_on_host_on_linux(monkeypatch):
    """Linux blocks the calling thread instead, which measures faster there."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, raising=False)
    assert _gpu_side_render_var_sync_enabled() is False


def test_ovrtx_render_var_sync_is_gpu_side_on_linux_when_disabled(monkeypatch):
    """Opting out of the host wait puts Linux on the same GPU-side wait as every other platform."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, "1")
    assert _gpu_side_render_var_sync_enabled() is True


def test_ovrtx_render_var_sync_keeps_host_wait_when_explicitly_enabled(monkeypatch):
    """``0`` is the default, so setting it explicitly must not change anything."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, "0")
    assert _gpu_side_render_var_sync_enabled() is False


@pytest.mark.parametrize("value", ["", "true", "yes", "2"])
def test_ovrtx_render_var_sync_rejects_non_boolean_values(monkeypatch, value):
    """Values other than 0/1 are a configuration error, not a silent fallback to the host wait."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(_DISABLE_LINUX_CUDA_CPU_SYNC_ENV, value)
    with pytest.raises(ValueError, match="Expected 0 or 1"):
        _gpu_side_render_var_sync_enabled()


class _RecordingRenderVar:
    """Stand-in for an OVRTX ``RenderVarOutput`` that records how the read was ordered.

    Any of OVRTX's ordering mechanisms counts, so the test stays about *whether* the read is
    ordered rather than which call carries it.
    """

    def __init__(self):
        self.ordering: list[str] = []

    def map(self, *, device, sync_stream):
        if sync_stream:
            self.ordering.append("gpu")
        recorder = self

        class _Mapping:
            def wait(self):
                recorder.ordering.append("host")

            def wait_on(self, stream):
                recorder.ordering.append("gpu")

        return contextlib.nullcontext(_Mapping())


@pytest.mark.parametrize(("gpu_side", "expected"), [(True, "gpu"), (False, "host")])
def test_ovrtx_map_render_var_orders_the_read_against_render_completion(monkeypatch, gpu_side, expected):
    """The read is ordered exactly once -- by a GPU-side barrier or a host block, never by neither.

    Ordering by neither is a silent race on half-written render output rather than a failure, so
    this asserts which mechanism ran and not which API call carries it.
    """
    sentinel = object()
    render_var = _RecordingRenderVar()
    monkeypatch.setattr(ovrtx_renderer_module, "_gpu_side_render_var_sync_enabled", lambda: gpu_side)
    monkeypatch.setattr(ovrtx_renderer_module.wp, "from_dlpack", lambda mapping: sentinel)

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._device = "cuda:0"
    renderer._warp_device = types.SimpleNamespace(stream=types.SimpleNamespace(cuda_stream=99))
    with renderer._map_render_var_to_dlpack(render_var) as array:
        assert array is sentinel

    assert render_var.ordering == [expected]


def test_ovrtx_cleanup_releases_only_the_given_render_data():
    """``cleanup`` releases the render data's own buffers and leaves the renderer usable.

    The stage queries, tensor bindings and render products the renderer holds are shared with
    every other camera that resolved to it, so a single camera's cleanup must not take them.
    """
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._initialized_scene = True

    render_data = _make_ovrtx_render_data()
    render_data.warp_buffers = {"rgba": wp.zeros((8, 16, 4), dtype=wp.uint8, device="cpu")}
    render_data.renderer_info = {"semantic_segmentation": {"idToLabels": {}}}
    render_data.ppisp_pipeline = object()

    renderer.cleanup(render_data)

    assert render_data.warp_buffers == {}
    assert render_data.renderer_info == {}
    assert render_data.ppisp_pipeline is None

    assert renderer._render_product_paths == ["/Render/RenderProduct_camera"]
    assert renderer._initialized_scene is True


def test_ovrtx_cleanup_without_render_data_keeps_renderer_state():
    """``cleanup(None)`` has nothing to release and must not disturb the renderer."""
    renderer = _make_ovrtx_renderer_without_backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._initialized_scene = True

    renderer.cleanup(None)

    assert renderer._render_product_paths == ["/Render/RenderProduct_camera"]
    assert renderer._initialized_scene is True


class _RecordingBinding:
    def __init__(self, events: list[str], name: str):
        self._events = events
        self._name = name

    def unbind(self) -> None:
        self._events.append(f"unbind:{self._name}")


def _make_legacy_renderer_with_backend(events: list[str]) -> OVRTXRenderer:
    """Build a legacy-path renderer whose backend calls are recorded into ``events``."""

    class Backend:
        def reset_stage(self) -> None:
            events.append("reset_stage")

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = False
    renderer._camera_xform_binding = _RecordingBinding(events, "camera")
    renderer._object_xform_binding = _RecordingBinding(events, "object")
    renderer._deformable_points_binding = _RecordingBinding(events, "deformable")
    renderer._particle_points_binding = _RecordingBinding(events, "particle")
    renderer._cable_points_binding = _RecordingBinding(events, "cable")
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [1]
    renderer._particle_visual_offsets = [0]
    renderer._particle_visual_counts = [1]
    renderer._particle_workaround_applied = True
    renderer._cable_segment_counts = [1]
    renderer._renderer = Backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    return renderer


def _make_ovstage_renderer_with_backend(events: list[str]) -> OVRTXRenderer:
    """Build an ovstage-path renderer whose backend calls are recorded into ``events``."""

    class Completion:
        def wait(self) -> None:
            return

    class Stage:
        def release_query(self, query):
            events.append(f"release_query:{query}")
            return Completion()

    class StagePaths:
        def destroy_path_list(self, path_list) -> None:
            events.append(f"destroy_path_list:{path_list}")

    class Backend:
        def detach_ovstage(self) -> None:
            events.append("detach_ovstage")

    class ExitStack:
        def close(self) -> None:
            events.append("exit_stack_close")

    renderer = _make_ovrtx_renderer_without_backend()
    renderer._use_ovstage = True
    renderer._stage = Stage()
    renderer._stage_paths = StagePaths()
    renderer._camera_xform_query = "camera"
    renderer._camera_paths_list = "camera"
    renderer._object_xform_query = "object"
    renderer._object_paths_list = "object"
    renderer._deformable_points_query = "deformable"
    renderer._deformable_paths_list = "deformable"
    renderer._particle_points_query = "particle"
    renderer._particle_paths_list = "particle"
    renderer._cable_points_query = "cable"
    renderer._cable_paths_list = "cable"
    renderer._object_newton_indices = object()
    renderer._deformable_particle_offsets = [0]
    renderer._deformable_particle_counts = [1]
    renderer._particle_visual_offsets = [0]
    renderer._particle_visual_counts = [1]
    renderer._renderer = Backend()
    renderer._ovstage_exit_stack = ExitStack()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._output_id_color_buffers = {"semantic_segmentation": object()}
    renderer._initialized_scene = True
    renderer._current_ordinal = 7
    return renderer


def test_ovrtx_close_releases_legacy_renderer_state():
    """``close`` unbinds the tensor bindings and resets the stage the renderer owns."""
    events: list[str] = []
    renderer = _make_legacy_renderer_with_backend(events)

    renderer.close()

    assert events == [
        "unbind:camera",
        "unbind:object",
        "unbind:deformable",
        "unbind:particle",
        "unbind:cable",
        "reset_stage",
    ]
    assert renderer._camera_xform_binding is None
    assert renderer._object_xform_binding is None
    assert renderer._object_transform_buffer is None
    assert renderer._deformable_points_binding is None
    assert renderer._particle_points_binding is None
    assert renderer._cable_points_binding is None
    assert renderer._particle_workaround_applied is False
    assert renderer._renderer is None
    assert renderer._render_product_paths == []
    assert renderer._output_id_color_buffers == {}
    assert renderer._initialized_scene is False


def test_ovrtx_close_releases_ovstage_renderer_state():
    """``close`` releases the queries and path lists, then detaches before closing the ExitStack.

    The ExitStack owns the ovstage ``Stage`` and ``PathDictionary`` as context managers, so it is the
    only thing that releases them — ``ExitStack`` has no finalizer, and garbage collection never
    invokes ``__exit__``. Detaching first avoids a use-after-free while the renderer still references
    the stage.
    """
    events: list[str] = []
    renderer = _make_ovstage_renderer_with_backend(events)

    renderer.close()

    assert events == [
        "release_query:camera",
        "destroy_path_list:camera",
        "release_query:object",
        "destroy_path_list:object",
        "release_query:deformable",
        "destroy_path_list:deformable",
        "release_query:particle",
        "destroy_path_list:particle",
        "release_query:cable",
        "destroy_path_list:cable",
        "detach_ovstage",
        "exit_stack_close",
    ]
    assert renderer._camera_xform_query is None
    assert renderer._particle_paths_list is None
    assert renderer._cable_points_query is None
    assert renderer._cable_paths_list is None
    assert renderer._object_newton_indices is None
    assert renderer._renderer is None
    assert renderer._ovstage_exit_stack is None
    assert renderer._stage is None
    assert renderer._stage_paths is None
    assert renderer._render_product_paths == []
    assert renderer._output_id_color_buffers == {}
    assert renderer._initialized_scene is False
    assert renderer._current_ordinal is None


def test_ovrtx_close_is_idempotent():
    """A second ``close`` releases nothing again, so a repeated teardown cannot double-free."""
    events: list[str] = []
    renderer = _make_ovstage_renderer_with_backend(events)

    renderer.close()
    events.clear()
    renderer.close()

    assert events == []


@pytest.mark.parametrize("newton_is_manager", [True, False])
def test_ovstage_pose_source_follows_the_simulating_backend(monkeypatch, newton_is_manager):
    """The renderer asks for a Newton model, so one exists under OVPhysX too.

    Picking the pose source by the model's presence would read OVPhysX's poses out of a backend
    that is not simulating them, so the choice has to follow the physics manager instead.
    """
    newton_physics = pytest.importorskip("isaaclab_newton.physics")

    class _OtherManager:
        pass

    manager = newton_physics.NewtonManager if newton_is_manager else _OtherManager
    monkeypatch.setattr(
        ovrtx_renderer_module,
        "SimulationContext",
        types.SimpleNamespace(instance=lambda: types.SimpleNamespace(physics_manager=manager)),
    )
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)

    assert renderer._newton_owns_poses() is newton_is_manager


def test_ovstage_world_transform_read_resolves_rows_through_the_group_maps():
    """A group's tensor carries the whole column, not just the queried rows.

    The queried prims are a subset in their own order, so a read that walks the tensor
    instead of the group's maps mismatches every row and runs off the end of the group once
    the column is larger than the query -- which is any scene past a couple of environments.
    """
    import numpy as np
    import ovstage

    wanted = ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    # Column for four prims; the two we asked for sit at rows 3 and 1, reversed.
    column = np.zeros((4, 4, 4), dtype=np.float64)
    column[3] = np.diag([1.0, 1.0, 1.0, 1.0])
    column[3][3, :3] = [7.0, 0.0, 0.0]
    column[1] = np.diag([1.0, 1.0, 1.0, 1.0])
    column[1][3, :3] = [9.0, 0.0, 0.0]

    class _Group:
        prim_count = 2
        data_count = 2
        tensor_count = 1

        def array(self, index: int):
            return column.reshape(-1)

        def prim_index(self, local: int) -> int:
            if not 0 <= local < self.prim_count:
                raise IndexError(f"prim index {local} out of range [0, {self.prim_count})")
            return local

        def data_row_index(self, local: int) -> int:
            return (3, 1)[local]

    class _Stage:
        def __init__(self):
            self._pending = [_Group()]

        def query_from_path_list(self, path_list):
            return object()

        def read_attributes(self, query, tokens, ordinals):
            return types.SimpleNamespace(release=lambda: types.SimpleNamespace(wait=lambda: None))

        def fetch_read_next(self, read):
            return self._pending.pop() if self._pending else None

        def release_group(self, group) -> None: ...

        def release_query(self, query):
            return types.SimpleNamespace(wait=lambda: None)

    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer._stage = _Stage()
    renderer._stage_paths = types.SimpleNamespace(
        create_path_list_from_strings=lambda paths: object(),
        intern_token=lambda name: 1,
        destroy_path_list=lambda path_list: None,
    )
    assert ovstage.OrdinalRange.latest(1) is not None

    transforms = renderer._read_world_transforms_ovstage(wanted)

    assert transforms[0][3][:3].tolist() == [7.0, 0.0, 0.0]
    assert transforms[1][3][:3].tolist() == [9.0, 0.0, 0.0]


@pytest.mark.parametrize("handover", [True, False])
def test_ovstage_xform_mechanism_follows_the_installed_ovstage(monkeypatch, handover: bool):
    """Both mechanisms fail silently on the version that does not render them.

    OVStage 0.1 stores a handover to a physics-domain prim without drawing it, and 0.2 does the
    same with a mapped fill to a camera, so nothing downstream reports a mechanism picked wrongly.
    """
    monkeypatch.setattr(ovrtx_renderer_module, "XFORM_HANDOVER", handover)
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    transforms = wp.zeros(2, dtype=wp.mat44d, device="cpu")
    calls: list[str] = []

    class _Mapping:
        def wait(self) -> None: ...

        def groups(self) -> list:
            return []

        def unmap(self):
            return types.SimpleNamespace(wait=lambda: None)

    class _Stage:
        def map_attribute(self, *args, **kwargs):
            calls.append("mapped fill")
            return _Mapping()

        def write_attribute(self, *args, **kwargs):
            calls.append("handover")
            return types.SimpleNamespace(wait=lambda: None)

    renderer._stage = _Stage()
    renderer._current_ordinal = 7
    renderer._warp_device = types.SimpleNamespace(stream=types.SimpleNamespace(cuda_stream=0))

    renderer._author_xforms_ovstage(object(), transforms)

    assert calls == ["handover" if handover else "mapped fill"]


def test_ovstage_render_commits_above_an_ordinal_the_physics_consumer_sealed_in_between():
    """Holding an output ordinal across another consumer's seal fails every write at it.

    Ordinals are shared with OVPhysX, which seals a control ordinal whenever a ``mode="reset"``
    gravity event runs. An ordinal reserved before that seal is at or below the write floor by
    the time the next frame writes at it, and OVStage rejects it with ``WRITE_FLOOR_VIOLATION``.
    """
    from isaaclab_ov.ovstage_ordinals import OvStageOrdinalLanes

    lanes = OvStageOrdinalLanes()
    sealed: list[int] = []

    class _SharedStage:
        ordinals = lanes

        def seal(self, ordinal: int) -> None:
            sealed.append(ordinal)

    class _Backend:
        def step(self, *, render_products, delta_time, ordinal):
            return {}

    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer._initialized_scene = True
    renderer._renderer = _Backend()
    renderer._render_product_paths = ["/Render/RenderProduct_camera"]
    renderer._visual_material_writer_ref = None
    renderer._shared_stage = _SharedStage()
    renderer._current_ordinal = None
    render_data = types.SimpleNamespace(ppisp_pipeline=None)

    renderer._render_ovstage(render_data)
    control_ordinal = lanes.next_control()
    renderer._render_ovstage(render_data)

    assert sealed[-1] > control_ordinal


def test_ovstage_rereads_the_bound_scene_data_backend_every_frame():
    """Reading the backend is what refreshes it, so a cached read would freeze the scene."""
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer._object_newton_indices = None
    rows = wp.array([0, 1], dtype=wp.int32, device="cpu")
    poses = wp.zeros(2, dtype=wp.transformf, device="cpu")
    reads = []

    class _Backend:
        @property
        def transforms(self):
            reads.append(1)
            return types.SimpleNamespace(transforms=poses)

    renderer._object_scene_data_backend = _Backend()
    renderer._object_scene_data_rows = rows

    for _ in range(2):
        got_poses, got_rows = renderer._object_body_poses_ovstage()
        assert got_poses is poses
        assert got_rows is rows
    assert len(reads) == 2
