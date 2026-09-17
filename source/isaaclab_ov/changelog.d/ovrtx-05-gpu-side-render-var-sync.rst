Changed
^^^^^^^

* Updated the optional Omniverse runtime pins to ``ovrtx==0.5.0.377615`` and
  ``ovstage==0.2.0.377349``. Reinstall the extras with ``uv sync --extra ov`` to pick them up.
  This pin moves :func:`~isaaclab_ov.stage.create_ovstage` onto the ovstage 0.2 GPU hierarchy
  computation model, which computes world transforms on the device instead of the host. OvPhysX
  stays at ``0.5.11``, which requests ``ovstage==0.1.1.355824``; a uv override forces the newer
  ovstage over that request, so OvPhysX is untested against this pin combination.
* Changed OVRTX camera-output reads on Linux to order against render completion with a GPU-side
  wait on the consuming Warp stream, which is the ordering every other platform already uses. The
  host wait that measured faster on OVRTX 0.4 dominates the frame on OVRTX 0.5, where dropping it
  is worth several times the end-to-end throughput at typical resolutions and environment counts.
  Set ``ISAAC_LAB_OVRTX_DISABLE_LINUX_CUDA_CPU_SYNC=0`` to restore the host wait. Camera outputs
  themselves are unchanged.
