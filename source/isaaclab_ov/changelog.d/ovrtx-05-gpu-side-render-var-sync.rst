Changed
^^^^^^^

* Updated the optional Omniverse runtime pins to ``ovrtx==0.5.0.377615`` and
  ``ovstage==0.2.0.377349``. Reinstall the extras with ``uv sync --extra ov`` to pick them up.
  This pin moves :func:`~isaaclab_ov.stage.create_ovstage` onto the ovstage 0.2 GPU hierarchy
  computation model, which computes world transforms on the device instead of the host. OvPhysX
  stays at ``0.5.11``, which requests ``ovstage==0.1.1.355824``; a uv override forces the newer
  ovstage over that request, so OvPhysX is untested against this pin combination.
* Changed OVRTX camera-output reads to always order against render completion with a stream
  barrier on the consuming Warp stream, which is the ordering the OVRTX API is designed around and
  the only ordering non-Linux platforms used. Linux continues to block the host as well, so its
  correctness no longer rests on the host wait alone. Adding the barrier costs nothing measurable,
  because by the time the host wakes the barrier has already been satisfied. Camera outputs
  themselves are unchanged.
* Changed ``ISAAC_LAB_OVRTX_DISABLE_LINUX_CUDA_CPU_SYNC=1`` to drop only the Linux host wait,
  leaving the stream barrier in place. It remains an opt-in diagnostic and is not recommended: at
  1024 environments and 256 px it measures ~1.5x slower end to end, because a host that does not
  block queues the next step's work behind the barrier and exposes the full render latency.
