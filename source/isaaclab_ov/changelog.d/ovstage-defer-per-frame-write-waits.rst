Changed
^^^^^^^

* Changed the OVRTX ovstage path to complete its per-frame attribute writes at the existing
  ``advance_write_floor`` barrier instead of waiting on each write individually, removing two to
  five host stalls per rendered frame. Those writes are now ordered against their producing Warp
  kernels with a recorded CUDA event rather than by draining the Warp stream, so ovstage waits on
  the producing kernel alone instead of on all work queued on that stream. No migration is
  required: this applies only when ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``, the renderer's public API is
  unchanged, and the legacy OVRTX write path is untouched.
