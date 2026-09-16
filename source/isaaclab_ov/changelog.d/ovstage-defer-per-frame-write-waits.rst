Changed
^^^^^^^

* Changed the OVRTX ovstage path to complete its per-frame attribute writes at the existing
  ``advance_write_floor`` barrier instead of waiting on each write individually, removing two to
  five host stalls per rendered frame. Those writes are then ordered against their producing Warp
  kernels with a recorded CUDA event rather than by draining the Warp stream, so ovstage waits on
  the producing kernel alone instead of on all work queued on that stream. The change is gated on
  ovstage 0.2 and later, which compute the prim hierarchy on the device; while that work runs on
  the host the per-write waits overlap it with the caller's preparation of the next write, and
  deferring them measures slower. The gate is resolved once at import by
  :mod:`isaaclab_ov.ovstage_compat`, alongside the hierarchy computation model. No migration is
  required: the public extras stay pinned to ``ovstage==0.1.1.355824``, so the per-write waits
  remain in force until that pin moves, this applies only when ``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``,
  the renderer's public API is unchanged, and the legacy OVRTX write path is untouched.
