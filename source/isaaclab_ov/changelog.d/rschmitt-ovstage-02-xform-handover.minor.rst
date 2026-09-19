Fixed
^^^^^

* Fixed OVRTX cameras never moving on OVStage 0.2, which left every frame rendered from inside
  the scene geometry. ``omni:xform`` is now authored by the mechanism the installed OVStage
  renders: 0.1 stores a tensor handed to ``write_attribute`` for a physics-domain prim without
  drawing it, so the mapped column is filled there, while 0.2 drops a mapped fill for camera
  prims and draws the handover. Neither failure is reported by the write, so the choice is made
  from the version by :func:`~isaaclab_ov.ovstage_compat.supports_xform_handover` rather than
  discovered at runtime. On OVStage 0.2 the handover is also zero-copy and stays flat as the
  bound prim count grows, measuring 0.26 ms against 1.22 ms at 4096 prims and 0.20 ms against
  9.56 ms at 32768.
