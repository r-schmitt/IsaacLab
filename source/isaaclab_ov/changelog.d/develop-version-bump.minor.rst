Changed
^^^^^^^

* Changed the optional OV runtime pins to ``ovphysx==0.6.3``, ``ovrtx==0.5.0.377615``, and
  ``ovstage==0.2.0.377349``. Run ``uv sync --inexact --extra ov`` to pick up the new wheels.
  Moving the pins puts the compatibility paths already carried for these releases into force:
  the OVPhysX 0.6 ``warmup()`` / ``destroy()`` lifecycle entry points without the reversed-joint
  sign correction, RenderVar prim-path frame keys on OVRTX 0.5, and the ``GPU_INCREMENTAL``
  hierarchy computation model on OVStage 0.2, which moves world-transform computation off the
  host. Pinning the previous versions restores the previous behavior in each case.
