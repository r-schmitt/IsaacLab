Changed
^^^^^^^

* Changed the OVRTX renderer's OVStage path to author per-frame object transforms relative to each
  prim's parent, rather than pinning ``omni:resetXformStack`` and authoring absolute world
  transforms. The inverse parent transform is read from the stage once, on the first frame, since
  the parents of bound prims do not move. Rendering output is unchanged, and the pin's failure mode
  is avoided: on a stage whose population includes the physics domain, resetting the xform stack is
  not honored for simulated prims, which left them composed onto their environment root and drawn
  at twice its offset, outside the frustum.
