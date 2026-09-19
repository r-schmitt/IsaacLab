Changed
^^^^^^^

* Changed the OVRTX renderer's OVStage path to draw from the OVStage physics is attached to,
  acquired through :class:`~isaaclab_ov.ovstage_owner.OvStageOwner`, rather than opening a second
  stage of its own. The scene is now parsed once and resident once for the whole simulation instead
  of once per consumer, and the renderer takes up the structural edits it makes after that
  population -- its clone, scene partitions and render product camera relationship -- explicitly.
  Render-side writes go to output ordinals, which physics never drains, so simulation output is
    10|  never mistaken for authored physics input. The legacy path is unchanged.
