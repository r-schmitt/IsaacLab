Changed
^^^^^^^

* Changed the OVRTX renderer's OVStage path to author per-frame ``omni:xform`` for objects and
  cameras by filling the mapped column rather than handing the transforms over as a tensor. The
  handover was accepted and stored -- both the attribute and the world matrix computed from it read
  back correctly -- but on a stage whose population includes the physics domain it never reached the
  render, leaving prims drawn at their pre-write transform or not at all. Filling the mapped column
  is picked up, and is also faster: on a scene of one rigid body and one camera per environment,
  authoring object transforms went from 1.89 ms to 0.40 ms per frame at 64 environments and from
  4.20 ms to 0.47 ms at 256, taking whole-frame time down by 17% and 30% respectively. Rendering
  output is unchanged.
