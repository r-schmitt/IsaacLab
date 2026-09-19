Fixed
^^^^^

* Fixed the OVRTX ovstage path failing with ``OvstageError: WRITE_FLOOR_VIOLATION`` on any task
  whose events change scene gravity, raised on the first frame written after the change. The
  renderer reserved its next output ordinal as soon as a frame was sealed and held it until the
  next frame, but ordinals are shared with OVPhysX: a control ordinal sealed in between (which
  is what ``randomize_physics_scene_gravity`` does on every reset) raises the write floor above
  the held ordinal, and OVStage then rejects every write at it. Output ordinals are now reserved
  when the first write of a frame needs one, so a seal by the other consumer can no longer strand
  the renderer beneath the write floor.
