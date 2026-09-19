Changed
^^^^^^^

* :meth:`~isaaclab_ov.physics.OvPhysxManager.set_gravity` now returns without authoring anything
  when the scene already runs with the requested gravity, compared in the float32 precision the
  value is written at. A ``mode="reset"`` randomization term that resamples a constant, which is
  how most tasks configure gravity randomization, previously sealed and drained a control ordinal
  on every reset. Draining is not free on a stage shared with a render consumer: OVPhysX
  re-ingests the stage, and any structural edit the render consumer made is applied along with
  the gravity change.
