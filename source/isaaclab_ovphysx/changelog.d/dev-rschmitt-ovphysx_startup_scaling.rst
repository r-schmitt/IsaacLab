Changed
^^^^^^^

* Added an opt-in :attr:`~isaaclab_ovphysx.physics.OvPhysxCfg.filter_env_collisions_with_env_ids` flag
  (default ``False``) to filter cross-environment collisions with the PhysX-native environment-id
  broadphase filter instead of USD collision groups. When enabled,
  :class:`~isaaclab.scene.InteractiveScene` skips authoring one ``PhysicsCollisionGroup`` per
  environment, which removes the per-environment collision-group parsing that dominated simulation
  start at high environment counts. It defaults to ``False`` because skipping that authoring currently
  also suppresses the lazy materialization of per-environment USD clones that USD-reading sensors
  (camera, ray caster, contact) depend on; enable it only for headless, sensor-free runs.
