Changed
^^^^^^^

* Enabled PhysX-native environment-id collision filtering for OvPhysX by default, controlled by the
  new :attr:`~isaaclab_ovphysx.physics.OvPhysxCfg.filter_env_collisions_with_env_ids` flag. When
  enabled, :class:`~isaaclab.scene.InteractiveScene` skips authoring one ``PhysicsCollisionGroup`` per
  environment and relies on the broadphase env-id filter, which removes the per-environment
  collision-group parsing that dominated simulation start at high environment counts. Set the flag to
  ``False`` to restore the legacy per-environment collision groups.
