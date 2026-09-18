Changed
^^^^^^^

* Changed the OVRTX renderer's OVStage path to take per-frame body poses from the active physics
  backend's :class:`~isaaclab.scene_data.SceneDataBackend` unless Newton is the physics manager, in
  which case it still reads Newton state directly. Under OVPhysX the poses previously reached the
  renderer through the Newton model the renderer requests rather than from the backend actually
  simulating them; a backend publishing a transform format other than ``wp.transformf`` is declined
  at setup rather than misread per frame. Newton-backed simulations are unaffected.
