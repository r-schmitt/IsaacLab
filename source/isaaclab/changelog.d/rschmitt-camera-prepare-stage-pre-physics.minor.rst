Added
^^^^^

* Added :meth:`~isaaclab.sensors.Camera.prepare_stage_for_rendering`, which authors a camera's
  renderer-side USD setup (the backend's per-camera overrides and its stage preparation) and
  returns the resulting render spec. It depends only on the published clone plan and the authored
  camera prims, so it may run before the physics backend is warmed up, and it is idempotent.

Changed
^^^^^^^

* Changed :class:`~isaaclab.scene.InteractiveScene` to author every configured camera's
  renderer-side USD setup while the scene is built, instead of leaving it to sensor
  initialization after physics has warmed up. A renderer that shares one stage with physics has to
  contribute its content before physics attaches to that stage, which initialization is too late
  for. Cameras constructed outside a scene are unaffected and still author their setup during
  initialization.
