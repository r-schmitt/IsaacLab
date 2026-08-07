Changed
^^^^^^^

* Improved the performance of :meth:`~isaaclab.scene.InteractiveScene.filter_collisions` so that
  PhysX and OvPhysX scene construction no longer slows down disproportionately as the number of
  environments grows. The authored USD is byte-for-byte identical, so simulation behavior is
  unchanged.
* Added a :meth:`~isaaclab.physics.PhysicsManager.filters_cross_env_collisions_natively` hook that lets
  a physics backend skip per-environment ``PhysicsCollisionGroup`` authoring in
  :meth:`~isaaclab.scene.InteractiveScene.filter_collisions`, removing the per-environment
  collision-group parsing that dominates simulation start at high environment counts. The base default
  is ``False`` (author the groups); backends that filter cross-environment collisions natively override
  it to ``True``.

Added
^^^^^

* Added :func:`~isaaclab.utils.nsys_capture_range` and the ``ISAACLAB_NSYS_CAPTURE`` environment
  variable to bracket named code regions with NVTX ranges and an optional ``cudaProfilerApi`` capture
  window. Named :class:`~isaaclab.utils.Timer` regions (e.g. scene creation and simulation start) now
  emit these ranges so an nsys profile can attribute and capture startup phases.
