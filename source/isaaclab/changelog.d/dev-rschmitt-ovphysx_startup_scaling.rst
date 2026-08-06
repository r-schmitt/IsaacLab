Changed
^^^^^^^

* Improved the performance of :meth:`~isaaclab.scene.InteractiveScene.filter_collisions` so that
  PhysX and OvPhysX scene construction no longer slows down disproportionately as the number of
  environments grows. The authored USD is byte-for-byte identical, so simulation behavior is
  unchanged.
