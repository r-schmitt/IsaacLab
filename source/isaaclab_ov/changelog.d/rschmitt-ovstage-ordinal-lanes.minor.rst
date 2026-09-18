Added
^^^^^

* Added :class:`~isaaclab_ov.ovstage_ordinals.OvStageOrdinalLanes`, which hands out OVStage
  ordinals from a single monotonic counter split into a control lane for application edits OVPhysX
  ingests and an output lane for simulation output it must never ingest. Sharing one counter lets
  two consumers write to the same stage without colliding or writing beneath a write floor the
  other already sealed, and ``drain_range`` rejects an ordinal that was not reserved for the
  control lane so simulation output cannot be fed back into OVPhysX.

Changed
^^^^^^^

* Changed :class:`~isaaclab_ov.physics.OvPhysxManager` to allocate its OVStage control ordinals
  through :class:`~isaaclab_ov.ovstage_ordinals.OvStageOrdinalLanes` instead of an internal
  counter. Scene updates such as :meth:`~isaaclab_ov.physics.OvPhysxManager.set_gravity` author
  and drain the same ordinals as before, so no behavior changes.
