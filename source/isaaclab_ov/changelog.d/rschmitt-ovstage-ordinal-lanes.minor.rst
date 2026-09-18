Added
^^^^^

* Added :class:`~isaaclab_ov.ovstage_ordinals.OvStageOrdinalLanes`, which hands out OVStage
  ordinals from a single monotonic counter split into a control lane for application edits OVPhysX
  ingests and an output lane for simulation output it must never ingest. Sharing one counter lets
  two consumers write to the same stage without colliding or writing beneath a write floor the
  other already sealed, and ``drain_range`` rejects an ordinal that was not reserved for the
  control lane so simulation output cannot be fed back into OVPhysX.
* Added :class:`~isaaclab_ov.stage.SharedOvStage`, which bundles an OVStage with the
  :class:`ovstage.PathDictionary` and the ordinal lanes bound to it, and owns their lifetime.
  Teardown destroys the path dictionary before the stage and is idempotent, so a stage handed to
  more than one consumer can be released by whichever one finishes last.

Changed
^^^^^^^

* Changed :class:`~isaaclab_ov.physics.OvPhysxManager` to own its OVStage through
  :class:`~isaaclab_ov.stage.SharedOvStage` rather than holding the stage directly. Scene updates
  such as :meth:`~isaaclab_ov.physics.OvPhysxManager.set_gravity` now reuse the stage's path
  dictionary instead of building one per call, and author and drain the same ordinals as before,
  so no behavior changes.
