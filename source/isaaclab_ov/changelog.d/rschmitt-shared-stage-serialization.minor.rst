Added
^^^^^

* Added :class:`~isaaclab_ov.stage_usda.SharedStageUsda`, the single serialization of the host USD
  stage that every OVStage consumer populates from. OVPhysX and OVRTX previously serialized the
  stage independently, so the scene was exported and parsed twice and the two copies could
  disagree: whatever one consumer authored after the other had serialized was missing from that
  other consumer's copy. Consumers may also contribute USDA that has no place on the host stage,
  such as OVRTX's ``/Render`` scope, before the serialization is first requested.

Changed
^^^^^^^

* Changed :class:`~isaaclab_ov.physics.OvPhysxManager` to populate OVStage from the shared
  serialization, which keeps one row per :class:`~isaaclab.cloner.ClonePlan` source instead of only
  ``env_0``. A scene whose environments are not all clones of ``env_0`` now hands OVPhysX every
  prototype it replicates rather than just the first, and the serialization is no longer flattened.
  Environments are still replicated in the physics runtime through ``physx.clone()``, so the
  ingestion cost of a large environment count is unchanged. A full-stage load, which
  :class:`~isaaclab_ov.assets.DeformableObject` requests, keeps its own flattened export.
