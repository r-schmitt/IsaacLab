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
* Changed the OVRTX renderer's OVStage path to populate from the shared serialization rather than
  exporting the stage itself, so enabling it through ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` now exports the
  host stage once instead of twice. Each consumer still populates its own OVStage from that text, so
  it is still parsed once per consumer. Its render product is declared while the scene is built,
  early enough to reach the serialization before physics warms up, and the population therefore
  also carries what physics authors between scene construction and its warmup. The legacy path is
  unchanged and still exports on its own.
