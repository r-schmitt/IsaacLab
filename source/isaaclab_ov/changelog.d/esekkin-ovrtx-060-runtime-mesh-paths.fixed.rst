* Fixed runtime meshes disappearing from OVRTX 0.6 renders after a color, material, UV, or texture update.
  OVRTX 0.6 stops rendering a USD reference re-added at a removed reference's prim path once it is written,
  so each updated mesh layer is now added at a new prim path.
