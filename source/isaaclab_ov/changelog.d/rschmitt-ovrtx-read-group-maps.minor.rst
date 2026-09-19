Fixed
^^^^^

* Fixed the OVRTX ovstage path failing with ``IndexError: prim index N out of range`` on any
  scene larger than a couple of environments, raised while reading the parent transforms that
  per-frame object poses are authored against. A read group's tensors carry the whole attribute
  column rather than only the queried rows, and neither the prims nor the rows need arrive in
  order, so rows are now resolved through the group's ``prim_index`` and ``data_row_index`` maps
  instead of by arrival order. Mapped writes, which allocate exactly the queried rows in order,
  are unaffected and now assert that.
