# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ordinal bookkeeping for an OVStage shared by OVPhysX and a render consumer.

Every committed OVStage write lands at an *ordinal*, a version number the application owns and
advances. OVPhysX ingests application edits by draining a range of them, and it must never drain
an ordinal carrying its own simulation output: doing so re-ingests the last result as if it were
a newly authored change and corrupts the simulation. OVPhysX's ovstage integration guide calls
this the ordinal-lane rule, and makes the application responsible for enforcing it.

:class:`OvStageOrdinalLanes` is that enforcement point. Ordinals come from one monotonic counter
so two consumers writing to the same stage cannot collide or write beneath the write floor the
other already sealed, and each ordinal is tagged with the lane it was reserved for:

* **control** -- application edits OVPhysX must ingest. The only ordinals ever passed to
  ``update_from_ovstage``.
* **output** -- simulation output written back for a render consumer to draw. Never passed to
  ``update_from_ovstage``.

Lanes are interleaved on demand rather than assigned by parity, because the two consumers do not
allocate in lockstep: a parity scheme would let one lane fall behind the sealed write floor and
turn its next write into a write-floor violation.
"""

from __future__ import annotations

# First ordinal an OVStage can be written at; ordinal 0 is its empty, unwritten state. Population
# writes the initial scene here and the application seals it before either consumer attaches, so
# both lanes start above it.
POPULATION_ORDINAL = 1


class OvStageOrdinalLanes:
    """Monotonic OVStage ordinal source split into a control lane and an output lane.

    Ordinals increase across both lanes, so sealing one lane's ordinal never strands the other
    below the write floor. Which lane an ordinal belongs to is recorded rather than derived, so
    :meth:`drain_range` can reject a drain that would feed simulation output back into OVPhysX.
    """

    def __init__(self, population_ordinal: int = POPULATION_ORDINAL) -> None:
        """Initialize the lanes above a sealed population ordinal.

        Args:
            population_ordinal: Ordinal the initial scene was populated and sealed at. Both lanes
                allocate strictly above it.

        Raises:
            ValueError: If ``population_ordinal`` is not a positive ordinal.
        """
        if population_ordinal < 1:
            raise ValueError(f"population_ordinal must be a positive OVStage ordinal, got {population_ordinal}.")
        self._population_ordinal = population_ordinal
        self._next_ordinal = population_ordinal + 1
        self._latest_control: int | None = None
        self._latest_output: int | None = None

    @property
    def population_ordinal(self) -> int:
        """Ordinal the initial scene was populated and sealed at."""
        return self._population_ordinal

    @property
    def latest_control(self) -> int | None:
        """Most recently reserved control ordinal, or ``None`` before the first reservation."""
        return self._latest_control

    @property
    def latest_output(self) -> int | None:
        """Most recently reserved output ordinal, or ``None`` before the first reservation."""
        return self._latest_output

    def next_control(self) -> int:
        """Reserve the next ordinal for application edits OVPhysX must ingest.

        Returns:
            The reserved ordinal. Seal it, then drain it with the range from :meth:`drain_range`.
        """
        self._latest_control = self._reserve()
        return self._latest_control

    def next_output(self) -> int:
        """Reserve the next ordinal for simulation output OVPhysX must never ingest.

        Returns:
            The reserved ordinal. Seal it so a render consumer can read it, but never name it in
            an ``update_from_ovstage`` range.
        """
        self._latest_output = self._reserve()
        return self._latest_output

    def drain_range(self, ordinal: int) -> tuple[int, int]:
        """Return the closed ``update_from_ovstage`` range for a control ordinal.

        Naming only the new ordinal keeps producer ownership explicit; OVPhysX skips ordinals it
        has already consumed, so a wider range would work but would obscure which edits are new.

        Args:
            ordinal: Control ordinal previously reserved by :meth:`next_control`.

        Returns:
            The inclusive ``(start, end)`` range covering ``ordinal`` alone.

        Raises:
            ValueError: If ``ordinal`` was not reserved as a control ordinal. Draining an output
                ordinal would feed simulation output back into OVPhysX.
        """
        if ordinal != self._latest_control:
            raise ValueError(
                f"Ordinal {ordinal} is not the current control ordinal ({self._latest_control}); draining it risks"
                " feeding simulation output back into OVPhysX."
            )
        return (ordinal, ordinal)

    def _reserve(self) -> int:
        """Take the next ordinal from the shared counter."""
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        return ordinal
