# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX collision-group authoring for clones."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


def filter_collisions(
    stage: Usd.Stage,
    physicsscene_path: str,
    collision_root_path: str,
    prim_paths: list[str],
    global_paths: list[str] = [],
) -> None:
    """Create inverted collision groups for clones (PhysX only).

    Sets PhysX scene attributes and collision groups on the prim at ``physicsscene_path``
    (no PhysxSchema import). Call only when the physics backend is PhysX; Newton uses
    its own collision/world handling and does not use USD PhysX collision groups.

    Creates one PhysicsCollisionGroup per prim under ``collision_root_path``, enabling
    inverted filtering so clones don't collide across groups. Optionally adds a global
    group that collides with all.

    Args:
        stage: USD stage.
        physicsscene_path: Path to PhysicsScene prim.
        collision_root_path: Root scope for collision groups.
        prim_paths: Per-clone prim paths.
        global_paths: Optional global-collider paths.

    """
    # Deferred: importing pxr from the kit-less usd-core wheel before Kit boots corrupts Kit's
    # own USD runtime. Keeping it in the body means resolving the ``cloner.filter_collisions``
    # attribute stays pxr-free — only calling it, on a live PhysX stage, pulls pxr in.
    from pxr import Sdf, Usd, UsdGeom  # noqa: PLC0415

    scene_prim = stage.GetPrimAtPath(physicsscene_path)
    # We invert the collision group filters for more efficient collision filtering across environments
    invert_attr = scene_prim.CreateAttribute("physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool)
    invert_attr.Set(True)

    # Make sure we create the collision_scope in the RootLayer since the edit target
    # may be a live layer in the case of Live Sync.
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetRootLayer())):
        UsdGeom.Scope.Define(stage, collision_root_path)

    # Loop-invariant handles hoisted out of the per-environment loop below: the collision-root
    # parent spec and the collider-collection API token list are identical for every group, so
    # resolving/constructing them once (instead of once per environment) removes O(num_envs)
    # redundant layer lookups and TokenListOp allocations from scene construction.
    collision_root_spec = stage.GetRootLayer().GetPrimAtPath(collision_root_path)
    colliders_api_schemas = Sdf.TokenListOp.Create({"CollectionAPI:colliders"})
    has_global = len(global_paths) > 0

    with Sdf.ChangeBlock():
        if has_global:
            global_collision_group_path = collision_root_path + "/global_group"
            # add collision group prim
            global_collision_group = Sdf.PrimSpec(
                collision_root_spec,
                "global_group",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            global_collision_group.SetInfo(Usd.Tokens.apiSchemas, colliders_api_schemas)

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                global_collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            global_includes_rel = Sdf.RelationshipSpec(global_collision_group, "collection:colliders:includes", False)
            global_includes_rel.targetPathList.appendedItems = list(global_paths)

            # filteredGroups rel
            global_filtered_groups = Sdf.RelationshipSpec(global_collision_group, "physics:filteredGroups", False)
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other. Per-environment groups are accumulated here and assigned in a single bulk write after the
            # loop: appending to this relationship one path at a time is O(num_envs) per call (list-op uniqueness
            # check), i.e. O(num_envs^2) overall, which dominates scene construction at high environment counts.
            global_filtered_paths = [global_collision_group_path]

        # set collision groups and filters
        for i, prim_path in enumerate(prim_paths):
            collision_group_path = collision_root_path + f"/group{i}"
            # add collision group prim
            collision_group = Sdf.PrimSpec(
                collision_root_spec,
                f"group{i}",
                Sdf.SpecifierDef,
                "PhysicsCollisionGroup",
            )
            # prepend collision API schema
            collision_group.SetInfo(Usd.Tokens.apiSchemas, colliders_api_schemas)

            # expansion rule
            expansion_rule = Sdf.AttributeSpec(
                collision_group,
                "collection:colliders:expansionRule",
                Sdf.ValueTypeNames.Token,
                Sdf.VariabilityUniform,
            )
            expansion_rule.default = "expandPrims"

            # includes rel
            includes_rel = Sdf.RelationshipSpec(collision_group, "collection:colliders:includes", False)
            includes_rel.targetPathList.Append(prim_path)

            # filteredGroups rel
            filtered_groups = Sdf.RelationshipSpec(collision_group, "physics:filteredGroups", False)
            # We are using inverted collision group filtering, which means objects by default don't collide across
            # groups. We need to add this group as a filtered group, so that objects within this group collide with
            # each other.
            filtered_groups.targetPathList.Append(collision_group_path)
            if has_global:
                filtered_groups.targetPathList.Append(global_collision_group_path)
                global_filtered_paths.append(collision_group_path)

        # Single bulk write of the global group's filtered-groups targets (see note above). This is
        # byte-for-byte identical to appending each path individually, but O(num_envs) instead of
        # O(num_envs^2).
        if has_global:
            global_filtered_groups.targetPathList.appendedItems = global_filtered_paths
