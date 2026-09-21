"""The experiment graph: lineage over scientific candidates.

An experiment is a DAG of :class:`~xaytune.core.domain.experiment.ExperimentNode`,
not a list. A candidate is derived from one or more predecessors, and the edges
are what let the record answer *why does this node exist* — which is the
question ADR-011's four-level lineage is built around.

**A node is a scientific candidate, never an infrastructure attempt.** Worker
restarts, preemptions and checkpoint restores create `RunAttempt`s and never
touch this graph (ADR-003). If a retry ever appears here as a branch, the
comparability rule has been broken somewhere upstream.

Traversal uses recursive CTEs rather than repeated round trips: ancestry is
read on every branching decision, and walking it a row at a time from Python
turns one question into a query per generation.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from xaytune.core.domain.experiment import ExperimentNode
from xaytune.core.errors import DomainError

__all__ = ["CandidateComparison", "ExperimentGraph", "LineageError"]


class LineageError(DomainError):
    """A proposed edge would make the lineage graph meaningless."""


@dataclass(frozen=True)
class CandidateComparison:
    """What two candidates share and where they diverge.

    The question a planner asks before deciding whether a result is evidence
    about one hypothesis or two.

    Attributes:
        common_ancestors: Nodes both descend from, nearest first. Empty for
            candidates from independent roots, which is what makes them
            independent.
        same_candidate: Whether the two carry the same candidate fingerprint --
            the same scientific proposition, possibly run twice.
    """

    left: ExperimentNode
    right: ExperimentNode
    common_ancestors: tuple[ExperimentNode, ...]
    same_candidate: bool

    @property
    def nearest_common_ancestor(self) -> ExperimentNode | None:
        """The closest node both descend from, if any."""
        return self.common_ancestors[0] if self.common_ancestors else None

    @property
    def is_comparable(self) -> bool:
        """Whether these are alternatives rather than the same thing twice.

        Two nodes carrying the same fingerprint are replicates, and treating
        them as competing candidates would count one hypothesis twice.
        """
        return not self.same_candidate


class ExperimentGraph:
    """Lineage queries over an experiment's nodes."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    # ---- direct relations ----------------------------------------------

    def parents(self, node_id: str) -> tuple[ExperimentNode, ...]:
        """Return the node's immediate predecessors, in creation order."""
        return self._nodes(
            "SELECT n.payload_json FROM experiment_nodes n "
            "JOIN experiment_edges e ON e.parent_id = n.id "
            "WHERE e.child_id = ? ORDER BY n.created_at, n.id",
            (node_id,),
        )

    def children(self, node_id: str) -> tuple[ExperimentNode, ...]:
        """Return the node's immediate successors, in creation order."""
        return self._nodes(
            "SELECT n.payload_json FROM experiment_nodes n "
            "JOIN experiment_edges e ON e.child_id = n.id "
            "WHERE e.parent_id = ? ORDER BY n.created_at, n.id",
            (node_id,),
        )

    def roots(self, experiment_id: str) -> tuple[ExperimentNode, ...]:
        """Return the experiment's nodes with no predecessor.

        More than one is normal: an experiment may start from several
        independent hypotheses rather than a single seed.
        """
        return self._nodes(
            "SELECT n.payload_json FROM experiment_nodes n "
            "WHERE n.experiment_id = ? "
            "AND NOT EXISTS (SELECT 1 FROM experiment_edges e WHERE e.child_id = n.id) "
            "ORDER BY n.created_at, n.id",
            (experiment_id,),
        )

    def leaves(self, experiment_id: str) -> tuple[ExperimentNode, ...]:
        """Return the experiment's nodes with no successor — its frontier."""
        return self._nodes(
            "SELECT n.payload_json FROM experiment_nodes n "
            "WHERE n.experiment_id = ? "
            "AND NOT EXISTS (SELECT 1 FROM experiment_edges e WHERE e.parent_id = n.id) "
            "ORDER BY n.created_at, n.id",
            (experiment_id,),
        )

    # ---- transitive relations ------------------------------------------

    def ancestors(self, node_id: str) -> tuple[ExperimentNode, ...]:
        """Return every node this one descends from, nearest first.

        Ordered by distance so the first entry is the immediate predecessor a
        result should be compared against, not an arbitrary root.
        """
        return self._traverse(node_id, up=True)

    def descendants(self, node_id: str) -> tuple[ExperimentNode, ...]:
        """Return every node derived from this one, nearest first."""
        return self._traverse(node_id, up=False)

    def lineage(self, node_id: str) -> tuple[ExperimentNode, ...]:
        """Return the path from a root to this node, root first.

        With several ancestral paths -- a candidate derived from two
        predecessors has more than one -- this returns the longest, which is
        the full derivation story rather than a shortcut through it.
        """
        best: tuple[ExperimentNode, ...] = ()
        node = self._node(node_id)
        if node is None:
            return ()

        for parent in self.parents(node_id):
            path = self.lineage(str(parent.id))
            if len(path) > len(best):
                best = path
        return (*best, node)

    def is_descendant_of(self, node_id: str, ancestor_id: str) -> bool:
        """Whether *node_id* derives from *ancestor_id*, at any depth."""
        return any(str(n.id) == ancestor_id for n in self.ancestors(node_id))

    # ---- comparison ------------------------------------------------------

    def compare(self, left_id: str, right_id: str) -> CandidateComparison:
        """Return what two candidates share and where they diverge.

        Raises:
            LineageError: If either node does not exist, or they belong to
                different experiments -- comparing across experiments would
                answer a question neither experiment asked.
        """
        left, right = self._node(left_id), self._node(right_id)
        if left is None or right is None:
            missing = left_id if left is None else right_id
            raise LineageError(f"node {missing} does not exist")
        if left.experiment_id != right.experiment_id:
            raise LineageError(
                f"nodes {left_id} and {right_id} belong to different "
                f"experiments; lineage is meaningful only within one"
            )

        right_ancestry = {str(n.id) for n in self.ancestors(right_id)}
        common = tuple(node for node in self.ancestors(left_id) if str(node.id) in right_ancestry)
        return CandidateComparison(
            left=left,
            right=right,
            common_ancestors=common,
            same_candidate=left.training_fingerprint == right.training_fingerprint,
        )

    # ---- validation ------------------------------------------------------

    def validate_parents(self, node: ExperimentNode) -> None:
        """Refuse a node whose lineage would be unsound.

        Three ways it can be, all checked before the node is written:

        * **A parent that does not exist.** The foreign key would catch it, but
          as an integrity error rather than something a caller can act on.
        * **A parent in another experiment.** Nothing in the schema forbids it,
          and it would make ancestry span experiments -- so "compare against the
          previous candidate" could reach into work that answered a different
          question entirely.
        * **A cycle.** A node cannot descend from itself, directly or through
          any path, because then no node in the loop has a derivation.

        Raises:
            LineageError: With the specific parent and the reason.
        """
        for parent_id in node.parent_ids:
            # Structural first: a node being created does not exist yet, so a
            # self-edge would otherwise be reported as a missing parent, which
            # sends the reader looking for the wrong thing.
            if str(parent_id) == str(node.id):
                raise LineageError(f"node {node.id} cannot be its own parent")

            parent = self._node(str(parent_id))
            if parent is None:
                raise LineageError(f"node {node.id} names parent {parent_id}, which does not exist")
            if parent.experiment_id != node.experiment_id:
                raise LineageError(
                    f"node {node.id} names parent {parent_id} from experiment "
                    f"{parent.experiment_id}; lineage may not span experiments"
                )
            if self.is_descendant_of(str(parent_id), str(node.id)):
                raise LineageError(
                    f"node {node.id} cannot take {parent_id} as a parent: "
                    f"{parent_id} already descends from it, so the edge would "
                    f"close a cycle and leave neither node with a derivation"
                )

    # ---- machinery -------------------------------------------------------

    def _traverse(self, node_id: str, *, up: bool) -> tuple[ExperimentNode, ...]:
        # One recursive CTE rather than a query per generation: ancestry is
        # read on every branching decision, and depth is not bounded.
        near, far = ("child_id", "parent_id") if up else ("parent_id", "child_id")
        rows = self._connection.execute(
            f"""
            WITH RECURSIVE walk(id, depth) AS (
                SELECT {far}, 1 FROM experiment_edges WHERE {near} = ?
                UNION
                SELECT e.{far}, walk.depth + 1
                FROM experiment_edges e JOIN walk ON e.{near} = walk.id
            )
            SELECT n.payload_json, MIN(walk.depth) AS depth
            FROM walk JOIN experiment_nodes n ON n.id = walk.id
            GROUP BY n.id
            ORDER BY depth, n.created_at, n.id
            """,  # noqa: S608 - both interpolations come from the literal pair above
            (node_id,),
        ).fetchall()
        return tuple(ExperimentNode.model_validate_json(row["payload_json"]) for row in rows)

    def _node(self, node_id: str) -> ExperimentNode | None:
        row = self._connection.execute(
            "SELECT payload_json FROM experiment_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return None if row is None else ExperimentNode.model_validate_json(row["payload_json"])

    def _nodes(self, sql: str, params: tuple[str, ...]) -> tuple[ExperimentNode, ...]:
        rows = self._connection.execute(sql, params).fetchall()
        return tuple(ExperimentNode.model_validate_json(row["payload_json"]) for row in rows)
