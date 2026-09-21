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

    The question a planner asks before deciding whether two results are
    evidence about one hypothesis or two.

    Ancestry here is the **closure** — a node is in its own ancestry — because
    the common case is comparing a candidate against the one it was derived
    from. With strict ancestry, comparing a parent with its child finds nothing
    in common, which is the opposite of the truth.

    Attributes:
        common_ancestors: Every node both descend from, including either node
            itself. Order is by id: this is a set, and any depth ordering would
            have to pick one of the two nodes to measure from, which is what
            made an earlier version asymmetric.
        nearest_common_ancestors: The **lowest** common ancestors -- those with
            no other common ancestor below them. Plural because a DAG can have
            several incomparable ones, which a tree cannot.
        same_candidate: Whether the two carry the same candidate fingerprint.
    """

    left: ExperimentNode
    right: ExperimentNode
    common_ancestors: tuple[ExperimentNode, ...]
    nearest_common_ancestors: tuple[ExperimentNode, ...]
    same_candidate: bool

    @property
    def nearest_common_ancestor(self) -> ExperimentNode | None:
        """The single closest shared ancestor, when there is exactly one.

        ``None`` when the candidates share nothing **or** when they share
        several incomparable lowest ancestors. A DAG has no single answer in
        that case, and returning an arbitrary one would read as certainty.
        Callers that must handle it read :attr:`nearest_common_ancestors`.
        """
        if len(self.nearest_common_ancestors) == 1:
            return self.nearest_common_ancestors[0]
        return None

    @property
    def is_comparable(self) -> bool:
        """Whether these are alternatives rather than one candidate written twice.

        Two nodes carrying the same fingerprint are duplicate representations
        of a single scientific candidate, so treating them as competing
        alternatives would count one hypothesis twice. (Repeated *seeds* are
        runs under one node, not separate nodes -- see ADR-011.)
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

    def lineage_paths(self, node_id: str) -> tuple[tuple[ExperimentNode, ...], ...]:
        """Return every path from a root to this node, each root-first.

        **All** paths, not one. An earlier version returned the longest and
        called it "the full derivation story", which is false for a DAG: a
        candidate derived from two predecessors has two derivations, and
        picking the longer silently dropped the other. The domain has no notion
        of a primary parent, so there is no principled basis for choosing.

        Paths are ordered longest-first, then by the ids along them, so the
        result is deterministic.

        Built iteratively rather than by recursing per parent: depth is
        unbounded, and a recursive walk would both hit Python's limit on a deep
        graph and re-traverse shared ancestry once per merge point.
        """
        nodes, parents = self._closure(node_id)
        if node_id not in nodes:
            return ()

        # Depth-first from the node upwards, carrying the partial path. The
        # visited set is per-path, so a diamond yields both routes while a
        # cycle -- which validation forbids, but which a corrupted record could
        # still contain -- cannot loop forever.
        complete: list[tuple[ExperimentNode, ...]] = []
        stack: list[tuple[tuple[str, ...], frozenset[str]]] = [((node_id,), frozenset({node_id}))]
        while stack:
            path, seen = stack.pop()
            unvisited = [p for p in parents.get(path[0], ()) if p not in seen]
            if not unvisited:
                complete.append(tuple(nodes[i] for i in path))
                continue
            for parent in unvisited:
                stack.append(((parent, *path), seen | {parent}))

        return tuple(sorted(complete, key=lambda p: (-len(p), tuple(str(n.id) for n in p))))

    def lineage(self, node_id: str) -> tuple[ExperimentNode, ...]:
        """Return the node's ancestry closure in topological order, roots first.

        Every node this one derives from, plus the node itself -- the lineage
        *subgraph* flattened, rather than any single path through it. Use
        :meth:`lineage_paths` when the individual derivations matter.

        The guarantee is topological, not by distance:

        ```text
        for every edge X -> Y within the closure, X appears before Y
        ```

        An earlier version reversed :meth:`ancestors`, which orders by *minimum*
        distance from the target. Those differ whenever a shortcut edge exists:

        ```text
        A -> B -> N   and   A -> N
        ```

        gives A and B the same minimum distance from N, so reversing could
        place B before its own ancestor A. Ties are broken by creation time
        then id, so the order is stable across processes.
        """
        nodes, parents = self._closure(node_id)
        if node_id not in nodes:
            return ()

        # Kahn's algorithm. A node is ready once every parent inside the
        # closure has been emitted; the ready set is kept sorted so equal
        # candidates come out in a defined order rather than a dict's.
        remaining = {
            child: {p for p in parent_ids if p in nodes} for child, parent_ids in parents.items()
        }
        for node_key in nodes:
            remaining.setdefault(node_key, set())

        emitted: list[ExperimentNode] = []
        ready = sorted(
            (key for key, waiting in remaining.items() if not waiting),
            key=lambda key: (nodes[key].created_at, key),
        )
        while ready:
            key = ready.pop(0)
            emitted.append(nodes[key])
            del remaining[key]
            freed = []
            for child, waiting in remaining.items():
                if key in waiting:
                    waiting.discard(key)
                    if not waiting:
                        freed.append(child)
            for child in freed:
                ready.append(child)
            ready.sort(key=lambda key: (nodes[key].created_at, key))

        return tuple(emitted)

    def is_descendant_of(self, node_id: str, ancestor_id: str) -> bool:
        """Whether *node_id* derives from *ancestor_id*, at any depth."""
        return any(str(n.id) == ancestor_id for n in self.ancestors(node_id))

    # ---- comparison ------------------------------------------------------

    def compare(self, left_id: str, right_id: str) -> CandidateComparison:
        """Return what two candidates share and where they diverge.

        Symmetric: ``compare(a, b)`` and ``compare(b, a)`` report the same
        shared ancestry. An earlier version ordered the result by distance from
        the left node alone, which made the "nearest" answer depend on argument
        order in any DAG where two shared ancestors sit at different depths.

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

        left_closure = self._ancestry_closure(left)
        right_closure = self._ancestry_closure(right)
        common_ids = set(left_closure) & set(right_closure)
        common = tuple(left_closure[node_id] for node_id in sorted(common_ids))

        # Lowest: a common ancestor with no other common ancestor below it.
        # "Below" is the descendant direction, so an ancestor that has another
        # common ancestor among its descendants is not the lowest.
        lowest = tuple(
            node
            for node in common
            if not ({str(d.id) for d in self.descendants(str(node.id))} & common_ids)
        )

        return CandidateComparison(
            left=left,
            right=right,
            common_ancestors=common,
            nearest_common_ancestors=lowest,
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
        * **The same parent twice.** The payload would hold it twice and the
          edge table once, so the two records of one lineage would disagree.

        Raises:
            LineageError: With the specific parent and the reason.
        """
        if len(set(node.parent_ids)) != len(node.parent_ids):
            # The payload would record the parent twice while the edge table --
            # keyed on (parent_id, child_id) -- records it once, leaving two
            # representations of one lineage that disagree.
            raise LineageError(
                f"node {node.id} names a parent more than once: {[str(p) for p in node.parent_ids]}"
            )

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

    def _closure(
        self, node_id: str
    ) -> tuple[dict[str, ExperimentNode], dict[str, tuple[str, ...]]]:
        """Load a node's ancestry closure and its internal edges.

        Two queries regardless of depth. Reading the nodes and then asking for
        each one's parents would be a query per ancestor -- the per-generation
        round trip this module uses recursive CTEs to avoid, reintroduced one
        level up.

        Returns:
            The closure's nodes keyed by id (including *node_id* itself), and
            each node's parents **within the closure**.
        """
        walk = (
            "WITH RECURSIVE closure(id) AS ("
            "  SELECT ?"
            "  UNION"
            "  SELECT e.parent_id FROM experiment_edges e "
            "  JOIN closure ON e.child_id = closure.id"
            ")"
        )
        node_rows = self._connection.execute(
            f"{walk} SELECT n.payload_json FROM experiment_nodes n "
            f"WHERE n.id IN (SELECT id FROM closure)",
            (node_id,),
        ).fetchall()
        nodes = {
            str(n.id): n
            for n in (ExperimentNode.model_validate_json(row["payload_json"]) for row in node_rows)
        }

        edge_rows = self._connection.execute(
            f"{walk} SELECT e.parent_id, e.child_id FROM experiment_edges e "
            f"WHERE e.child_id IN (SELECT id FROM closure) "
            f"ORDER BY e.child_id, e.parent_id",
            (node_id,),
        ).fetchall()

        parents: dict[str, list[str]] = {}
        for row in edge_rows:
            if str(row["parent_id"]) in nodes:
                parents.setdefault(str(row["child_id"]), []).append(str(row["parent_id"]))

        return nodes, {child: tuple(ps) for child, ps in parents.items()}

    def _ancestry_closure(self, node: ExperimentNode) -> dict[str, ExperimentNode]:
        """The node's ancestry including itself, for comparison.

        A node is its own ancestor here. Comparing a candidate with the one it
        was derived from is the common case, and strict ancestry reports those
        two as sharing nothing.
        """
        return self._closure(str(node.id))[0]

    def _node(self, node_id: str) -> ExperimentNode | None:
        row = self._connection.execute(
            "SELECT payload_json FROM experiment_nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return None if row is None else ExperimentNode.model_validate_json(row["payload_json"])

    def _nodes(self, sql: str, params: tuple[str, ...]) -> tuple[ExperimentNode, ...]:
        rows = self._connection.execute(sql, params).fetchall()
        return tuple(ExperimentNode.model_validate_json(row["payload_json"]) for row in rows)
