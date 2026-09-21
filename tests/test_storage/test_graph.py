"""Experiment graph: lineage, traversal, comparison and cycle prevention."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from xaytune.core import Actor
from xaytune.storage import ControlPlaneRepository, LineageError

from .conftest import make_experiment, make_node

ACTOR = Actor(type="system", id="controller")


@pytest.fixture
def repo(connection: sqlite3.Connection) -> ControlPlaneRepository:
    return ControlPlaneRepository(connection)


@pytest.fixture
def experiment(repo: ControlPlaneRepository) -> Any:
    return repo.create_experiment(make_experiment(), actor=ACTOR)


def _node(repo: ControlPlaneRepository, experiment: Any, name: str, *parents: Any) -> Any:
    return repo.create_node(
        make_node(experiment, fingerprint=f"sha256:{name}", parents=tuple(p.id for p in parents)),
        actor=ACTOR,
    )


@pytest.fixture
def chain(repo: ControlPlaneRepository, experiment: Any) -> dict[str, Any]:
    """a -> b -> c, with d branching off b.

    ```text
    a ── b ── c
         └─── d
    ```
    """
    a = _node(repo, experiment, "a")
    b = _node(repo, experiment, "b", a)
    c = _node(repo, experiment, "c", b)
    d = _node(repo, experiment, "d", b)
    return {"a": a, "b": b, "c": c, "d": d}


# ---- direct relations ----------------------------------------------------


def test_parents_and_children(repo: ControlPlaneRepository, chain: dict[str, Any]) -> None:
    graph = repo.graph

    assert [n.id for n in graph.parents(str(chain["c"].id))] == [chain["b"].id]
    assert [n.id for n in graph.children(str(chain["b"].id))] == [
        chain["c"].id,
        chain["d"].id,
    ]
    assert graph.parents(str(chain["a"].id)) == ()


def test_roots_and_leaves(
    repo: ControlPlaneRepository, experiment: Any, chain: dict[str, Any]
) -> None:
    """Several roots are normal: an experiment may start from independent ideas."""
    other_root = _node(repo, experiment, "independent")
    graph = repo.graph

    assert {n.id for n in graph.roots(str(experiment.id))} == {
        chain["a"].id,
        other_root.id,
    }
    assert {n.id for n in graph.leaves(str(experiment.id))} == {
        chain["c"].id,
        chain["d"].id,
        other_root.id,
    }


# ---- transitive relations ------------------------------------------------


def test_ancestors_are_ordered_nearest_first(
    repo: ControlPlaneRepository, chain: dict[str, Any]
) -> None:
    """The first entry is what a result should be compared against."""
    assert [n.id for n in repo.graph.ancestors(str(chain["c"].id))] == [
        chain["b"].id,
        chain["a"].id,
    ]


def test_descendants_reach_every_branch(
    repo: ControlPlaneRepository, chain: dict[str, Any]
) -> None:
    assert {n.id for n in repo.graph.descendants(str(chain["a"].id))} == {
        chain["b"].id,
        chain["c"].id,
        chain["d"].id,
    }
    assert repo.graph.descendants(str(chain["c"].id)) == ()


def test_lineage_runs_root_first(repo: ControlPlaneRepository, chain: dict[str, Any]) -> None:
    assert [n.id for n in repo.graph.lineage(str(chain["c"].id))] == [
        chain["a"].id,
        chain["b"].id,
        chain["c"].id,
    ]


def test_lineage_of_a_merged_candidate_takes_the_longest_path(
    repo: ControlPlaneRepository, experiment: Any, chain: dict[str, Any]
) -> None:
    """A candidate derived from two predecessors has more than one path back.

    The longest is the full derivation story rather than a shortcut through it.
    """
    shallow = _node(repo, experiment, "shallow")
    merged = _node(repo, experiment, "merged", chain["c"], shallow)

    path = [n.id for n in repo.graph.lineage(str(merged.id))]
    assert path == [chain["a"].id, chain["b"].id, chain["c"].id, merged.id]


def test_is_descendant_of(repo: ControlPlaneRepository, chain: dict[str, Any]) -> None:
    graph = repo.graph
    assert graph.is_descendant_of(str(chain["c"].id), str(chain["a"].id))
    assert not graph.is_descendant_of(str(chain["a"].id), str(chain["c"].id))
    assert not graph.is_descendant_of(str(chain["c"].id), str(chain["d"].id))


def test_a_diamond_reports_each_ancestor_once(
    repo: ControlPlaneRepository, experiment: Any
) -> None:
    """Two paths to the same ancestor is one ancestor, not two."""
    root = _node(repo, experiment, "root")
    left = _node(repo, experiment, "left", root)
    right = _node(repo, experiment, "right", root)
    bottom = _node(repo, experiment, "bottom", left, right)

    ancestors = [n.id for n in repo.graph.ancestors(str(bottom.id))]
    assert ancestors.count(root.id) == 1
    assert set(ancestors) == {left.id, right.id, root.id}
    # Nearest first: both parents precede the shared grandparent.
    assert ancestors[-1] == root.id


# ---- comparison ----------------------------------------------------------


def test_siblings_share_their_nearest_ancestor(
    repo: ControlPlaneRepository, chain: dict[str, Any]
) -> None:
    comparison = repo.graph.compare(str(chain["c"].id), str(chain["d"].id))

    assert comparison.nearest_common_ancestor is not None
    assert comparison.nearest_common_ancestor.id == chain["b"].id
    assert comparison.is_comparable


def test_independent_roots_share_nothing(
    repo: ControlPlaneRepository, experiment: Any, chain: dict[str, Any]
) -> None:
    """Which is what makes them independent evidence."""
    other = _node(repo, experiment, "independent")

    comparison = repo.graph.compare(str(chain["c"].id), str(other.id))

    assert comparison.common_ancestors == ()
    assert comparison.nearest_common_ancestor is None


def test_two_nodes_with_one_fingerprint_are_replicates_not_alternatives(
    repo: ControlPlaneRepository, experiment: Any, chain: dict[str, Any]
) -> None:
    """Treating them as competing candidates would count one hypothesis twice."""
    twin = repo.create_node(
        make_node(
            experiment, fingerprint=chain["c"].training_fingerprint, parents=(chain["b"].id,)
        ),
        actor=ACTOR,
    )

    comparison = repo.graph.compare(str(chain["c"].id), str(twin.id))

    assert comparison.same_candidate
    assert not comparison.is_comparable


def test_comparing_across_experiments_is_refused(
    repo: ControlPlaneRepository, chain: dict[str, Any]
) -> None:
    other_experiment = repo.create_experiment(make_experiment("other"), actor=ACTOR)
    elsewhere = _node(repo, other_experiment, "elsewhere")

    with pytest.raises(LineageError, match="different"):
        repo.graph.compare(str(chain["c"].id), str(elsewhere.id))


# ---- lineage validation --------------------------------------------------


def test_a_parent_in_another_experiment_is_refused(
    repo: ControlPlaneRepository, experiment: Any, chain: dict[str, Any]
) -> None:
    """Nothing in the schema forbids it, and it makes ancestry span experiments.

    "Compare against the previous candidate" could then reach into work that
    answered a different question entirely.
    """
    other_experiment = repo.create_experiment(make_experiment("other"), actor=ACTOR)
    cross = make_node(other_experiment, parents=(chain["a"].id,))

    with pytest.raises(LineageError, match="may not span experiments"):
        repo.create_node(cross, actor=ACTOR)

    assert repo.aggregates.get_node(str(cross.id)) is None


def test_a_parent_that_does_not_exist_is_refused(
    repo: ControlPlaneRepository, experiment: Any
) -> None:
    from xaytune.core.ids import ExperimentNodeId

    orphan = make_node(experiment, parents=(ExperimentNodeId.generate(),))

    with pytest.raises(LineageError, match="does not exist"):
        repo.create_node(orphan, actor=ACTOR)


def test_a_node_cannot_be_its_own_parent(repo: ControlPlaneRepository, experiment: Any) -> None:
    node = make_node(experiment)
    self_parented = type(node).model_validate(
        {**node.model_dump(mode="python"), "parent_ids": (node.id,)}
    )

    with pytest.raises(LineageError, match="its own parent"):
        repo.create_node(self_parented, actor=ACTOR)


def test_an_edge_that_would_close_a_cycle_is_refused(
    repo: ControlPlaneRepository, experiment: Any, chain: dict[str, Any]
) -> None:
    """A node in a cycle has no derivation, so neither does anything after it.

    Constructed by hand because the public creation path cannot reach it: a new
    node has no descendants yet. This guards the invariant against a future
    path that adds edges after creation.
    """
    ancestor = chain["a"]
    would_cycle = type(ancestor).model_validate(
        {**ancestor.model_dump(mode="python"), "parent_ids": (chain["c"].id,)}
    )

    with pytest.raises(LineageError, match="close a cycle"):
        repo.graph.validate_parents(would_cycle)


def test_a_rejected_node_leaves_no_edges(
    repo: ControlPlaneRepository,
    connection: sqlite3.Connection,
    experiment: Any,
    chain: dict[str, Any],
) -> None:
    """Validation runs before anything is written, inside the transaction."""
    other_experiment = repo.create_experiment(make_experiment("other"), actor=ACTOR)
    cross = make_node(other_experiment, parents=(chain["a"].id,))

    with pytest.raises(LineageError):
        repo.create_node(cross, actor=ACTOR)

    edges = connection.execute(
        "SELECT COUNT(*) AS n FROM experiment_edges WHERE child_id = ?", (str(cross.id),)
    ).fetchone()
    assert edges["n"] == 0


# ---- a retry is never a branch -------------------------------------------


def test_attempts_never_appear_in_the_graph(
    repo: ControlPlaneRepository, experiment: Any, chain: dict[str, Any]
) -> None:
    """ADR-003: retries and preemptions create attempts, never nodes.

    If a retry ever showed up here as a branch, the comparability rule would
    have been broken upstream.
    """
    from .conftest import make_attempt, make_run

    run = repo.create_run(make_run(chain["b"]), actor=ACTOR)
    for number in (1, 2, 3):
        repo.create_attempt_with_submit_intent(
            make_attempt(run, attempt_number=number),
            request_digest=f"sha256:attempt-{number}",
            actor=ACTOR,
        )

    assert [n.id for n in repo.graph.children(str(chain["b"].id))] == [
        chain["c"].id,
        chain["d"].id,
    ]
    assert len(repo.aggregates.attempts_for_run(str(run.id))) == 3
