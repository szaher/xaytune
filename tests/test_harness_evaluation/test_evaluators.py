"""ADR-022 V4–V7: the two evaluators, the local driver, disclosure, metrics and variance sources."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import xaytune.evaluation.harness as harness_module
from tests.test_harness.fixtures import harness
from tests.test_harness_evaluation import fixtures as fx
from xaytune.agent import AgentModelIdentity, ScriptedAgentModel
from xaytune.core.agent_execution import HarnessRunResult, TerminationReason
from xaytune.core.domain.benchmark import FirstTasks
from xaytune.core.harness_evaluation import (
    ABSENCE_REASONS,
    METRICS,
    SUCCESSFUL_TASKS,
    HarnessRunEvidence,
    HarnessRunEvidenceBody,
    ReferenceEgressGrant,
    SuccessRule,
    execution_observations,
    run_evidence_digest,
    success_observation,
)
from xaytune.core.immutable import FrozenDict, thaw
from xaytune.core.refs import Actor
from xaytune.core.state.status import EvaluationAttemptStatus, EvaluationRunStatus
from xaytune.evaluation import EvaluationContext, UnsupportedEvaluationError
from xaytune.evaluation.harness import (
    EVIDENCE_FILE,
    EvaluationRefusedError,
    HarnessJudgeEvaluator,
    HarnessOutcomeEvaluator,
    bind_evaluation,
    provider_feedback,
    run_harness_evaluation,
)
from xaytune.trajectory import SecretDetectedError

SHIPPING = "support/shipping-001"


class Cancelled:
    def is_set(self) -> bool:
        return True


def _outcome(replicate: fx.Replicate, **config: Any) -> Any:
    evaluator = HarnessOutcomeEvaluator(replicate.source)
    spec = bind_evaluation(evaluator, fx.evaluation_spec(fx.outcome_config(**config)))
    return evaluator, spec


def _evaluate(
    replicate: fx.Replicate,
    out: Path,
    *,
    config: dict[str, Any] | None = None,
    disclosure: str | None = None,
    seed: int | None = 0,
) -> Any:
    evaluator = HarnessOutcomeEvaluator(replicate.source)
    metadata = {"disclosure": disclosure} if disclosure else {}
    spec = bind_evaluation(evaluator, fx.evaluation_spec(config or fx.outcome_config(), **metadata))
    return run_harness_evaluation(
        evaluator,
        spec,
        replicate.subject,
        source=replicate.source,
        output_uri=str(out),
        seed=seed,
        replicate=0,
    )


def _judge(
    replicate: fx.Replicate,
    out: Path,
    model: Any,
    *,
    grants: tuple[ReferenceEgressGrant, ...] = (),
    seed: int = 0,
    disclosure: str | None = None,
    **config: Any,
) -> Any:
    evaluator = HarnessJudgeEvaluator(replicate.source, grants=grants)
    metadata = {"disclosure": disclosure} if disclosure else {}
    spec = bind_evaluation(evaluator, fx.evaluation_spec(fx.judge_config(**config), **metadata))
    return run_harness_evaluation(
        evaluator,
        spec,
        replicate.subject,
        source=replicate.source,
        output_uri=str(out),
        seed=seed,
        replicate=0,
        judge_model=model,
    )


def _metric(evaluation: Any, name: str, slice: str | None = None) -> Any:
    assert evaluation.result is not None, evaluation.report.failure
    found = [m for m in evaluation.result.metrics if m.name == name and m.slice == slice]
    return found[0] if found else None


@pytest.fixture
def replicate(tmp_path: Path) -> fx.Replicate:
    return fx.Replicate(tmp_path / "replicate")


# ---- V4: the protocol --------------------------------------------------------------------------


class TestEvaluatorProtocol:
    def test_outcome_supports_resolves_and_supports_again(self, replicate: fx.Replicate) -> None:
        evaluator = HarnessOutcomeEvaluator(replicate.source)
        declared = fx.evaluation_spec(
            fx.outcome_config(suite={"suite_id": "support/core", "revision": "stable"})
        )
        assert evaluator.supports(declared)
        bound = bind_evaluation(evaluator, declared)
        assert bound.evaluator.config["suite"]["suite_fingerprint"] == (
            replicate.suite.ref.suite_fingerprint
        )
        assert bound.evaluator.version == evaluator.descriptor.plugin_version
        assert bound.evaluator.determinism.value == "deterministic"
        assert evaluator.supports(bound)

    def test_judge_is_stochastic_and_binds(self, replicate: fx.Replicate) -> None:
        evaluator = HarnessJudgeEvaluator(replicate.source)
        bound = bind_evaluation(evaluator, fx.evaluation_spec(fx.judge_config()))
        assert bound.evaluator.determinism.value == "stochastic"

    def test_prepare_is_deterministic_and_reads_nothing(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluator, spec = _outcome(replicate)
        absent = replicate.subject.model_copy(
            update={"uri": str(tmp_path / "nowhere" / EVIDENCE_FILE)}
        )
        context = EvaluationContext(
            experiment_id="e", node_id="n", evaluation_run_id="r", seed=1, output_uri="/tmp/x"
        )
        first = evaluator.prepare(absent, spec, context)
        assert first == evaluator.prepare(absent, spec, context)
        assert first.evaluation_fingerprint == spec.evaluation_fingerprint()

    @pytest.mark.parametrize(
        ("change", "needle"),
        [
            ({"timing_environment": "drop"}, "timing_environment"),
            ({"extra": 1}, "extra"),
            ({"success": {"kind": "exact-match"}}, "normalization"),
        ],
    )
    def test_g17_a_non_canonical_config(
        self, replicate: fx.Replicate, change: dict[str, Any], needle: str
    ) -> None:
        config = fx.outcome_config()
        for key, value in change.items():
            if value == "drop":
                del config[key]
            else:
                config[key] = value
        support = HarnessOutcomeEvaluator(replicate.source).supports(fx.evaluation_spec(config))
        assert not support
        assert any(needle in reason for reason in support.reasons), support.reasons

    def test_g17_a_config_that_parses_but_is_not_canonical(self, replicate: fx.Replicate) -> None:
        config = fx.judge_config()
        config["judge_model"]["generation"]["temperature"] = 0  # an int; canonical is 0.0
        evaluator = HarnessJudgeEvaluator(replicate.source)
        support = evaluator.supports(fx.evaluation_spec(config))
        assert not support
        assert any("canonical form" in reason for reason in support.reasons)
        assert evaluator.supports(fx.evaluation_spec(fx.judge_config()))

    def test_exact_match_on_a_task_without_a_reference(self, tmp_path: Path) -> None:
        members = tuple(
            fx.task(i, reference=None) if i == SHIPPING else fx.task(i) for i in fx.TASK_IDS
        )
        suite = fx.suite(members)
        evaluator = HarnessOutcomeEvaluator(fx.source(members))
        config = fx.outcome_config(suite=suite.ref.model_dump(mode="json"))
        support = evaluator.supports(fx.evaluation_spec(config))
        assert not support
        assert any("reference" in reason for reason in support.reasons)

    def test_an_unpinned_judge_model(self, replicate: fx.Replicate) -> None:
        config = fx.judge_config()
        config["judge_model"] = {
            **config["judge_model"],
            "revision": None,
            "artifact_digest": "sha256:" + "a" * 64,
        }
        support = HarnessJudgeEvaluator(replicate.source).supports(fx.evaluation_spec(config))
        assert not support
        assert any("not pinned" in reason for reason in support.reasons)

    def test_a_moving_alias_is_not_a_pin(self, replicate: fx.Replicate) -> None:
        config = fx.judge_config()
        config["judge_model"] = {**config["judge_model"], "revision": "latest"}
        assert not HarnessJudgeEvaluator(replicate.source).supports(fx.evaluation_spec(config))

    def test_prepare_refuses_another_subject_kind(self, replicate: fx.Replicate) -> None:
        evaluator, spec = _outcome(replicate)
        model = replicate.subject.model_copy(update={"kind": "model"})
        context = EvaluationContext(
            experiment_id="e", node_id="n", evaluation_run_id="r", output_uri="/tmp/x"
        )
        with pytest.raises(UnsupportedEvaluationError, match="agent_run_evidence"):
            evaluator.prepare(model, spec, context)

    def test_prepare_refuses_an_unbound_spec(self, replicate: fx.Replicate) -> None:
        evaluator = HarnessOutcomeEvaluator(replicate.source)
        context = EvaluationContext(
            experiment_id="e", node_id="n", evaluation_run_id="r", output_uri="/tmp/x"
        )
        with pytest.raises(UnsupportedEvaluationError, match="bind_evaluation"):
            evaluator.prepare(replicate.subject, fx.evaluation_spec(fx.outcome_config()), context)


# ---- the driver, and verification on every run (AC2, evaluator side) ------------------------


class TestLocalDriver:
    def test_a_run_walks_the_existing_lifecycle(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval")
        assert evaluation.run.status is EvaluationRunStatus.SUCCEEDED
        assert evaluation.attempt.status is EvaluationAttemptStatus.SUCCEEDED
        result = evaluation.result
        assert result is not None
        assert result.evaluation_run_id == evaluation.run.id
        assert result.subject == replicate.subject
        assert result.artifacts == (evaluation.report_ref,)
        assert evaluation.report_ref.kind == "evaluation_report"

    @pytest.mark.parametrize("which", ["outcome", "judge"])
    def test_every_run_verifies_its_subject_again(
        self,
        replicate: fx.Replicate,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        which: str,
    ) -> None:
        calls: list[str] = []
        original = harness_module.verify_run_evidence

        def spy(evidence: Any, *args: Any) -> Any:
            calls.append(evidence.evidence_digest)
            return original(evidence, *args)

        monkeypatch.setattr(harness_module, "verify_run_evidence", spy)
        for n in range(2):
            if which == "outcome":
                _evaluate(replicate, tmp_path / f"eval-{n}")
            else:
                _judge(replicate, tmp_path / f"eval-{n}", fx.judge_model(*[{"score": 4}] * 3))
        assert calls == [replicate.evidence.evidence_digest] * 2

    @pytest.mark.parametrize(
        "edit",
        ["retry-left-out", "attempt-added", "attested-to-absent", "another-seed", "another-cohort"],
    )
    def test_evidence_edited_after_assembly_is_refused_before_anything_is_measured(
        self, tmp_path: Path, edit: str
    ) -> None:
        def retried(spec: Any) -> None:
            with pytest.raises(SecretDetectedError):
                fx.run(spec, trajectory_secrets={"five days"})
            fx.run(spec)

        replicate = fx.Replicate(tmp_path / "replicate", per_task={"support/refund-001": retried})
        evidence = replicate.evidence
        entries = list(evidence.entries)
        fields = {n: getattr(evidence, n) for n in HarnessRunEvidenceBody.model_fields}
        if edit == "retry-left-out":
            first = entries[0]
            entries[0] = first.model_copy(
                update={
                    "attempts": tuple(a for a in first.attempts if a.completeness == "complete")
                }
            )
        elif edit == "attempt-added":
            extra = entries[1].attempts[0].model_copy(update={"trajectory_id": "att_x"})
            entries[1] = entries[1].model_copy(update={"attempts": (*entries[1].attempts, extra)})
        elif edit == "attested-to-absent":
            entries[1] = entries[1].model_copy(
                update={"state": "no-attested-result", "outcome": None}
            )
        elif edit == "another-seed":
            fields["seed"] = 99
        else:
            fields["assignment"] = evidence.assignment.model_copy(
                update={"cohort": "sha256:" + "3" * 64}
            )
        fields["entries"] = tuple(entries)
        body = HarnessRunEvidenceBody(**fields)
        forged = HarnessRunEvidence(**fields, evidence_digest=run_evidence_digest(body))
        location = Path(replicate.subject.uri)
        os.chmod(location, 0o600)
        location.write_text(forged.model_dump_json())
        subject = replicate.subject.model_copy(update={"digest": forged.evidence_digest})
        model = fx.judge_model({"score": 5})
        for run in (
            lambda: run_harness_evaluation(
                *_outcome(replicate),
                subject,
                source=replicate.source,
                output_uri=str(tmp_path / "o"),
                seed=0,
            ),
            lambda: run_harness_evaluation(
                HarnessJudgeEvaluator(replicate.source),
                bind_evaluation(
                    HarnessJudgeEvaluator(replicate.source),
                    fx.evaluation_spec(fx.judge_config()),
                ),
                subject,
                source=replicate.source,
                output_uri=str(tmp_path / "j"),
                seed=0,
                judge_model=model,
            ),
        ):
            evaluation = run()
            assert evaluation.result is None
            assert evaluation.run.status is EvaluationRunStatus.FAILED
            assert evaluation.report.tasks == () and evaluation.report.run_values == ()
        assert model.requests == (), "the judge is never asked about unverified evidence"

    def test_a_subject_whose_digest_is_not_the_evidence(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluator, spec = _outcome(replicate)
        subject = replicate.subject.model_copy(update={"digest": "sha256:" + "4" * 64})
        evaluation = run_harness_evaluation(
            evaluator, spec, subject, source=replicate.source, output_uri=str(tmp_path / "o")
        )
        assert evaluation.result is None
        assert "not the subject's digest" in (evaluation.report.failure or "")


# ---- V5: independence and disclosure --------------------------------------------------------


def _grant(replicate: fx.Replicate, **changes: Any) -> ReferenceEgressGrant:
    fields: dict[str, Any] = {
        "suite": replicate.suite.ref,
        "provider": fx.JUDGE.provider,
        "model": fx.JUDGE.name,
        "revision": fx.JUDGE.revision,
        "granted_by": Actor(type="human", id="owner"),
        "granted_at": "2026-10-13T00:00:00Z",
    }
    fields.update(changes)
    return ReferenceEgressGrant.model_validate(fields)


class TestIndependence:
    def test_g1_no_reference_reaches_the_execution_side_or_a_released_report(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval", disclosure="per-task")
        needles: list[bytes] = [fx.REFERENCES[SHIPPING].encode()]
        for task in replicate.suite.tasks:
            assert task.reference is not None
            needles.append(task.reference.uri.encode())
            # An answer equal to its reference has the reference's digest (ADR-021 F20).
            if fx.REFERENCES[task.id] != fx.ANSWER:
                needles.append(task.reference.digest.encode())
        released = [
            *replicate.store.root.rglob("*"),
            Path(evaluation.report_ref.uri),
        ]
        for path in released:
            if path.is_file():
                data = path.read_bytes()
                assert not any(n in data for n in needles), path
        for spec in replicate.specs:
            text = spec.model_dump_json().encode()
            assert not any(n in text for n in needles)

    def test_g2_a_judge_prompt_with_the_reference_stays_restricted(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        model = fx.judge_model(*[{"score": 5}] * 3)
        evaluation = _judge(
            replicate,
            tmp_path / "eval",
            model,
            grants=(_grant(replicate),),
            reference_access="reference",
        )
        reference = fx.REFERENCES[SHIPPING]
        assert any(reference in request.messages[0].content for request in model.requests)
        assert reference in Path(evaluation.restricted_ref.uri).read_text()
        assert reference not in Path(evaluation.report_ref.uri).read_text()
        assert reference not in evaluation.result.model_dump_json()
        assert evaluation.report.grant == _grant(replicate)

    @pytest.mark.parametrize(
        "grant",
        [
            None,
            {"model": "other-judge"},
            {"provider": "other-provider"},
            {"revision": "2026-09-01"},
            {"suite": "other"},
        ],
    )
    def test_g2a_no_matching_grant_and_nothing_is_sent(
        self, replicate: fx.Replicate, grant: Any
    ) -> None:
        grants: tuple[ReferenceEgressGrant, ...] = ()
        if grant is not None:
            if grant.get("suite") == "other":
                grant = {"suite": replicate.suite.ref.model_copy(update={"revision": "v9"})}
            grants = (_grant(replicate, **grant),)
        evaluator = HarnessJudgeEvaluator(replicate.source, grants=grants)
        spec = bind_evaluation(
            evaluator, fx.evaluation_spec(fx.judge_config(reference_access="reference"))
        )
        context = EvaluationContext(
            experiment_id="e", node_id="n", evaluation_run_id="r", output_uri="/tmp/x"
        )
        with pytest.raises(UnsupportedEvaluationError, match="ReferenceEgressGrant"):
            evaluator.prepare(replicate.subject, spec, context)

    def _prepared(
        self, replicate: fx.Replicate, out: Path, grant: ReferenceEgressGrant, **config: Any
    ) -> Any:
        evaluator = HarnessJudgeEvaluator(replicate.source, grants=(grant,))
        spec = bind_evaluation(evaluator, fx.evaluation_spec(fx.judge_config(**config)))
        context = EvaluationContext(
            experiment_id="e", node_id="n", evaluation_run_id="r", output_uri=str(out)
        )
        return evaluator.prepare(replicate.subject, spec, context)

    @staticmethod
    def _with_grant(execution: Any, grant: Any) -> Any:
        worker = thaw(execution.config)
        if grant is None:
            worker.pop("grant")
        else:
            worker["grant"] = grant
        return execution.model_copy(update={"config": FrozenDict(worker)})

    @pytest.mark.parametrize(
        ("change", "reason"),
        [
            ("absent", "carries no ReferenceEgressGrant"),
            ("another-revision", "does not name this suite and this pinned judge model"),
            ("another-suite", "does not name this suite and this pinned judge model"),
            ("another-grantor", "not one the host supplied"),
            ("host-supplied-none", "not one the host supplied"),
        ],
    )
    def test_g2a_execution_checks_the_grant_again_before_any_reference_is_read(
        self,
        replicate: fx.Replicate,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        change: str,
        reason: str,
    ) -> None:
        granted = _grant(replicate)
        execution = self._prepared(
            replicate, tmp_path / "eval", granted, reference_access="reference"
        )
        prepared = execution.config["grant"]
        host: tuple[ReferenceEgressGrant, ...] = (granted,)
        if change == "absent":
            execution = self._with_grant(execution, None)
        elif change == "another-revision":
            execution = self._with_grant(execution, {**prepared, "revision": "2026-09-01"})
        elif change == "another-suite":
            suite = replicate.suite.ref.model_copy(update={"revision": "v9"})
            execution = self._with_grant(
                execution, {**prepared, "suite": suite.model_dump(mode="json")}
            )
        elif change == "another-grantor":
            # Still names this suite and judge, but no host issued it.
            forged = {**prepared, "granted_by": {"type": "llm_agent", "id": "planner"}}
            execution = self._with_grant(execution, forged)
        else:
            host = ()

        def no_reference(*args: Any) -> Any:
            raise AssertionError("a reference was resolved before the grant was checked")

        monkeypatch.setattr(harness_module, "resolve_reference", no_reference)
        model = fx.judge_model(*[{"score": 5}] * 3)
        with pytest.raises(EvaluationRefusedError, match=reason):
            harness_module.execute_harness_evaluation(
                execution, source=replicate.source, judge_model=model, grants=host
            )
        assert model.requests == (), "nothing is sent"

    def test_an_execution_config_that_does_not_parse_is_refused(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        granted = _grant(replicate)
        execution = self._prepared(
            replicate, tmp_path / "eval", granted, reference_access="reference"
        )
        broken = self._with_grant(execution, {**execution.config["grant"], "revision": ""})
        model = fx.judge_model(*[{"score": 5}] * 3)
        with pytest.raises(EvaluationRefusedError, match="not one prepare"):
            harness_module.execute_harness_evaluation(
                broken, source=replicate.source, judge_model=model, grants=(granted,)
            )
        assert model.requests == ()

    def test_g2a_a_grant_on_an_evaluation_that_sends_no_reference(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluator = HarnessJudgeEvaluator(replicate.source)
        spec = bind_evaluation(evaluator, fx.evaluation_spec(fx.judge_config()))
        context = EvaluationContext(
            experiment_id="e", node_id="n", evaluation_run_id="r", output_uri=str(tmp_path / "e")
        )
        execution = evaluator.prepare(replicate.subject, spec, context)
        smuggled = self._with_grant(execution, _grant(replicate).model_dump(mode="json"))
        model = fx.judge_model(*[{"score": 5}] * 3)
        with pytest.raises(EvaluationRefusedError, match="sends no reference"):
            harness_module.execute_harness_evaluation(
                smuggled,
                source=replicate.source,
                judge_model=model,
                grants=(_grant(replicate),),
            )
        assert model.requests == ()

    def test_the_prepared_grant_executes_and_a_judge_without_the_reference_reads_none(
        self, replicate: fx.Replicate, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        granted = _grant(replicate)
        execution = self._prepared(
            replicate, tmp_path / "eval", granted, reference_access="reference"
        )
        model = fx.judge_model(*[{"score": 5}] * 3)
        harness_module.execute_harness_evaluation(
            execution, source=replicate.source, judge_model=model, grants=(granted,)
        )
        assert len(model.requests) == 3

        def no_reference(*args: Any) -> Any:
            raise AssertionError("a judge without the reference resolved one")

        monkeypatch.setattr(harness_module, "resolve_reference", no_reference)
        evaluation = _judge(replicate, tmp_path / "none", fx.judge_model(*[{"score": 4}] * 3))
        assert _metric(evaluation, "quality").value == 4.0

    def test_the_grant_is_outside_the_fingerprint_and_reference_access_inside(
        self, replicate: fx.Replicate
    ) -> None:
        none = fx.evaluation_spec(fx.judge_config()).evaluation_fingerprint()
        uses = fx.evaluation_spec(fx.judge_config(reference_access="reference"))
        assert none != uses.evaluation_fingerprint()
        aggregate = fx.evaluation_spec(fx.judge_config(), disclosure="aggregate")
        per_task = fx.evaluation_spec(fx.judge_config(), disclosure="per-task")
        assert aggregate.evaluation_fingerprint() == per_task.evaluation_fingerprint()

    def test_g2b_aggregate_disclosure(self, replicate: fx.Replicate, tmp_path: Path) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval")
        report = evaluation.report
        assert report.disclosure == "aggregate"
        for task in report.tasks:
            assert task.get("task_success") is None, "per-task scores stay restricted"
            assert task.get("turns") is not None, "execution measurements are released"
        restricted = evaluation.restricted
        assert all(t.get("task_success") is not None for t in restricted.tasks)
        assert _metric(evaluation, "task_success").sample_count == 3

    def test_g2b_a_run_value_over_one_task_is_withheld(self, tmp_path: Path) -> None:
        replicate = fx.Replicate(tmp_path / "replicate", sampling=FirstTasks(count=1))
        evaluation = _evaluate(
            replicate,
            tmp_path / "eval",
            config=fx.outcome_config(sampling={"kind": "first", "count": 1}),
        )
        assert _metric(evaluation, "task_success") is None
        assert "task_success" in evaluation.report.withheld
        assert _metric(evaluation, "turns") is not None

    def test_g2b_aggregate_withholds_every_successful_tasks_slice(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        """Three tasks, two succeed: a two-task slice next to per-task turns tells which."""
        aggregate = _evaluate(replicate, tmp_path / "aggregate")
        restricted = aggregate.restricted
        assert [t.get("task_success").value for t in restricted.tasks] == [1.0, 1.0, 0.0]
        report = aggregate.report
        assert all(v.slice is None for v in report.run_values)
        assert all(m.slice is None for m in aggregate.result.metrics)
        assert f"turns/{SUCCESSFUL_TASKS}" in report.withheld
        assert "task_success" not in report.withheld, "over three tasks it is released"
        assert _metric(aggregate, "task_success").sample_count == 3
        assert all(t.get("turns") is not None for t in report.tasks), "per-task turns stay"
        sliced = {
            v.metric: v.sample_count for v in restricted.run_values if v.slice == SUCCESSFUL_TASKS
        }
        assert sliced["turns"] == 2, "the restricted tier keeps every run value"
        assert set(sliced) == {w.split("/")[0] for w in report.withheld if "/" in w}
        assert SUCCESSFUL_TASKS not in json.dumps(
            json.loads(Path(aggregate.report_ref.uri).read_text())["run_values"]
        )

        per_task = _evaluate(replicate, tmp_path / "per-task", disclosure="per-task")
        assert per_task.report.withheld == ()
        assert _metric(per_task, "turns", SUCCESSFUL_TASKS).sample_count == 2
        assert per_task.result.metrics != aggregate.result.metrics
        assert per_task.report.evaluation_fingerprint == report.evaluation_fingerprint

    def test_per_task_disclosure_releases_per_task_scores(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval", disclosure="per-task")
        assert [t.get("task_success").value for t in evaluation.report.tasks] == [1.0, 1.0, 0.0]

    @pytest.mark.parametrize("disclosure", ["aggregate", "per-task"])
    def test_g2c_a_provider_never_receives_a_per_task_score(
        self, replicate: fx.Replicate, tmp_path: Path, disclosure: str
    ) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval", disclosure=disclosure)
        feedback = provider_feedback(evaluation.report)
        for task in feedback.tasks:
            assert task.get("task_success") is None and task.get("quality") is None
        text = feedback.model_dump_json()
        assert fx.REFERENCES[SHIPPING] not in text

    @pytest.mark.parametrize("disclosure", ["aggregate", "per-task"])
    def test_g2c_a_provider_never_receives_a_single_task_score_as_a_run_value(
        self, tmp_path: Path, disclosure: str
    ) -> None:
        replicate = fx.Replicate(tmp_path / "replicate", sampling=FirstTasks(count=1))
        one = {"kind": "first", "count": 1}
        evaluation = _evaluate(
            replicate,
            tmp_path / "eval",
            config=fx.outcome_config(sampling=one),
            disclosure=disclosure,
        )
        released = {(v.metric, v.slice) for v in evaluation.report.run_values}
        if disclosure == "per-task":
            assert ("task_success", None) in released, "people may see it under per-task"
            assert ("turns", SUCCESSFUL_TASKS) in released
        feedback = provider_feedback(evaluation.report)
        kept = {(v.metric, v.slice) for v in feedback.run_values}
        assert ("task_success", None) not in kept
        assert all(slice_ is None for _, slice_ in kept), "no successful-tasks slice"
        assert ("turns", None) in kept, "execution run values still reach the provider"
        if disclosure == "per-task":
            assert {"task_success", f"turns/{SUCCESSFUL_TASKS}"} <= set(feedback.withheld)

    @pytest.mark.parametrize("disclosure", ["aggregate", "per-task"])
    def test_g2c_a_provider_never_receives_a_single_task_quality(
        self, tmp_path: Path, disclosure: str
    ) -> None:
        replicate = fx.Replicate(tmp_path / "replicate", sampling=FirstTasks(count=1))
        evaluation = _judge(
            replicate,
            tmp_path / "eval",
            fx.judge_model({"score": 4}),
            disclosure=disclosure,
            sampling={"kind": "first", "count": 1},
        )
        if disclosure == "per-task":
            assert _metric(evaluation, "quality").value == 4.0
        feedback = provider_feedback(evaluation.report)
        assert feedback.run_values == ()
        assert "4.0" not in feedback.model_dump_json()

    def test_g2c_successful_task_slices_never_reach_a_provider(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        """With per-task turns, a slice's mean over the tasks that succeeded tells which did."""
        evaluation = _evaluate(replicate, tmp_path / "eval", disclosure="per-task")
        sliced = [v for v in evaluation.report.run_values if v.slice == SUCCESSFUL_TASKS]
        assert sliced and all(v.sample_count == 2 for v in sliced)
        feedback = provider_feedback(evaluation.report)
        assert all(v.slice is None for v in feedback.run_values)
        assert f"turns/{SUCCESSFUL_TASKS}" in feedback.withheld
        success = next(v for v in feedback.run_values if v.metric == "task_success")
        assert success.sample_count == 3, "an aggregate over many tasks still reaches it"

    def test_evaluation_opens_the_evidence_read_only(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        def tree() -> dict[str, tuple[int, int]]:
            return {
                str(p): (p.stat().st_mtime_ns, p.stat().st_size)
                for p in sorted(replicate.store.root.rglob("*"))
            }

        before = tree()
        _evaluate(replicate, tmp_path / "eval")
        _judge(replicate, tmp_path / "judge", fx.judge_model(*[{"score": 3}] * 3))
        assert tree() == before


# ---- V6: metrics --------------------------------------------------------------------------


class TestMetrics:
    def test_every_catalog_metric_on_known_evidence(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval", disclosure="per-task")
        task = evaluation.report.tasks[0]
        assert [o.metric for o in task.observations] == [m for m in METRICS if m != "quality"]
        values = {o.metric: o.value for o in task.observations if o.value is not None}
        absent = {o.metric: o.absent for o in task.observations if o.absent is not None}
        result = HarnessRunResult.model_validate_json(
            Path(replicate.specs[0].outputs[0].uri).read_bytes()
        )
        assert values["task_success"] == 1.0
        assert values["turns"] == result.turns == 2
        assert values["tool_calls"] == result.tool_calls == 1
        assert values["total_tokens"] == result.total_tokens
        assert values["retries"] == 0
        assert values["spend_total_tokens"] == result.total_tokens
        assert values["token_budget_utilization"] == result.total_tokens / 100000
        assert values["peak_prompt_tokens"] <= result.prompt_tokens
        assert "model_latency_ms" in values
        assert absent == {
            "cached_tokens": "not-recorded",
            "reasoning_tokens": "not-recorded",
            "cost": "not-recorded",
            "elapsed_ms": "not-recorded",
        }

    def test_metric_results_carry_unit_and_catalog(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval", disclosure="per-task")
        tokens = _metric(evaluation, "total_tokens")
        assert tokens.metadata["unit"] == "tokens(xaytune.whitespace-tokens/v1)"
        assert tokens.metadata["catalog"] == "xaytune.harness-metrics/v1"
        assert _metric(evaluation, "task_success").value == pytest.approx(2 / 3)
        assert _metric(evaluation, "turns", "successful-tasks").sample_count == 2

    def test_g6_cost_and_cached_tokens_are_absent_never_zero(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval")
        names = {m.name for m in evaluation.result.metrics}
        assert not names & {"cost", "cached_tokens", "reasoning_tokens", "elapsed_ms"}
        reasons = {(a.metric, a.reason) for a in evaluation.report.absences}
        assert ("cost", "not-recorded") in reasons and ("cached_tokens", "not-recorded") in reasons

    def test_g7_a_model_call_without_latency(self) -> None:
        call = SimpleNamespace(
            kind="model-call-finished",
            outcome="replied",
            latency_ms=None,
            prompt_tokens=3,
            output_tokens=2,
            parent=1,
            sequence=2,
        )
        found, _ = execution_observations(
            _result("final-answer"), (call,), ((None, (call,)),), None
        )
        latency = next(o for o in found if o.metric == "model_latency_ms")
        assert (latency.value, latency.absent) == (None, "not-recorded")

    def test_g8_a_retry_with_a_started_call_and_no_finish(self) -> None:
        started = SimpleNamespace(kind="model-call-started", sequence=1)
        finished = SimpleNamespace(
            kind="model-call-finished",
            outcome="replied",
            latency_ms=5,
            prompt_tokens=4,
            output_tokens=1,
            parent=1,
            sequence=2,
        )
        torn = SimpleNamespace(kind="model-call-started", sequence=7)
        found, lower_bound = execution_observations(
            _result("final-answer"),
            (started, finished),
            ((None, (started, finished)), ("writer-lost", (started, finished, torn))),
            None,
        )
        spend = next(o for o in found if o.metric == "spend_total_tokens")
        assert spend.absent == "spend-unaccounted"
        assert lower_bound == 10
        assert next(o for o in found if o.metric == "retries").value == 1

    def test_lost_evidence_leaves_spend_unaccounted(self) -> None:
        found, _ = execution_observations(None, (), (("post-commit-evidence-lost", ()),), None)
        spend = next(o for o in found if o.metric == "spend_total_tokens")
        assert spend.absent == "spend-unaccounted", "an empty prefix is not zero spend"

    def test_no_attested_result_is_absent_and_spend_is_kept(self, tmp_path: Path) -> None:
        def partial(spec: Any) -> None:
            with pytest.raises(SecretDetectedError):
                fx.run(spec, trajectory_secrets={"five days"})

        replicate = fx.Replicate(tmp_path / "replicate", per_task={"support/refund-001": partial})
        evaluation = _evaluate(replicate, tmp_path / "eval", disclosure="per-task")
        task = evaluation.report.tasks[0]
        assert task.get("task_success").absent == "no-attested-result"
        assert task.get("turns").absent == "no-attested-result"
        assert task.get("spend_total_tokens").value > 0
        assert _metric(evaluation, "task_success") is None, "no run value, no shrunken denominator"
        assert _metric(evaluation, "spend_total_tokens") is not None
        assert ("task_success", "no-attested-result") in {
            (a.metric, a.reason) for a in evaluation.report.absences
        }

    def test_g14_a_judge_answer_outside_the_scale_or_unparseable(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        model = fx.judge_model({"score": 9}, {"verdict": "good"}, {"score": 4})
        evaluation = _judge(replicate, tmp_path / "eval", model)
        observations = [t.get("quality") for t in evaluation.restricted.tasks]
        assert [o.absent for o in observations] == ["judge-failed", "judge-failed", None]
        assert observations[2].value == 4.0
        assert evaluation.result is None, "an absent task leaves no run value"
        assert ("quality", "judge-failed") in {
            (a.metric, a.reason) for a in evaluation.report.absences
        }

    def test_quality_is_absent_when_there_is_no_answer_to_judge(self, tmp_path: Path) -> None:
        stops = harness(
            stopping={
                "max_turns": 1,
                "max_total_tokens": 100000,
                "max_tool_calls": 30,
                "terminate_on": ["final-answer"],
            }
        )
        replicate = fx.Replicate(tmp_path / "replicate", harness_spec=stops)
        model = fx.judge_model(*[{"score": 5}] * 3)
        evaluation = _judge(replicate, tmp_path / "judge", model)
        assert model.requests == (), "nothing to judge, so nothing is asked"
        observations = [t.get("quality") for t in evaluation.restricted.tasks]
        assert all(o.value is None and o.absent == "not-applicable" for o in observations)
        assert evaluation.result is None, "no score is imputed, so there is no run value"
        assert ("quality", "not-applicable") in {
            (a.metric, a.reason) for a in evaluation.report.absences
        }
        outcome = _evaluate(replicate, tmp_path / "outcome")
        assert _metric(outcome, "task_success").value == 0.0, "task_success still scores 0"

    def test_judge_quality_unit_names_the_rubric(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        evaluation = _judge(replicate, tmp_path / "eval", fx.judge_model(*[{"score": 4}] * 3))
        quality = _metric(evaluation, "quality")
        assert quality.value == 4.0
        assert quality.metadata["unit"] == f"rubric:{fx.digest(fx.RUBRIC.encode())} points"

    def test_a_judge_offered_another_model_is_refused(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        other = ScriptedAgentModel(
            [{"score": 4}], model=AgentModelIdentity(provider="acme", name="judge")
        )
        evaluation = _judge(replicate, tmp_path / "eval", other)
        assert evaluation.result is None
        assert other.requests == ()

    def test_g18_every_task_cancelled(self, tmp_path: Path) -> None:
        def cancelled(spec: Any) -> None:
            fx.run(spec, cancel=Cancelled())

        replicate = fx.Replicate(
            tmp_path / "replicate", per_task={i: cancelled for i in fx.TASK_IDS}
        )
        evaluation = _evaluate(replicate, tmp_path / "eval")
        assert evaluation.result is None
        assert evaluation.run.status is EvaluationRunStatus.FAILED
        assert "cancelled" in (evaluation.report.failure or "")

    def test_no_reason_outside_the_catalog(self, replicate: fx.Replicate, tmp_path: Path) -> None:
        evaluation = _evaluate(replicate, tmp_path / "eval")
        assert {a.reason for a in evaluation.report.absences} <= ABSENCE_REASONS


# ---- V6: the termination mapping --------------------------------------------------------------


def _result(termination: TerminationReason, answer: str = fx.ANSWER) -> HarnessRunResult:
    completes = termination in ("final-answer", "no-tool-call")
    digest = "sha256:" + "5" * 64
    return HarnessRunResult(
        request_digest=digest,
        execution_fingerprint=digest,
        harness_fingerprint=digest,
        compiler={"name": "reference-harness", "version": "0.1.0"},
        runner={"name": "r", "version": "1"},
        model_digest=digest,
        task_id="t",
        seed=0,
        replicate=0,
        termination=termination,
        completed=completes,
        answer=answer if completes else None,
        error="ModelError" if termination == "model-error" else None,
        turns=1,
        tool_calls=0,
        refused_tool_calls=0,
        failed_tool_calls=0,
        context_truncations=0,
        prompt_tokens=3,
        output_tokens=2,
        total_tokens=5,
        token_counter="xaytune.whitespace-tokens/v1",
    )


RULE = SuccessRule(kind="exact-match", normalization="strip")


@pytest.mark.parametrize(
    ("termination", "expected"),
    [
        ("final-answer", 1.0),
        ("no-tool-call", 1.0),
        ("max-turns", 0.0),
        ("token-budget", 0.0),
        ("tool-call-budget", 0.0),
        ("context-overflow", 0.0),
        ("model-error", 0.0),
        ("cancelled", "unscored-cancelled"),
    ],
)
def test_each_termination_maps_as_v6_states(termination: TerminationReason, expected: Any) -> None:
    observation = success_observation(_result(termination), fx.ANSWER, RULE)
    if isinstance(expected, float):
        assert observation.value == expected
    else:
        assert observation.absent == expected


def test_a_completed_wrong_answer_fails_and_no_result_is_absent() -> None:
    assert success_observation(_result("final-answer", "nope"), fx.ANSWER, RULE).value == 0.0
    assert success_observation(None, fx.ANSWER, RULE).absent == "no-attested-result"


@pytest.mark.parametrize(
    ("normalization", "answer", "matches"),
    [
        ("none", " A ", False),
        ("strip", " A ", True),
        ("strip", "a", False),
        ("casefold-strip", " a ", True),
    ],
)
def test_normalizations(normalization: str, answer: str, matches: bool) -> None:
    rule = SuccessRule(kind="exact-match", normalization=normalization)  # type: ignore[arg-type]
    assert rule.matches(answer, "A") is matches


# ---- V7: variance sources ---------------------------------------------------------------------


class TestVarianceSources:
    def test_a_retry_changes_spend_and_never_the_outcome(self, tmp_path: Path) -> None:
        def retried(spec: Any) -> None:
            with pytest.raises(SecretDetectedError):
                fx.run(spec, trajectory_secrets={"five days"})
            fx.run(spec)

        plain = _evaluate(fx.Replicate(tmp_path / "a"), tmp_path / "ea", disclosure="per-task")
        with_retry = _evaluate(
            fx.Replicate(tmp_path / "b", per_task={"support/refund-001": retried}),
            tmp_path / "eb",
            disclosure="per-task",
        )
        first, retried_first = plain.report.tasks[0], with_retry.report.tasks[0]
        assert first.get("task_success") == retried_first.get("task_success")
        assert first.get("turns") == retried_first.get("turns")
        assert retried_first.get("retries").value == 1
        assert retried_first.get("spend_total_tokens").value > first.get("spend_total_tokens").value

    def test_g15_a_second_judging_replicate_is_a_new_run(
        self, replicate: fx.Replicate, tmp_path: Path
    ) -> None:
        model = fx.judge_model(*[{"score": 4}] * 3, *[{"score": 2}] * 3)
        first = _judge(replicate, tmp_path / "j0", model, seed=0)
        second = _judge(replicate, tmp_path / "j1", model, seed=1)
        assert len(model.requests) == 6, "the second sample was asked, not reused"
        assert first.run.id != second.run.id
        assert _metric(first, "quality").value == 4.0
        assert _metric(second, "quality").value == 2.0
        assert first.result.evaluation_fingerprint == second.result.evaluation_fingerprint


def test_reports_are_json_and_the_restricted_one_is_owner_only(
    replicate: fx.Replicate, tmp_path: Path
) -> None:
    evaluation = _evaluate(replicate, tmp_path / "eval")
    released = json.loads(Path(evaluation.report_ref.uri).read_text())
    assert released["schema_version"] == "xaytune.harness-evaluation-report/v1alpha1"
    restricted = Path(evaluation.restricted_ref.uri)
    assert restricted.parent.stat().st_mode & 0o077 == 0
    assert evaluation.restricted_ref.metadata["tier"] == "restricted"


def test_reports_are_never_written_inside_the_evidence(replicate: fx.Replicate) -> None:
    evaluator, spec = _outcome(replicate)
    inside = replicate.store.root / "reports"
    context = EvaluationContext(
        experiment_id="e", node_id="n", evaluation_run_id="r", output_uri=str(inside)
    )
    with pytest.raises(UnsupportedEvaluationError, match="read-only"):
        evaluator.prepare(replicate.subject, spec, context)
