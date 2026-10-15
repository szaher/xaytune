import { Link, useParams } from "@tanstack/react-router";
import type { Schemas } from "../api/client";
import { useEvaluationRun } from "../api/queries";
import { DocumentTree } from "../components/document";
import { Card, Facts, Loading, Page, Section } from "../components/layout";
import { Status } from "../components/status";
import { Absent, Num, ShortId, Time } from "../components/values";

/** J3: what an evaluation measured, under which evaluation, with uncertainty (chapter 24 §2). */
export function EvaluationRunPage() {
  const { evaluationRunId } = useParams({ from: "/evaluation-runs/$evaluationRunId" });
  const run = useEvaluationRun(evaluationRunId);
  return (
    <Loading query={run}>
      {() => {
        const data = run.data as Schemas["EvaluationRunDetail"];
        const { evaluation_run: evaluation } = data;
        return (
          <Page
            crumbs={
              <span className="flex gap-1">
                <Link
                  to="/experiments/$experimentId/candidates/$nodeId"
                  params={{ experimentId: evaluation.experiment_id, nodeId: evaluation.node_id }}
                  className="hover:underline"
                >
                  Candidate
                </Link>
                › Evaluation
              </span>
            }
            title={
              <span className="flex items-center gap-2">
                Evaluation <ShortId value={evaluation.id} keep={16} />
              </span>
            }
            subtitle={
              <span className="flex flex-wrap items-center gap-3">
                <Status value={evaluation.status} />
                <span>cycle {evaluation.evaluation_cycle}</span>
                <span>
                  evaluation <ShortId value={evaluation.evaluation_fingerprint} keep={14} />
                </span>
              </span>
            }
          >
            <Section title="Metrics">
              {evaluation.result ? (
                <table className="min-w-full text-sm" aria-label="Metrics">
                  <thead className="text-left text-xs uppercase text-gray-500">
                    <tr>
                      <th className="py-1 pr-4">Metric</th>
                      <th className="pr-4">Value</th>
                      <th className="pr-4">±SE</th>
                      <th className="pr-4">95% CI</th>
                      <th className="pr-4">Samples</th>
                      <th>Evaluator</th>
                    </tr>
                  </thead>
                  <tbody className="tabular-nums">
                    {evaluation.result.metrics.map((metric) => (
                      <tr key={`${metric.evaluator_name}/${metric.name}/${metric.slice ?? ""}`}>
                        <td className="py-1 pr-4">
                          {metric.name}
                          {metric.slice ? (
                            <span className="text-gray-500"> · {metric.slice}</span>
                          ) : null}
                        </td>
                        <td className="pr-4">
                          <Num value={metric.value} />
                        </td>
                        <td className="pr-4">
                          <Num value={metric.standard_error} reason="not reported" />
                        </td>
                        <td className="pr-4">
                          {metric.confidence_interval ? (
                            `[${metric.confidence_interval.join(", ")}]`
                          ) : (
                            <Absent reason="not reported" />
                          )}
                        </td>
                        <td className="pr-4">
                          <Num value={metric.sample_count} reason="not reported" />
                        </td>
                        <td>
                          {metric.evaluator_name} {metric.evaluator_version ?? ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <Absent reason="no result recorded" />
              )}
            </Section>
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <h3 className="mb-2 text-sm font-semibold">Evaluation spec</h3>
                <DocumentTree value={evaluation.spec} label="Evaluation spec" />
              </Card>
              <Card>
                <h3 className="mb-2 text-sm font-semibold">Subject and attempts</h3>
                <DocumentTree value={evaluation.subject} label="Subject" />
                <Facts
                  items={data.attempts.map((attempt) => [
                    `Attempt #${attempt.attempt_number}`,
                    <span key={attempt.id} className="flex items-center gap-2">
                      <Status value={attempt.status} /> <Time value={attempt.ended_at} />
                    </span>,
                  ])}
                />
              </Card>
            </div>
          </Page>
        );
      }}
    </Loading>
  );
}
