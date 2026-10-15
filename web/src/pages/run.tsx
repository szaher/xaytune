import { Link, useParams } from "@tanstack/react-router";
import type { Schemas } from "../api/client";
import { useAttempt, useRun } from "../api/queries";
import { DocumentTree } from "../components/document";
import { Card, Facts, Loading, Page, Section } from "../components/layout";
import { Status } from "../components/status";
import { Absent, Num, ShortId, Time } from "../components/values";

/** J4: a run's attempts in order, with what happened to each (chapter 24 §4.5). */
export function RunPage() {
  const { runId } = useParams({ from: "/runs/$runId" });
  const run = useRun(runId);
  return (
    <Loading query={run}>
      {() => {
        const data = run.data as Schemas["RunDetail"];
        const { run: detail } = data;
        return (
          <Page
            crumbs={
              <span className="flex gap-1">
                <Link
                  to="/experiments/$experimentId"
                  params={{ experimentId: detail.experiment_id }}
                  search={{ tab: "runs" }}
                  className="hover:underline"
                >
                  Experiment
                </Link>
                ›
                <Link
                  to="/experiments/$experimentId/candidates/$nodeId"
                  params={{ experimentId: detail.experiment_id, nodeId: detail.node_id }}
                  className="hover:underline"
                >
                  Candidate
                </Link>
                › Run
              </span>
            }
            title={
              <span className="flex items-center gap-2">
                Run <ShortId value={detail.id} keep={16} />
              </span>
            }
            subtitle={
              <span className="flex flex-wrap items-center gap-3">
                <Status value={detail.status} />
                <span>
                  seed <Num value={detail.seed} reason="no seed" />
                  {detail.seed_origin ? " (inherited)" : ""}
                </span>
                <span>{detail.attempts.length} attempt(s)</span>
              </span>
            }
          >
            <Section title="Attempts">
              <ol className="space-y-4" aria-label="Attempts">
                {detail.attempts.map((attempt) => (
                  <li key={attempt.id}>
                    <AttemptCard attempt={attempt} />
                  </li>
                ))}
              </ol>
            </Section>
            <Section title="Checkpoints">
              {data.checkpoints.length ? (
                <div className="grid gap-4 lg:grid-cols-2">
                  {data.checkpoints.map((checkpoint, index) => (
                    // biome-ignore lint/suspicious/noArrayIndexKey: checkpoints have no id on the wire
                    <Card key={index}>
                      <p className="mb-2 text-sm text-gray-600">
                        <Time value={checkpoint.created_at} />
                      </p>
                      <DocumentTree value={checkpoint.payload} label="Checkpoint" />
                    </Card>
                  ))}
                </div>
              ) : (
                <Absent reason="no checkpoints recorded" />
              )}
            </Section>
            {data.interventions.length ? (
              <Section title="Interventions">
                <div className="grid gap-4 lg:grid-cols-2">
                  {data.interventions.map((intervention, index) => (
                    // biome-ignore lint/suspicious/noArrayIndexKey: recorded documents only
                    <Card key={index}>
                      <DocumentTree value={intervention} label="Intervention" />
                    </Card>
                  ))}
                </div>
              </Section>
            ) : null}
          </Page>
        );
      }}
    </Loading>
  );
}

function AttemptCard({ attempt }: { attempt: Schemas["AttemptOut"] }) {
  const detail = useAttempt(attempt.id);
  return (
    <Card>
      <div
        role="group"
        className="space-y-3 text-sm"
        aria-label={`attempt ${attempt.attempt_number}`}
      >
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-semibold">Attempt #{attempt.attempt_number}</span>
          <Status value={attempt.status} />
          <span className="text-gray-500">
            <Time value={attempt.started_at} /> → <Time value={attempt.ended_at} />
          </span>
          <ShortId value={attempt.id} />
        </div>
        <Facts
          items={[
            [
              "Execution fingerprint",
              attempt.execution_fingerprint ? (
                <ShortId key="e" value={attempt.execution_fingerprint} keep={14} />
              ) : (
                <Absent key="e" reason="not compiled" />
              ),
            ],
            [
              "Overrides",
              attempt.execution_overrides.length ? (
                <DocumentTree key="o" value={attempt.execution_overrides} label="Overrides" />
              ) : (
                <Absent key="o" reason="none" />
              ),
            ],
            [
              "Restored from",
              attempt.checkpoint_ref ? (
                <DocumentTree key="c" value={attempt.checkpoint_ref} label="Checkpoint reference" />
              ) : (
                <Absent key="c" reason="started fresh" />
              ),
            ],
          ]}
        />
        <Loading query={detail}>
          {() => {
            const data = detail.data as Schemas["AttemptDetail"];
            return data.incidents.length ? (
              <div role="group" className="space-y-2" aria-label="Incidents">
                {data.incidents.map((incident) => (
                  <div key={incident.id} className="rounded border border-red-200 bg-red-50 p-2">
                    <p className="font-medium text-red-800">⚠ {incident.category}</p>
                    <DocumentTree value={incident.evidence} label="Evidence" />
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500">No incidents.</p>
            );
          }}
        </Loading>
      </div>
    </Card>
  );
}
