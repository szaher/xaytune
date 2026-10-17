import { Link, useParams } from "@tanstack/react-router";
import { useState } from "react";
import type { Schemas } from "../api/client";
import { useAttempt, useAttemptMetrics, useRun } from "../api/queries";
import { MetricChart } from "../components/chart";
import { DocumentTree } from "../components/document";
import { Card, Facts, Loading, Page, Section } from "../components/layout";
import {
  chartLines,
  type RecordedSeries,
  type SeriesChoice,
  seriesChoices,
} from "../components/metric-series";
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
            <Section title="Metrics">
              <RunMetrics attempts={detail.attempts} />
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
  const detail = useAttempt(attempt.id, attempt.ended_at !== null);
  return (
    <Card>
      <div
        role="group"
        className="space-y-3 text-sm"
        aria-label={`attempt ${attempt.attempt_number}`}
        data-attempt={attempt.id}
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

const PREFERRED = JSON.stringify(["training.loss", null]);

/**
 * The run's recorded metric series, one line per attempt, so a recovered
 * attempt is seen continuing (or restarting) where the failed one stopped.
 * The attempts come from the polled run, so one started while the page is
 * open is drawn as soon as it reports.
 */
function RunMetrics({ attempts }: { attempts: Schemas["AttemptOut"][] }) {
  const queries = useAttemptMetrics(
    attempts.map((attempt) => ({ id: attempt.id, ended: attempt.ended_at !== null })),
  );
  const [chosen, setChosen] = useState<string | undefined>(undefined);
  if (queries.every((query) => query.isPending)) {
    return <p className="text-sm text-gray-500">Loading metrics…</p>;
  }
  const failed = queries.find((query) => query.isError);
  if (failed) {
    return (
      <p className="text-sm text-red-700">Metrics could not be read: {String(failed.error)}</p>
    );
  }
  // An attempt whose series are still on their way has none yet; it is drawn when they arrive.
  const byAttempt: RecordedSeries[][] = queries.map(
    (query) => (query.data as Schemas["AttemptMetrics"] | undefined)?.series ?? [],
  );
  const choices = seriesChoices(byAttempt);
  if (choices.length === 0) {
    return <Absent reason="no metrics recorded for this run" />;
  }
  const choice = (choices.find((one) => one.key === chosen) ??
    choices.find((one) => one.key === PREFERRED) ??
    choices[0]) as SeriesChoice;
  const lines = chartLines(
    byAttempt,
    attempts.map((attempt) => attempt.attempt_number),
    choice,
  );
  const downsampled = byAttempt
    .flat()
    .some((one) => one.name === choice.name && one.unit === choice.unit && one.downsampled);
  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-sm">
        <span className="text-gray-600">Series</span>
        <select
          className="rounded border border-gray-300 px-2 py-1"
          value={choice.key}
          onChange={(event) => setChosen(event.target.value)}
        >
          {choices.map((option) => (
            <option key={option.key} value={option.key}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <MetricChart title={choice.name} unit={choice.unit} lines={lines} />
      {downsampled ? (
        <p className="text-xs text-gray-500">
          Downsampled for display: every bucket keeps its lowest and highest value.
        </p>
      ) : null}
    </div>
  );
}
