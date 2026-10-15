import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { createColumnHelper } from "@tanstack/react-table";
import type { Schemas } from "../api/client";
import { useExperiments } from "../api/queries";
import { Loading, Page } from "../components/layout";
import { Status } from "../components/status";
import { DataTable } from "../components/table";
import { Absent, ShortId, Text, Time } from "../components/values";

/** Statuses an experiment can be in (`ExperimentStatus`), for the filter. */
export const EXPERIMENT_STATUSES = [
  "created",
  "active",
  "paused",
  "succeeded",
  "failed",
  "cancelled",
  "budget_exhausted",
] as const;

const column = createColumnHelper<Schemas["ExperimentSummary"]>();

const COLUMNS = [
  column.accessor("name", {
    header: "Name",
    cell: (info) => (
      <Link
        to="/experiments/$experimentId"
        params={{ experimentId: info.row.original.id }}
        className="font-medium text-blue-700 hover:underline"
      >
        {info.getValue()}
      </Link>
    ),
  }),
  column.accessor("status", {
    header: "Status",
    cell: (info) => <Status value={info.getValue()} />,
  }),
  column.accessor((row) => objective(row.objective), {
    id: "objective",
    header: "Objective",
    cell: (info) => <Text value={info.getValue()} reason="not recorded" />,
  }),
  column.accessor((row) => Object.values(row.node_counts).reduce((a, b) => a + b, 0), {
    id: "candidates",
    header: "Candidates",
    cell: (info) => <NodeCounts counts={info.row.original.node_counts} total={info.getValue()} />,
  }),
  column.accessor("best_node_id", {
    header: "Best",
    enableSorting: false,
    cell: (info) => {
      const best = info.getValue();
      return best ? <ShortId value={best} /> : <Absent reason="no candidate decided yet" />;
    },
  }),
  column.accessor("created_at", {
    header: "Created",
    cell: (info) => <Time value={info.getValue()} />,
  }),
];

/** "maximize task_success ≥ 0.82": the primary objective, from the recorded document. */
export function objective(document: Record<string, unknown>): string | null {
  const primary = document.primary as { name?: string; direction?: string } | undefined;
  if (!primary?.name) return null;
  const target = document.target;
  const bound =
    typeof target === "number" ? ` ${primary.direction === "minimize" ? "≤" : "≥"} ${target}` : "";
  return `${primary.direction ?? ""} ${primary.name}${bound}`.trim();
}

function NodeCounts({ counts, total }: { counts: Record<string, number>; total: number }) {
  const summary = Object.entries(counts)
    .map(([status, count]) => `${count} ${status}`)
    .join(", ");
  return (
    <span title={summary} className="tabular-nums">
      {total} <span className="text-xs text-gray-500">{summary ? `(${summary})` : ""}</span>
    </span>
  );
}

/** J2: every experiment, newest first, keyset-paged (chapter 24 §4.2). */
export function ExperimentsPage() {
  const search = useSearch({ from: "/experiments" });
  const navigate = useNavigate({ from: "/experiments" });
  const experiments = useExperiments({ status: search.status, before: search.before });
  return (
    <Page
      title="Experiments"
      subtitle="Newest first. The list is live: pages follow by key, not number."
    >
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-sm">
          Status
          <select
            className="rounded border border-gray-300 bg-white px-2 py-1 text-sm"
            value={search.status ?? ""}
            onChange={(event) =>
              void navigate({
                search: { status: event.target.value || undefined, before: undefined },
              })
            }
          >
            <option value="">all</option>
            {EXPERIMENT_STATUSES.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Loading query={experiments}>
        {() => (
          <>
            <DataTable
              label="Experiments"
              data={experiments.data?.items ?? []}
              columns={COLUMNS}
              empty="No experiments match."
            />
            <div className="flex justify-between text-sm">
              {search.before ? (
                <Link
                  to="/experiments"
                  search={{ status: search.status }}
                  className="text-blue-700"
                >
                  ← newest
                </Link>
              ) : (
                <span />
              )}
              {experiments.data?.next_before ? (
                <Link
                  to="/experiments"
                  search={{ status: search.status, before: experiments.data.next_before }}
                  className="text-blue-700"
                >
                  older →
                </Link>
              ) : null}
            </div>
          </>
        )}
      </Loading>
    </Page>
  );
}
