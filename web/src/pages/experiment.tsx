import { useQueries } from "@tanstack/react-query";
import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { createColumnHelper } from "@tanstack/react-table";
import type { ReactNode } from "react";
import { api, read, type Schemas } from "../api/client";
import {
  useActions,
  useAgentInvocations,
  useBudget,
  useDecisions,
  useEvents,
  useExperiment,
  useGraph,
  useIncidents,
} from "../api/queries";
import { DocumentTree } from "../components/document";
import { CandidateGraph } from "../components/graph";
import { Card, Facts, Loading, Page, Section } from "../components/layout";
import { Status } from "../components/status";
import { DataTable } from "../components/table";
import { Absent, Num, ShortId, Text, Time } from "../components/values";
import { objective } from "./experiments";

export const EXPERIMENT_TABS = [
  "overview",
  "candidates",
  "runs",
  "decisions",
  "actions",
  "activity",
  "budget",
  "configuration",
] as const;
export type ExperimentTab = (typeof EXPERIMENT_TABS)[number];

const FROM = "/experiments/$experimentId" as const;

/** One experiment, tab by tab (chapter 24 §3). Live while it is not at rest. */
export function ExperimentPage() {
  const { experimentId } = useParams({ from: FROM });
  const search = useSearch({ from: FROM });
  const experiment = useExperiment(experimentId);
  const tab = search.tab ?? "overview";
  return (
    <Loading query={experiment}>
      {() => {
        const data = experiment.data as Schemas["ExperimentOut"];
        const live = !data.quiescent;
        return (
          <Page
            crumbs={
              <Link to="/experiments" className="hover:underline">
                Experiments
              </Link>
            }
            title={data.name}
            subtitle={
              <span className="flex flex-wrap items-center gap-3">
                <Status value={data.status} />
                <span>
                  next stage:{" "}
                  {data.next_stage ? (
                    <span className="font-medium">{data.next_stage}</span>
                  ) : (
                    <Absent reason="nothing left to do" />
                  )}
                </span>
                {live ? <span className="text-blue-700">● live, updating every 2 s</span> : null}
                <ShortId value={data.id} />
              </span>
            }
          >
            <Tabs experimentId={experimentId} current={tab} />
            <div role="tabpanel" aria-label={tab}>
              {tab === "overview" ? <OverviewTab experiment={data} /> : null}
              {tab === "candidates" ? (
                <CandidatesTab experimentId={experimentId} selected={search.node} live={live} />
              ) : null}
              {tab === "runs" ? <RunsTab experimentId={experimentId} live={live} /> : null}
              {tab === "decisions" ? (
                <DecisionsTab experimentId={experimentId} live={live} />
              ) : null}
              {tab === "actions" ? <ActionsTab experimentId={experimentId} live={live} /> : null}
              {tab === "activity" ? (
                <ActivityTab experimentId={experimentId} after={search.after ?? 0} live={live} />
              ) : null}
              {tab === "budget" ? <BudgetTab experimentId={experimentId} live={live} /> : null}
              {tab === "configuration" ? <ConfigurationTab experiment={data} /> : null}
            </div>
          </Page>
        );
      }}
    </Loading>
  );
}

function Tabs({ experimentId, current }: { experimentId: string; current: ExperimentTab }) {
  return (
    <div
      role="tablist"
      aria-label="Experiment"
      className="flex flex-wrap gap-1 border-b border-gray-200"
    >
      {EXPERIMENT_TABS.map((tab) => (
        <Link
          key={tab}
          role="tab"
          aria-selected={tab === current}
          to={FROM}
          params={{ experimentId }}
          search={{ tab }}
          className={`-mb-px border-b-2 px-3 py-2 text-sm capitalize ${
            tab === current
              ? "border-blue-600 font-medium text-blue-700"
              : "border-transparent text-gray-600 hover:text-gray-900"
          }`}
        >
          {tab}
        </Link>
      ))}
    </div>
  );
}

// ---- overview ----------------------------------------------------------------------------------

function OverviewTab({ experiment }: { experiment: Schemas["ExperimentOut"] }) {
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <Facts
          items={[
            [
              "Objective",
              <Text key="o" value={objective(experiment.objective)} reason="not recorded" />,
            ],
            [
              "Best candidate",
              experiment.best_node_id ? (
                <Link
                  key="b"
                  to="/experiments/$experimentId/candidates/$nodeId"
                  params={{ experimentId: experiment.id, nodeId: experiment.best_node_id }}
                  className="text-blue-700 hover:underline"
                >
                  <ShortId value={experiment.best_node_id} />
                </Link>
              ) : (
                <Absent key="b" reason="no candidate decided yet" />
              ),
            ],
            ["At rest", experiment.quiescent ? "yes: all executable work is settled" : "no"],
            [
              "Controller",
              `${experiment.controller_host.kind} ${experiment.controller_host.id ?? ""}`,
            ],
            ["Created", <Time key="c" value={experiment.created_at} />],
            ["Updated", <Time key="u" value={experiment.updated_at} />],
          ]}
        />
      </Card>
      <Card>
        <h3 className="mb-2 text-sm font-semibold">Candidates by status</h3>
        {Object.keys(experiment.node_counts).length === 0 ? (
          <Absent reason="no candidates yet" />
        ) : (
          <ul className="flex flex-wrap gap-2">
            {Object.entries(experiment.node_counts).map(([status, count]) => (
              <li key={status} className="flex items-center gap-1 text-sm">
                <Status value={status} /> <span className="tabular-nums">{count}</span>
              </li>
            ))}
          </ul>
        )}
        <h3 className="mb-2 mt-4 text-sm font-semibold">Budget</h3>
        <BudgetDimensions dimensions={experiment.budget_status} />
      </Card>
    </div>
  );
}

function BudgetDimensions({ dimensions }: { dimensions: Schemas["DimensionOut"][] | null }) {
  if (!dimensions) return <Absent reason="no budget limits this experiment" />;
  return (
    <table className="text-sm" aria-label="Budget balances">
      <thead className="text-left text-xs text-gray-500">
        <tr>
          <th className="pr-4">dimension</th>
          <th className="pr-4">limit</th>
          <th className="pr-4">consumed</th>
          <th className="pr-4">reserved</th>
          <th>remaining</th>
        </tr>
      </thead>
      <tbody className="tabular-nums">
        {dimensions.map((d) => (
          <tr key={d.dimension}>
            <td className="pr-4">{d.dimension}</td>
            <td className="pr-4">{d.limit}</td>
            <td className="pr-4">{d.consumed}</td>
            <td className="pr-4">{d.reserved}</td>
            <td>{d.remaining}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---- candidates --------------------------------------------------------------------------------

function CandidatesTab({
  experimentId,
  selected,
  live,
}: {
  experimentId: string;
  selected: string | undefined;
  live: boolean;
}) {
  const graph = useGraph(experimentId, live);
  const navigate = useNavigate({ from: FROM });
  const select = (node: string) =>
    void navigate({ search: (previous) => ({ ...previous, tab: "candidates", node }) });
  return (
    <Loading query={graph}>
      {() => {
        const data = graph.data as Schemas["GraphOut"];
        if (data.nodes.length === 0)
          return <p className="text-sm text-gray-500">No candidates yet.</p>;
        const chosen = data.nodes.find((node) => node.id === selected);
        return (
          <div className="space-y-6">
            <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
              <CandidateGraph graph={data} selected={selected} onSelect={select} />
              <CandidatePanel experimentId={experimentId} node={chosen} />
            </div>
            <CandidateTable experimentId={experimentId} nodes={data.nodes} onSelect={select} />
          </div>
        );
      }}
    </Loading>
  );
}

function CandidatePanel({
  experimentId,
  node,
}: {
  experimentId: string;
  node: Schemas["NodeOut"] | undefined;
}) {
  if (!node) {
    return (
      <Card>
        <p className="text-sm text-gray-500">
          Select a candidate in the graph (click, or focus it and press Enter) or in the table.
        </p>
      </Card>
    );
  }
  const parent = node.parent_ids[0];
  return (
    <Card>
      <div role="group" className="space-y-3 text-sm" aria-label="Selected candidate">
        <div className="flex items-center justify-between">
          <ShortId value={node.id} />
          {node.is_best ? <span className="text-amber-600">★ best</span> : null}
        </div>
        <div className="flex flex-wrap gap-1">
          <Status value={node.status} />
          <span className="text-gray-500">cycle {node.evaluation_cycle}</span>
        </div>
        <Facts
          items={[
            ["Hypothesis", <Text key="h" value={node.hypothesis} reason="not recorded" />],
            ["Reason", <Text key="r" value={node.reason} reason="not recorded" />],
            ["Created by", `${node.created_by.type} ${node.created_by.id}`],
            [
              "Decision",
              node.latest_decision ? (
                <Status key="d" value={node.latest_decision.outcome} />
              ) : (
                <Absent key="d" reason="not decided" />
              ),
            ],
          ]}
        />
        <div className="flex flex-col gap-2">
          <Link
            to="/experiments/$experimentId/candidates/$nodeId"
            params={{ experimentId, nodeId: node.id }}
            className="rounded bg-blue-600 px-3 py-1.5 text-center text-white hover:bg-blue-700"
          >
            Open candidate →
          </Link>
          {parent ? (
            <Link
              to="/compare"
              search={{ left: node.id, right: parent }}
              className="rounded border border-gray-300 px-3 py-1.5 text-center hover:bg-gray-50"
            >
              Compare with parent
            </Link>
          ) : null}
        </div>
      </div>
    </Card>
  );
}

const nodeColumn = createColumnHelper<Schemas["NodeOut"]>();

function CandidateTable({
  experimentId,
  nodes,
  onSelect,
}: {
  experimentId: string;
  nodes: Schemas["NodeOut"][];
  onSelect: (id: string) => void;
}) {
  const columns = [
    nodeColumn.accessor("id", {
      header: "Candidate",
      cell: (info) => (
        <span className="flex items-center gap-2">
          <Link
            to="/experiments/$experimentId/candidates/$nodeId"
            params={{ experimentId, nodeId: info.getValue() }}
            className="text-blue-700 hover:underline"
          >
            <span className="font-mono text-xs">{info.getValue()}</span>
          </Link>
          {info.row.original.is_best ? <span className="text-amber-600">★</span> : null}
          <button
            type="button"
            className="text-xs text-gray-500 underline"
            onClick={() => onSelect(info.getValue())}
          >
            select
          </button>
        </span>
      ),
    }),
    nodeColumn.accessor("status", {
      header: "Status",
      cell: (info) => <Status value={info.getValue()} />,
    }),
    nodeColumn.accessor((row) => row.latest_decision?.outcome ?? "", {
      id: "decision",
      header: "Decision",
      cell: (info) =>
        info.getValue() ? <Status value={info.getValue()} /> : <Absent reason="not decided" />,
    }),
    nodeColumn.accessor((row) => row.parent_ids.join(", "), {
      id: "parents",
      header: "Parents",
      cell: (info) =>
        info.row.original.parent_ids.length ? (
          <span className="font-mono text-xs">{info.getValue()}</span>
        ) : (
          <span className="text-gray-500">root</span>
        ),
    }),
    nodeColumn.accessor((row) => row.created_by.type, { id: "creator", header: "Created by" }),
    nodeColumn.accessor("hypothesis", {
      header: "Hypothesis",
      enableSorting: false,
      cell: (info) => <Text value={info.getValue()} reason="not recorded" />,
    }),
  ];
  return (
    <Section title="Candidates">
      <DataTable label="Candidates" data={nodes} columns={columns} />
    </Section>
  );
}

// ---- runs --------------------------------------------------------------------------------------

function useNodeDetails(experimentId: string, live: boolean) {
  const graph = useGraph(experimentId, live);
  const details = useQueries({
    queries: (graph.data?.nodes ?? []).map((node) => ({
      queryKey: ["node", node.id],
      queryFn: () =>
        read(api.GET("/api/v1/nodes/{node_id}", { params: { path: { node_id: node.id } } })),
      refetchInterval: live ? 2_000 : (false as const),
    })),
  });
  return { graph, details: details.flatMap((query) => (query.data ? [query.data] : [])) };
}

function RunsTab({ experimentId, live }: { experimentId: string; live: boolean }) {
  const { graph, details } = useNodeDetails(experimentId, live);
  const runs = details.flatMap((detail) => detail.runs);
  const evaluations = details.flatMap((detail) => detail.evaluation_runs);
  return (
    <Loading query={graph}>
      {() => (
        <div className="space-y-6">
          <Section title="Training runs">
            <DataTable
              label="Training runs"
              data={runs}
              columns={RUN_COLUMNS}
              empty="No runs yet."
            />
          </Section>
          <Section title="Evaluation runs">
            <DataTable
              label="Evaluation runs"
              data={evaluations}
              columns={EVALUATION_COLUMNS}
              empty="No evaluations yet."
            />
          </Section>
        </div>
      )}
    </Loading>
  );
}

const runColumn = createColumnHelper<Schemas["RunOut"]>();
export const RUN_COLUMNS = [
  runColumn.accessor("id", {
    header: "Run",
    cell: (info) => (
      <Link
        to="/runs/$runId"
        params={{ runId: info.getValue() }}
        className="text-blue-700 hover:underline"
      >
        <span className="font-mono text-xs">{info.getValue()}</span>
      </Link>
    ),
  }),
  runColumn.accessor("status", {
    header: "Status",
    cell: (info) => <Status value={info.getValue()} />,
  }),
  runColumn.accessor("node_id", {
    header: "Candidate",
    cell: (info) => <ShortId value={info.getValue()} />,
  }),
  runColumn.accessor("seed", {
    header: "Seed",
    cell: (info) => (
      <span>
        <Num value={info.getValue()} reason="no seed" />
        {info.row.original.seed_origin ? (
          <span className="text-xs text-gray-500"> (inherited)</span>
        ) : null}
      </span>
    ),
  }),
  runColumn.accessor((row) => row.attempts.length, { id: "attempts", header: "Attempts" }),
  runColumn.accessor("created_at", {
    header: "Created",
    cell: (info) => <Time value={info.getValue()} />,
  }),
];

const evaluationColumn = createColumnHelper<Schemas["EvaluationRunOut"]>();
export const EVALUATION_COLUMNS = [
  evaluationColumn.accessor("id", {
    header: "Evaluation run",
    cell: (info) => (
      <Link
        to="/evaluation-runs/$evaluationRunId"
        params={{ evaluationRunId: info.getValue() }}
        className="text-blue-700 hover:underline"
      >
        <span className="font-mono text-xs">{info.getValue()}</span>
      </Link>
    ),
  }),
  evaluationColumn.accessor("status", {
    header: "Status",
    cell: (info) => <Status value={info.getValue()} />,
  }),
  evaluationColumn.accessor("node_id", {
    header: "Candidate",
    cell: (info) => <ShortId value={info.getValue()} />,
  }),
  evaluationColumn.accessor("evaluation_cycle", { header: "Cycle" }),
  evaluationColumn.accessor((row) => row.result?.metrics ?? [], {
    id: "metrics",
    header: "Metrics",
    enableSorting: false,
    cell: (info) =>
      info.row.original.result ? (
        <ul>
          {info.getValue().map((metric: Schemas["MetricOut"]) => (
            <li key={`${metric.evaluator_name}/${metric.name}/${metric.slice ?? ""}`}>
              {metric.name}: <Num value={metric.value} />
            </li>
          ))}
        </ul>
      ) : (
        <Absent reason="no result recorded" />
      ),
  }),
];

// ---- decisions, actions ------------------------------------------------------------------------

const decisionColumn = createColumnHelper<Schemas["DecisionOut"]>();
export const DECISION_COLUMNS = [
  decisionColumn.accessor("created_at", {
    header: "When",
    cell: (info) => <Time value={info.getValue()} />,
  }),
  decisionColumn.accessor("outcome", {
    header: "Outcome",
    cell: (info) => <Status value={info.getValue()} />,
  }),
  decisionColumn.accessor("node_id", {
    header: "Candidate",
    cell: (info) => <ShortId value={info.getValue()} />,
  }),
  decisionColumn.accessor("reason", {
    header: "Reason",
    enableSorting: false,
    cell: (info) => <Text value={info.getValue()} />,
  }),
  decisionColumn.accessor((row) => `${row.engine_name} ${row.engine_version}`, {
    id: "engine",
    header: "Engine",
  }),
  decisionColumn.accessor("evaluation_result_ids", {
    header: "Read results",
    enableSorting: false,
    cell: (info) =>
      info.getValue().length ? (
        <ul>
          {info.getValue().map((id: string) => (
            <li key={id}>
              <ShortId value={id} />
            </li>
          ))}
        </ul>
      ) : (
        <Absent reason="no results read" />
      ),
  }),
];

function DecisionsTab({ experimentId, live }: { experimentId: string; live: boolean }) {
  const decisions = useDecisions(experimentId, live);
  return (
    <Loading query={decisions}>
      {() => (
        <DataTable
          label="Decisions"
          data={decisions.data?.items ?? []}
          columns={DECISION_COLUMNS}
          empty="No decisions yet."
        />
      )}
    </Loading>
  );
}

const actionColumn = createColumnHelper<Schemas["ActionOut"]>();
const ACTION_COLUMNS = [
  actionColumn.accessor("created_at", {
    header: "When",
    cell: (info) => <Time value={info.getValue()} />,
  }),
  actionColumn.accessor("type", { header: "Type" }),
  actionColumn.accessor("status", {
    header: "Status",
    cell: (info) => <Status value={info.getValue()} />,
  }),
  actionColumn.accessor((row) => row.policy_decision?.verdict ?? "", {
    id: "policy",
    header: "Policy",
    cell: (info) =>
      info.getValue() ? <Status value={info.getValue()} /> : <Absent reason="no policy decision" />,
  }),
  actionColumn.accessor((row) => `${row.target.kind as string} ${row.target.id as string}`, {
    id: "target",
    header: "Target",
  }),
  actionColumn.accessor("reason", {
    header: "Reason",
    enableSorting: false,
    cell: (info) => <Text value={info.getValue()} />,
  }),
  actionColumn.accessor((row) => `${row.proposed_by.type} ${row.proposed_by.id}`, {
    id: "proposer",
    header: "Proposed by",
  }),
];

function ActionsTab({ experimentId, live }: { experimentId: string; live: boolean }) {
  const actions = useActions(experimentId, live);
  return (
    <Loading query={actions}>
      {() => (
        <DataTable
          label="Actions"
          data={actions.data?.items ?? []}
          columns={ACTION_COLUMNS}
          empty="No actions recorded."
        />
      )}
    </Loading>
  );
}

// ---- activity ----------------------------------------------------------------------------------

const PAGE = 100;

function ActivityTab({
  experimentId,
  after,
  live,
}: {
  experimentId: string;
  after: number;
  live: boolean;
}) {
  const events = useEvents(experimentId, after, PAGE, live);
  const incidents = useIncidents(experimentId, live);
  return (
    <div className="space-y-6">
      <Section title="Incidents and recovery">
        <Loading query={incidents}>
          {() => <Incidents data={incidents.data as Schemas["IncidentList"]} />}
        </Loading>
      </Section>
      <Section
        title="Event journal"
        aside={
          <span className="flex gap-3 text-sm">
            {after > 0 ? (
              <Link
                to={FROM}
                params={{ experimentId }}
                search={{ tab: "activity" }}
                className="text-blue-700"
              >
                ← first
              </Link>
            ) : null}
            {events.data && events.data.items.length === PAGE ? (
              <Link
                to={FROM}
                params={{ experimentId }}
                search={{ tab: "activity", after: events.data.next_after }}
                className="text-blue-700"
              >
                next →
              </Link>
            ) : null}
          </span>
        }
      >
        <Loading query={events}>
          {() => (
            <DataTable
              label="Events"
              data={events.data?.items ?? []}
              columns={EVENT_COLUMNS}
              empty="No events after this point."
            />
          )}
        </Loading>
      </Section>
    </div>
  );
}

const eventColumn = createColumnHelper<Schemas["EventOut"]>();
const EVENT_COLUMNS = [
  eventColumn.accessor("sequence", { header: "#" }),
  eventColumn.accessor("occurred_at", {
    header: "When",
    cell: (info) => <Time value={info.getValue()} />,
  }),
  eventColumn.accessor("event_type", { header: "Event" }),
  eventColumn.accessor((row) => `${row.aggregate_type} ${row.aggregate_id}`, {
    id: "aggregate",
    header: "Aggregate",
    cell: (info) => (
      <span>
        {info.row.original.aggregate_type} <ShortId value={info.row.original.aggregate_id} />
      </span>
    ),
  }),
  eventColumn.accessor((row) => `${row.actor.type} ${row.actor.id}`, {
    id: "actor",
    header: "Actor",
  }),
];

function Incidents({ data }: { data: Schemas["IncidentList"] }) {
  if (data.incidents.length === 0 && data.recovery_episodes.length === 0) {
    return <p className="text-sm text-gray-500">No incidents recorded.</p>;
  }
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {data.incidents.map((incident) => (
        <Card key={incident.id}>
          <div
            role="group"
            className="space-y-2 text-sm"
            aria-label={`incident ${incident.category}`}
          >
            <div className="flex items-center gap-2">
              <span className="font-medium text-red-700">⚠ {incident.category}</span>
              <Time value={incident.created_at} />
            </div>
            <DocumentTree value={incident.evidence} label="Evidence" />
          </div>
        </Card>
      ))}
      {data.recovery_episodes.map((episode) => (
        <Card key={episode.id}>
          <div role="group" className="space-y-2 text-sm" aria-label="recovery episode">
            <p className="font-medium">
              Recovery of attempt {episode.attempt_number} · {episode.coordinator_name}{" "}
              {episode.coordinator_version}
            </p>
            {episode.effective_plan ? (
              <DocumentTree value={episode.effective_plan} label="Effective plan" />
            ) : (
              <Absent reason="no plan recorded" />
            )}
          </div>
        </Card>
      ))}
    </div>
  );
}

// ---- budget, configuration ---------------------------------------------------------------------

const ledgerColumn = createColumnHelper<Schemas["LedgerEntryOut"]>();
const LEDGER_COLUMNS = [
  ledgerColumn.accessor("created_at", {
    header: "When",
    cell: (info) => <Time value={info.getValue()} />,
  }),
  ledgerColumn.accessor("dimension", { header: "Dimension" }),
  ledgerColumn.accessor("kind", { header: "Entry" }),
  ledgerColumn.accessor("amount", {
    header: "Amount",
    cell: (info) => <span className="tabular-nums">{info.getValue()}</span>,
  }),
  ledgerColumn.accessor((row) => `${row.subject_kind} ${row.subject_id}`, {
    id: "subject",
    header: "Subject",
  }),
];

function BudgetTab({ experimentId, live }: { experimentId: string; live: boolean }) {
  const budget = useBudget(experimentId, live);
  return (
    <Loading query={budget}>
      {() => {
        const data = budget.data as Schemas["BudgetOut"];
        return (
          <div className="space-y-6">
            <Card>
              <BudgetDimensions dimensions={data.status} />
            </Card>
            <Section title="Ledger">
              <DataTable label="Budget ledger" data={data.entries} columns={LEDGER_COLUMNS} />
            </Section>
          </div>
        );
      }}
    </Loading>
  );
}

function ConfigurationTab({ experiment }: { experiment: Schemas["ExperimentOut"] }) {
  const invocations = useAgentInvocations(experiment.id);
  const documents: [string, unknown][] = [
    ["Objective", experiment.objective],
    ["Budget", experiment.budget],
    ["Planner", experiment.planner],
    ["Evaluation", experiment.evaluation],
    ["Compiler", experiment.compiler],
    ["Runtime", experiment.runtime],
    ["Resilience", experiment.resilience],
    ["Numerical recovery", experiment.numerical_recovery],
    ["Metadata", experiment.metadata],
  ];
  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-2">
        {documents.map(([title, value]) => (
          <Card key={title}>
            <h3 className="mb-2 text-sm font-semibold">{title}</h3>
            <DocumentTree value={value} label={title} />
          </Card>
        ))}
      </div>
      <Section title="Planner model calls">
        <Loading query={invocations}>
          {() => <Invocations items={invocations.data?.items ?? []} />}
        </Loading>
      </Section>
    </div>
  );
}

function Invocations({ items }: { items: Schemas["AgentInvocationOut"][] }): ReactNode {
  if (items.length === 0)
    return <p className="text-sm text-gray-500">No planner model calls recorded.</p>;
  return (
    <ul className="space-y-2 text-sm">
      {items.map((item) => (
        <li key={item.id} className="flex flex-wrap items-center gap-2">
          <Status value={item.status} />
          <span>
            {item.planner_kind} {item.planner_version}
          </span>
          <Time value={item.created_at} />
          {item.failure_kind ? <span className="text-red-700">{item.failure_kind}</span> : null}
        </li>
      ))}
    </ul>
  );
}
