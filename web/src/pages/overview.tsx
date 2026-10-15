import { useQueries } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { api, read, type Schemas } from "../api/client";
import { useExperiments } from "../api/queries";
import { Card, Loading, Page, Section } from "../components/layout";
import { Status } from "../components/status";
import { ShortId, Time } from "../components/values";

/** J1, J6: what is running, what failed, and what waits for a person. */
export function OverviewPage() {
  const active = useExperiments({ status: "active", limit: 50 });
  const failed = useExperiments({ status: "failed", limit: 10 });
  const recent = useExperiments({ limit: 10 });
  return (
    <Page
      title="Overview"
      subtitle="Experiments and their state, read from the control-plane record."
    >
      <Section
        title="Active experiments"
        aside={
          <Link to="/experiments" search={{ status: "active" }} className="text-sm text-blue-700">
            view all →
          </Link>
        }
      >
        <Loading query={active}>
          {() => <ExperimentList items={active.data?.items ?? []} empty="Nothing is running." />}
        </Loading>
      </Section>
      {active.data && active.data.items.length > 0 ? (
        <WaitingForApproval experiments={active.data.items} />
      ) : null}
      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="Recent failures">
          <Loading query={failed}>
            {() => (
              <ExperimentList items={failed.data?.items ?? []} empty="No failed experiments." />
            )}
          </Loading>
        </Section>
        <Section title="Recently created">
          <Loading query={recent}>
            {() => <ExperimentList items={recent.data?.items ?? []} empty="No experiments yet." />}
          </Loading>
        </Section>
      </div>
    </Page>
  );
}

function ExperimentList({
  items,
  empty,
}: {
  items: Schemas["ExperimentSummary"][];
  empty: string;
}) {
  if (items.length === 0) return <p className="text-sm text-gray-500">{empty}</p>;
  return (
    <Card>
      <ul className="divide-y divide-gray-100">
        {items.map((item) => (
          <li key={item.id} className="flex flex-wrap items-center justify-between gap-2 py-2">
            <Link
              to="/experiments/$experimentId"
              params={{ experimentId: item.id }}
              className="font-medium text-blue-700 hover:underline"
            >
              {item.name}
            </Link>
            <span className="flex items-center gap-3 text-xs text-gray-500">
              <Status value={item.status} />
              <Time value={item.updated_at} />
            </span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

/** Actions in approval_pending across active experiments, with the command to answer them. */
function WaitingForApproval({ experiments }: { experiments: Schemas["ExperimentSummary"][] }) {
  const actions = useQueries({
    queries: experiments.map((experiment) => ({
      queryKey: ["experiment", experiment.id, "actions"],
      queryFn: () =>
        read(
          api.GET("/api/v1/experiments/{experiment_id}/actions", {
            params: { path: { experiment_id: experiment.id } },
          }),
        ),
    })),
  });
  const waiting = actions.flatMap((query, index) =>
    (query.data?.items ?? [])
      .filter((action) => action.status === "approval_pending")
      .map((action) => ({ action, experiment: experiments[index] })),
  );
  if (waiting.length === 0) return null;
  return (
    <Section title="Waiting for approval">
      <Card>
        <ul className="divide-y divide-gray-100 text-sm">
          {waiting.map(({ action, experiment }) => (
            <li key={action.id} className="space-y-1 py-2">
              <div className="flex items-center gap-2">
                <Status value={action.status} />
                <span className="font-medium">{action.type}</span>
                <span className="text-gray-500">in {experiment?.name}</span>
              </div>
              <p className="text-gray-600">{action.reason}</p>
              <p className="text-xs text-gray-500">
                Studio is read-only. Answer it from the CLI:{" "}
                <code className="rounded bg-gray-100 px-1">
                  xaytune approve {action.id} --reason "…"
                </code>{" "}
                or{" "}
                <code className="rounded bg-gray-100 px-1">
                  xaytune reject {action.id} --reason "…"
                </code>{" "}
                <ShortId value={action.id} />
              </p>
            </li>
          ))}
        </ul>
      </Card>
    </Section>
  );
}
