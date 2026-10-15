import { Link, useSearch } from "@tanstack/react-router";
import type { Schemas } from "../api/client";
import { useCompare } from "../api/queries";
import { Card, Loading, Page, Section } from "../components/layout";
import { Status } from "../components/status";
import { Absent, Num, ShortId } from "../components/values";

/**
 * J2, J3: two candidates side by side (chapter 24 §4.6, ADR-023 §S6).
 * Lineage is shown apart from the measurements, and two values are subtracted
 * only when the API says they are comparable: same evaluation, same unit.
 */
export function ComparePage() {
  const { left, right } = useSearch({ from: "/compare" });
  const comparison = useCompare(left, right);
  if (!left || !right) {
    return (
      <Page title="Compare">
        <Card>
          <p className="text-sm text-gray-600">
            Pick two candidates to compare: open an experiment's Candidates tab and use{" "}
            <em>Compare with parent</em>, or a candidate's own page.
          </p>
        </Card>
      </Page>
    );
  }
  return (
    <Loading query={comparison}>
      {() => <Comparison data={comparison.data as Schemas["CompareOut"]} />}
    </Loading>
  );
}

function Comparison({ data }: { data: Schemas["CompareOut"] }) {
  const { left, right, lineage } = data;
  return (
    <Page
      crumbs={
        <Link
          to="/experiments/$experimentId"
          params={{ experimentId: left.experiment_id }}
          search={{ tab: "candidates" }}
          className="hover:underline"
        >
          Experiment
        </Link>
      }
      title="Compare candidates"
      subtitle={
        <span className="flex flex-wrap items-center gap-2">
          <Side node={left} label="A" /> vs <Side node={right} label="B" />
        </span>
      }
    >
      <Section title="Lineage">
        <Card>
          <p className="text-sm">
            {lineage.distinct_candidates
              ? "Two distinct candidates."
              : "The same candidate fingerprint recorded twice: not alternatives."}{" "}
            Nearest common ancestor:{" "}
            {lineage.nearest_common_ancestor_ids.length ? (
              lineage.nearest_common_ancestor_ids.map((id) => <ShortId key={id} value={id} />)
            ) : (
              <Absent reason="none shared, or several" />
            )}
          </p>
          <p className="mt-1 text-xs text-gray-500">
            Lineage only: it does not make their measurements comparable. That is judged per metric
            below.
          </p>
        </Card>
      </Section>
      <Section title="Metrics">
        <MetricTable pairs={data.metrics} />
      </Section>
      <Section title="Candidate spec difference">
        {data.candidate_differences.length ? (
          <table className="text-sm" aria-label="Candidate spec difference">
            <thead className="text-left text-xs uppercase text-gray-500">
              <tr>
                <th className="py-1 pr-6">Path</th>
                <th className="pr-6">A</th>
                <th>B</th>
              </tr>
            </thead>
            <tbody className="font-mono text-xs">
              {data.candidate_differences.map((difference) => (
                <tr key={difference.path}>
                  <td className="py-1 pr-6">{difference.path}</td>
                  <td className="pr-6">{show(difference.left)}</td>
                  <td>{show(difference.right)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-gray-500">The candidate specs are identical.</p>
        )}
      </Section>
    </Page>
  );
}

function Side({ node, label }: { node: Schemas["NodeOut"]; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className="font-semibold">{label}</span>
      <Link
        to="/experiments/$experimentId/candidates/$nodeId"
        params={{ experimentId: node.experiment_id, nodeId: node.id }}
        className="text-blue-700 hover:underline"
      >
        <span className="font-mono text-xs">{node.id.slice(0, 14)}…</span>
      </Link>
      {node.is_best ? <span className="text-amber-600">★</span> : null}
      <Status value={node.status} />
    </span>
  );
}

function show(value: unknown) {
  return value === null || value === undefined ? (
    <Absent reason="not set" />
  ) : (
    JSON.stringify(value)
  );
}

export function MetricTable({ pairs }: { pairs: Schemas["MetricPair"][] }) {
  if (pairs.length === 0)
    return <p className="text-sm text-gray-500">Neither candidate has results.</p>;
  return (
    <div className="overflow-x-auto rounded border border-gray-200 bg-white">
      <table className="min-w-full text-sm" aria-label="Metrics compared">
        <thead className="bg-gray-50 text-left text-xs uppercase text-gray-500">
          <tr>
            <th className="px-3 py-2">Metric (evaluator · slice)</th>
            <th className="px-3 py-2">A</th>
            <th className="px-3 py-2">B</th>
            <th className="px-3 py-2">A − B</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {pairs.map((pair) => (
            <tr key={`${pair.evaluator_name}/${pair.name}/${pair.slice ?? ""}`}>
              <td className="px-3 py-2">
                {pair.name}{" "}
                <span className="text-gray-500">
                  ({pair.evaluator_name} · {pair.slice ?? "—"})
                </span>
              </td>
              <td className="px-3 py-2">
                <MeasuredValue side={pair.left} />
              </td>
              <td className="px-3 py-2">
                <MeasuredValue side={pair.right} />
              </td>
              <td className="px-3 py-2" data-testid={`difference-${pair.name}`}>
                {pair.comparable ? (
                  <span className="tabular-nums">
                    {pair.difference !== null && pair.difference > 0 ? "+" : ""}
                    <Num value={pair.difference} />
                    <span className="block text-xs text-green-700">✓ same evaluation</span>
                  </span>
                ) : (
                  <span className="text-amber-800">
                    ⚠ not comparable
                    <span className="block text-xs">{pair.reasons.join("; ")}</span>
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MeasuredValue({ side }: { side: Schemas["MetricSide"] | null }) {
  if (!side) return <Absent reason="not measured" />;
  return (
    <span>
      <span className="tabular-nums">
        <Num value={side.metric.value} />
        {side.unit ? ` ${side.unit}` : ""}
      </span>
      {side.metric.standard_error !== null ? (
        <span className="text-xs text-gray-500"> ±{side.metric.standard_error}</span>
      ) : null}
      <span className="block text-xs text-gray-500">
        eval <ShortId value={side.evaluation_fingerprint} keep={12} />
      </span>
    </span>
  );
}
