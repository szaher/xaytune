import { Link, useParams } from "@tanstack/react-router";
import type { Schemas } from "../api/client";
import { useNode } from "../api/queries";
import { DocumentTree } from "../components/document";
import { Card, Facts, Loading, Page, Section } from "../components/layout";
import { Status } from "../components/status";
import { DataTable } from "../components/table";
import { Absent, ShortId, Text, Time } from "../components/values";
import { DECISION_COLUMNS, EVALUATION_COLUMNS, RUN_COLUMNS } from "./experiment";

/** J2, J3: one candidate, its lineage, runs, evaluations and decisions (chapter 24 §4.4). */
export function CandidatePage() {
  const { experimentId, nodeId } = useParams({
    from: "/experiments/$experimentId/candidates/$nodeId",
  });
  const node = useNode(nodeId);
  return (
    <Loading query={node}>
      {() => {
        const data = node.data as Schemas["NodeDetail"];
        const { node: candidate } = data;
        const parent = candidate.parent_ids[0];
        return (
          <Page
            crumbs={
              <span className="flex gap-1">
                <Link to="/experiments" className="hover:underline">
                  Experiments
                </Link>
                ›
                <Link
                  to="/experiments/$experimentId"
                  params={{ experimentId }}
                  search={{ tab: "candidates", node: nodeId }}
                  className="hover:underline"
                >
                  Experiment
                </Link>
                › Candidate
              </span>
            }
            title={
              <span className="flex items-center gap-2">
                Candidate <ShortId value={candidate.id} keep={16} />
                {candidate.is_best ? <span className="text-amber-600">★ best</span> : null}
              </span>
            }
            subtitle={
              <span className="flex flex-wrap items-center gap-3">
                <Status value={candidate.status} />
                <span>cycle {candidate.evaluation_cycle}</span>
                <span>
                  fingerprint <ShortId value={candidate.candidate_fingerprint} keep={14} />
                </span>
              </span>
            }
          >
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <Facts
                  items={[
                    [
                      "Hypothesis",
                      <Text key="h" value={candidate.hypothesis} reason="not recorded" />,
                    ],
                    ["Reason", <Text key="r" value={candidate.reason} reason="not recorded" />],
                    ["Created by", `${candidate.created_by.type} ${candidate.created_by.id}`],
                    ["Created", <Time key="c" value={candidate.created_at} />],
                    [
                      "Lineage",
                      <ol key="l" className="space-y-1" aria-label="Lineage">
                        {data.lineage_paths.map((path) => (
                          <li key={path.join(">")} className="flex flex-wrap items-center gap-1">
                            {path.map((step, index) => (
                              <span key={step} className="flex items-center gap-1">
                                {index > 0 ? <span aria-hidden="true">→</span> : null}
                                <Link
                                  to="/experiments/$experimentId/candidates/$nodeId"
                                  params={{ experimentId, nodeId: step }}
                                  className="text-blue-700 hover:underline"
                                >
                                  <span className="font-mono text-xs">{step.slice(0, 14)}…</span>
                                </Link>
                              </span>
                            ))}
                          </li>
                        ))}
                      </ol>,
                    ],
                  ]}
                />
                {parent ? (
                  <Link
                    to="/compare"
                    search={{ left: candidate.id, right: parent }}
                    className="mt-4 inline-block rounded border border-gray-300 px-3 py-1.5 text-sm hover:bg-gray-50"
                  >
                    Compare with parent
                  </Link>
                ) : null}
              </Card>
              <Card>
                <h3 className="mb-2 text-sm font-semibold">Candidate spec</h3>
                <DocumentTree value={data.candidate} label="Candidate spec" />
              </Card>
            </div>
            {candidate.branch_origin ? (
              <Section title="Branch origin">
                <Card>
                  <DocumentTree value={candidate.branch_origin} label="Branch origin" />
                </Card>
              </Section>
            ) : null}
            <Section title="Runs">
              <DataTable label="Runs" data={data.runs} columns={RUN_COLUMNS} empty="No runs yet." />
            </Section>
            <Section title="Evaluations">
              <DataTable
                label="Evaluations"
                data={data.evaluation_runs}
                columns={EVALUATION_COLUMNS}
                empty="No evaluations yet."
              />
            </Section>
            <Section title="Decisions">
              {data.decisions.length ? (
                <DataTable label="Decisions" data={data.decisions} columns={DECISION_COLUMNS} />
              ) : (
                <Absent reason="not decided yet" />
              )}
            </Section>
          </Page>
        );
      }}
    </Loading>
  );
}
