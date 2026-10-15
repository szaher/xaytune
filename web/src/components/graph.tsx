/**
 * The candidate graph (chapter 24 §4.3): React Flow, laid out by ELK's layered
 * algorithm, top-down. Every node is focusable and selectable from the
 * keyboard, and the same candidates are listed in a table beneath it, so the
 * lineage never depends on seeing the canvas.
 */
import {
  Background,
  Controls,
  type Edge,
  Handle,
  type Node,
  type NodeProps,
  Position,
  ReactFlow,
} from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import type { Schemas } from "../api/client";
import { Status } from "./status";

type GraphNode = Schemas["NodeOut"];
type CandidateData = { node: GraphNode; selected: boolean };

const WIDTH = 220;
const HEIGHT = 92;
/** ELK is large (~1.5 MB); it loads with the first graph, not with Studio. */
let elk: Promise<{ layout: (graph: ElkGraph) => Promise<ElkGraph> }> | undefined;
type ElkGraph = {
  id: string;
  layoutOptions?: Record<string, string>;
  children?: { id: string; width?: number; height?: number; x?: number; y?: number }[];
  edges?: { id: string; sources: string[]; targets: string[] }[];
};
const loadElk = () => {
  elk ??= import("elkjs/lib/elk.bundled.js").then(({ default: ELK }) => new ELK());
  return elk;
};

function CandidateCard({ data }: NodeProps<Node<CandidateData>>) {
  const { node, selected } = data;
  return (
    <div
      className={`w-[220px] rounded-md border bg-white px-3 py-2 text-left text-xs shadow-sm ${
        selected ? "border-blue-600 ring-2 ring-blue-300" : "border-gray-300"
      }`}
      data-testid={`candidate-${node.id}`}
    >
      <Handle type="target" position={Position.Top} className="!bg-gray-400" />
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono" title={node.id}>
          {node.id.slice(0, 12)}…
        </span>
        {node.is_best ? (
          <span className="text-amber-600" title="best candidate">
            ★
          </span>
        ) : null}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-1">
        <Status value={node.status} />
        {node.latest_decision ? <Status value={node.latest_decision.outcome} /> : null}
      </div>
      <p className="mt-1 truncate text-gray-600" title={node.hypothesis ?? undefined}>
        {node.hypothesis ?? node.reason ?? "no hypothesis recorded"}
      </p>
      <Handle type="source" position={Position.Bottom} className="!bg-gray-400" />
    </div>
  );
}

const NODE_TYPES = { candidate: CandidateCard };

export function CandidateGraph({
  graph,
  selected,
  onSelect,
}: {
  graph: Schemas["GraphOut"];
  selected: string | undefined;
  onSelect: (nodeId: string) => void;
}) {
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  // Laid out again only when the topology changes, not on every poll.
  const topology = JSON.stringify({
    ids: graph.nodes.map((node) => node.id),
    edges: graph.edges.map((edge) => [edge.parent_id, edge.child_id]),
  });

  useEffect(() => {
    const { ids, edges } = JSON.parse(topology) as { ids: string[]; edges: [string, string][] };
    let current = true;
    void loadElk()
      .then((engine) =>
        engine.layout({
          id: "root",
          layoutOptions: {
            "elk.algorithm": "layered",
            "elk.direction": "DOWN",
            "elk.spacing.nodeNode": "40",
            "elk.layered.spacing.nodeNodeBetweenLayers": "60",
          },
          children: ids.map((id) => ({ id, width: WIDTH, height: HEIGHT })),
          edges: edges.map(([parent, child]) => ({
            id: `${parent}->${child}`,
            sources: [parent],
            targets: [child],
          })),
        }),
      )
      .then((laid) => {
        if (!current) return;
        const next: Record<string, { x: number; y: number }> = {};
        for (const child of laid.children ?? [])
          next[child.id] = { x: child.x ?? 0, y: child.y ?? 0 };
        setPositions(next);
      });
    return () => {
      current = false;
    };
  }, [topology]);

  const byId = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph.nodes]);
  const nodes: Node<CandidateData>[] = graph.nodes.map((node) => ({
    id: node.id,
    type: "candidate",
    position: positions[node.id] ?? { x: 0, y: 0 },
    data: { node, selected: node.id === selected },
    ariaLabel: `candidate ${node.id}, ${node.status}${node.is_best ? ", best" : ""}`,
    // Enter or Space selects it, so it is announced as a button.
    ariaRole: "button",
    draggable: false,
    connectable: false,
  }));
  const edges: Edge[] = graph.edges.map((edge) => ({
    id: `${edge.parent_id}->${edge.child_id}`,
    source: edge.parent_id,
    target: edge.child_id,
    label: byId.get(edge.child_id)?.created_by.type,
    labelStyle: { fontSize: 11 },
  }));

  return (
    <div
      role="group"
      className="h-[480px] rounded border border-gray-200 bg-white"
      aria-label="Candidate graph"
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.2, maxZoom: 1 }}
        maxZoom={1.5}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        nodesFocusable
        onNodeClick={(_, node) => onSelect(node.id)}
        onKeyDown={(event) => {
          const id = (event.target as HTMLElement).closest("[data-id]")?.getAttribute("data-id");
          if (id && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            onSelect(id);
          }
        }}
        proOptions={{ hideAttribution: false }}
        key={Object.keys(positions).length > 0 ? "laid" : "initial"}
      >
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
