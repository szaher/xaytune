import { Card, Page } from "../components/layout";

/**
 * Sections whose data is not in the control-plane record yet (ADR-023 §S12).
 * They are present so the product's shape is visible, and say what they wait
 * for rather than showing anything invented (chapter 24 §6).
 */
const WAITING = {
  monitoring: {
    title: "Monitoring",
    body: "Training-metric series (loss curves, telemetry) are emitted by workers but not yet recorded in the control-plane store. Charts arrive with the metric-point store (Track B, migration 020). Failures, incidents and recovery are shown per experiment, under Activity.",
  },
  trajectories: {
    title: "Trajectories",
    body: "Agent trajectories are file stores (ADR-021) not yet referenced from the control plane. The inspector arrives with durable harness execution and trajectory references (Track B), and shows released-tier events only.",
  },
  optimization: {
    title: "Optimization",
    body: "Search and mutation history arrives with H07 (Track C). Today, each experiment's planner and its invocation history are shown under the experiment's Configuration tab.",
  },
} as const;

export function WaitingPage({ section }: { section: keyof typeof WAITING }) {
  const { title, body } = WAITING[section];
  return (
    <Page title={title} subtitle="Not in the record yet">
      <Card>
        <p className="max-w-3xl text-sm text-gray-700">{body}</p>
      </Card>
    </Page>
  );
}
