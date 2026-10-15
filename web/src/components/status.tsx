/**
 * Statuses exactly as recorded, never renamed (chapter 24 §3), with one color
 * and icon per family across every aggregate. A status this build does not
 * know (a newer server) renders as itself, in neutral.
 */
type Tone = "running" | "success" | "failure" | "stopped" | "waiting" | "neutral";

const TONES: Record<string, Tone> = {
  active: "running",
  running: "running",
  starting: "running",
  queued: "waiting",
  evaluating: "running",
  checkpointing: "running",
  recovering: "running",
  executing: "running",
  validating: "running",
  deciding: "waiting",
  planned: "waiting",
  ready: "waiting",
  created: "neutral",
  proposed: "waiting",
  validated: "waiting",
  approval_pending: "waiting",
  approved: "waiting",
  paused: "waiting",
  succeeded: "success",
  completed: "success",
  stop_succeeded: "success",
  applied: "success",
  allow: "success",
  failed: "failure",
  budget_exhausted: "failure",
  stop_failed: "failure",
  deny: "failure",
  preempted: "stopped",
  cancelled: "stopped",
  rejected: "stopped",
  branch: "neutral",
  require_approval: "waiting",
};

const STYLE: Record<Tone, { className: string; icon: string }> = {
  running: { className: "bg-blue-50 text-blue-800 ring-blue-200", icon: "●" },
  success: { className: "bg-green-50 text-green-800 ring-green-200", icon: "✓" },
  failure: { className: "bg-red-50 text-red-800 ring-red-200", icon: "✕" },
  stopped: { className: "bg-gray-100 text-gray-700 ring-gray-300", icon: "■" },
  waiting: { className: "bg-amber-50 text-amber-900 ring-amber-200", icon: "◐" },
  neutral: { className: "bg-gray-50 text-gray-700 ring-gray-200", icon: "○" },
};

export function toneOf(status: string): Tone {
  return TONES[status.toLowerCase()] ?? "neutral";
}

export function Status({ value }: { value: string }) {
  const { className, icon } = STYLE[toneOf(value)];
  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset ${className}`}
    >
      <span aria-hidden="true">{icon}</span>
      {value}
    </span>
  );
}
