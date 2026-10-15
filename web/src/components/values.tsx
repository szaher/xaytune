import { useState } from "react";

/**
 * An absent value: `—`, with its reason on hover and for screen readers.
 * The one place a missing value is rendered, and it never renders `0`
 * (ADR-023 S-F7, chapter 24 §5).
 */
export function Absent({ reason }: { reason?: string }) {
  const label = reason ? `absent: ${reason}` : "absent";
  return (
    <span className="text-gray-400" title={label} data-absent="true">
      <span aria-hidden="true">—</span>
      <span className="sr-only">{label}</span>
    </span>
  );
}

/** A number as recorded, or {@link Absent} when there is none. */
export function Num({
  value,
  reason,
  digits = 4,
}: {
  value: number | null | undefined;
  reason?: string;
  digits?: number;
}) {
  if (value === null || value === undefined) {
    return <Absent reason={reason} />;
  }
  return <span className="tabular-nums">{format(value, digits)}</span>;
}

export function format(value: number, digits = 4): string {
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toPrecision(digits)));
}

/** Free text from the record, always as text (ADR-023 §S10): never HTML or Markdown. */
export function Text({ value, reason }: { value: string | null | undefined; reason?: string }) {
  if (value === null || value === undefined || value === "") return <Absent reason={reason} />;
  return <span className="whitespace-pre-wrap break-words">{value}</span>;
}

/** An id or fingerprint, abbreviated, with the full value on hover and a copy button. */
export function ShortId({ value, keep = 10 }: { value: string; keep?: number }) {
  const [copied, setCopied] = useState(false);
  const short = value.length > keep + 1 ? `${value.slice(0, keep)}…` : value;
  return (
    <span className="inline-flex items-center gap-1 font-mono text-xs" title={value}>
      <span>{short}</span>
      <button
        type="button"
        className="rounded px-1 text-gray-400 hover:bg-gray-100 hover:text-gray-700"
        aria-label={`copy ${value}`}
        onClick={() => {
          void navigator.clipboard?.writeText(value).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1_000);
          });
        }}
      >
        {copied ? "✓" : "⧉"}
      </button>
    </span>
  );
}

/** A recorded timestamp: absolute in UTC, relative on hover. */
export function Time({ value }: { value: string | null | undefined }) {
  if (!value) return <Absent reason="not recorded" />;
  const date = new Date(value);
  return (
    <time dateTime={value} title={relative(date)} className="tabular-nums">
      {date.toISOString().replace("T", " ").slice(0, 19)}Z
    </time>
  );
}

export function relative(date: Date, now: Date = new Date()): string {
  const seconds = Math.round((now.getTime() - date.getTime()) / 1_000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.round(hours / 24)} days ago`;
}
