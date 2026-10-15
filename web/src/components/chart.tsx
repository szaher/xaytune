/**
 * A line chart of recorded metric series (Track B, migration 020).
 *
 * Plain SVG: a few hundred points per line need no charting library. The x
 * axis is the optimizer step when every point has one, and the position in
 * the stream otherwise -- a missing step is never drawn as step 0. The same
 * points are always available as a table, which is also what assistive
 * technology reads.
 */
import { useId } from "react";
import { Absent, format } from "./values";

export type ChartPoint = {
  optimizer_step: number | null;
  value: number;
  stream_generation: number;
  sequence: number;
  event_id?: string;
};

export type ChartLine = { label: string; points: ChartPoint[] };

const WIDTH = 640;
const HEIGHT = 240;
const PAD = { top: 12, right: 16, bottom: 32, left: 64 };
const COLORS = ["#2563eb", "#d97706", "#059669", "#db2777", "#7c3aed", "#0891b2"];

/** Whether every point of every line has a step, so steps can be the x axis. */
export function stepAxis(lines: ChartLine[]): boolean {
  return lines.every((line) => line.points.every((point) => point.optimizer_step !== null));
}

/** Each point's x: its step, or its index in stream order when steps are not all known. */
export function xOf(lines: ChartLine[]): (point: ChartPoint, index: number) => number {
  return stepAxis(lines) ? (point) => point.optimizer_step as number : (_, index) => index;
}

function extent(values: number[]): [number, number] {
  let low = Math.min(...values);
  let high = Math.max(...values);
  if (low === high) {
    low -= Math.abs(low) * 0.05 || 1;
    high += Math.abs(high) * 0.05 || 1;
  }
  return [low, high];
}

export function MetricChart({
  title,
  unit,
  lines,
}: {
  title: string;
  unit: string | null;
  lines: ChartLine[];
}) {
  const id = useId();
  const drawn = lines.filter((line) => line.points.length > 0);
  if (drawn.length === 0) return <Absent reason="no points recorded" />;
  const x = xOf(drawn);
  const xs = drawn.flatMap((line) => line.points.map(x));
  const ys = drawn.flatMap((line) => line.points.map((point) => point.value));
  const [x0, x1] = extent(xs);
  const [y0, y1] = extent(ys);
  const sx = (value: number) =>
    PAD.left + ((value - x0) / (x1 - x0)) * (WIDTH - PAD.left - PAD.right);
  const sy = (value: number) =>
    HEIGHT - PAD.bottom - ((value - y0) / (y1 - y0)) * (HEIGHT - PAD.top - PAD.bottom);
  const axis = stepAxis(drawn) ? "optimizer step" : "position in the stream";
  return (
    <figure className="space-y-2" aria-labelledby={`${id}-caption`}>
      <figcaption id={`${id}-caption`} className="text-sm text-gray-600">
        {title}
        {unit ? ` (${unit})` : ""} by {axis}
      </figcaption>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-auto w-full max-w-3xl rounded border border-gray-200 bg-white"
        aria-hidden="true"
        data-testid="metric-chart"
      >
        <line
          x1={PAD.left}
          y1={HEIGHT - PAD.bottom}
          x2={WIDTH - PAD.right}
          y2={HEIGHT - PAD.bottom}
          stroke="#9ca3af"
        />
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={HEIGHT - PAD.bottom} stroke="#9ca3af" />
        <text x={PAD.left - 6} y={PAD.top + 4} textAnchor="end" fontSize="11" fill="#4b5563">
          {format(y1)}
        </text>
        <text
          x={PAD.left - 6}
          y={HEIGHT - PAD.bottom}
          textAnchor="end"
          fontSize="11"
          fill="#4b5563"
        >
          {format(y0)}
        </text>
        <text x={PAD.left} y={HEIGHT - 10} fontSize="11" fill="#4b5563">
          {format(x0)}
        </text>
        <text x={WIDTH - PAD.right} y={HEIGHT - 10} textAnchor="end" fontSize="11" fill="#4b5563">
          {format(x1)}
        </text>
        {drawn.map((line, lineIndex) => (
          <polyline
            key={line.label}
            data-line={line.label}
            fill="none"
            stroke={COLORS[lineIndex % COLORS.length]}
            strokeWidth="1.5"
            points={line.points
              .map(
                (point, index) => `${sx(x(point, index)).toFixed(1)},${sy(point.value).toFixed(1)}`,
              )
              .join(" ")}
          />
        ))}
      </svg>
      <ul className="flex flex-wrap gap-4 text-xs text-gray-600" aria-label="Legend">
        {drawn.map((line, lineIndex) => (
          <li key={line.label} className="flex items-center gap-1">
            <span
              aria-hidden="true"
              className="inline-block h-2 w-4 rounded"
              style={{ backgroundColor: COLORS[lineIndex % COLORS.length] }}
            />
            {line.label} ({line.points.length} points)
          </li>
        ))}
      </ul>
      <details>
        <summary className="cursor-pointer text-sm text-gray-600">Points as a table</summary>
        <table className="mt-2 text-sm" aria-label={`${title} points`}>
          <thead>
            <tr className="text-left text-gray-500">
              <th className="pr-4 font-normal">Series</th>
              <th className="pr-4 font-normal">Step</th>
              <th className="pr-4 font-normal">Value</th>
            </tr>
          </thead>
          <tbody>
            {drawn.flatMap((line) =>
              line.points.map((point) => (
                <tr key={`${line.label}-${point.stream_generation}-${point.sequence}`}>
                  <td className="pr-4">{line.label}</td>
                  <td className="pr-4 tabular-nums">
                    {point.optimizer_step === null ? (
                      <Absent reason="no step reported" />
                    ) : (
                      point.optimizer_step
                    )}
                  </td>
                  <td className="pr-4 tabular-nums">{format(point.value, 6)}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
