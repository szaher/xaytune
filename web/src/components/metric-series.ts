/**
 * Which recorded series a run's chart can show, and the lines of one of them.
 *
 * A series is a name *and* its unit (Track B, migration 020): a worker that
 * reported `custom.duration` in ms and then in s recorded two series, and a
 * chart of one must never draw the other's values on its axis.
 */
import type { ChartLine, ChartPoint } from "./chart";

export type RecordedSeries = {
  name: string;
  unit: string | null;
  worker: string | null;
  rank: number | null;
  downsampled: boolean;
  points: ChartPoint[];
};

export type SeriesChoice = { key: string; name: string; unit: string | null; label: string };

/** A choice's identity: name and unit together, unambiguous whatever the name holds. */
export const seriesKey = (series: { name: string; unit: string | null }) =>
  JSON.stringify([series.name, series.unit]);

/** Every distinct (name, unit) any attempt recorded, by name, then unitless first, then unit. */
export function seriesChoices(byAttempt: RecordedSeries[][]): SeriesChoice[] {
  const found = new Map<string, SeriesChoice>();
  for (const series of byAttempt.flat()) {
    const key = seriesKey(series);
    if (!found.has(key)) {
      found.set(key, {
        key,
        name: series.name,
        unit: series.unit,
        label: series.unit === null ? series.name : `${series.name} (${series.unit})`,
      });
    }
  }
  const compare = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
  const unitless = (one: SeriesChoice) => (one.unit === null ? 0 : 1);
  return [...found.values()].sort(
    (a, b) =>
      compare(a.name, b.name) || unitless(a) - unitless(b) || compare(a.unit ?? "", b.unit ?? ""),
  );
}

/** One line per attempt (and per rank or worker) of exactly the chosen name and unit. */
export function chartLines(
  byAttempt: RecordedSeries[][],
  attemptNumbers: number[],
  choice: SeriesChoice,
): ChartLine[] {
  return byAttempt.flatMap((series, index) =>
    series
      .filter((one) => seriesKey(one) === choice.key)
      .map((one) => ({
        label:
          `attempt ${attemptNumbers[index] ?? index + 1}` +
          (one.rank === null ? "" : ` rank ${one.rank}`) +
          (one.worker === null ? "" : ` ${one.worker}`),
        points: one.points,
      })),
  );
}
