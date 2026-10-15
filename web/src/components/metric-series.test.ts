import { describe, expect, it } from "vitest";
import {
  chartLines,
  type RecordedSeries,
  type SeriesChoice,
  seriesChoices,
  seriesKey,
} from "./metric-series";

const series = (
  name: string,
  unit: string | null,
  values: number[],
  extra: Partial<RecordedSeries> = {},
): RecordedSeries => ({
  name,
  unit,
  worker: null,
  rank: null,
  downsampled: false,
  points: values.map((value, sequence) => ({
    optimizer_step: sequence + 1,
    value,
    stream_generation: 0,
    sequence,
  })),
  ...extra,
});

describe("metric series choices (Track B)", () => {
  it("offers one name in two units as two series, and never mixes their values", () => {
    const byAttempt = [
      [series("custom.duration", "ms", [1000, 1500]), series("custom.duration", "s", [1])],
      [series("custom.duration", "s", [2, 3])],
    ];
    const choices = seriesChoices(byAttempt);
    expect(choices.map((choice) => choice.label)).toEqual([
      "custom.duration (ms)",
      "custom.duration (s)",
    ]);
    const [ms, s] = choices as [SeriesChoice, SeriesChoice];
    const drawn = (choice: SeriesChoice) =>
      chartLines(byAttempt, [1, 2], choice).map((line) => [
        line.label,
        line.points.map((p) => p.value),
      ]);
    expect(drawn(ms)).toEqual([["attempt 1", [1000, 1500]]]);
    expect(drawn(s)).toEqual([
      ["attempt 1", [1]],
      ["attempt 2", [2, 3]],
    ]);
  });

  it("orders by name, a unitless series first, and keys cannot collide", () => {
    const choices = seriesChoices([
      [series("training.loss", null, [1]), series("a", "s", [1]), series("a", null, [1])],
    ]);
    expect(choices.map((choice) => choice.label)).toEqual(["a", "a (s)", "training.loss"]);
    expect(seriesKey({ name: "a (s)", unit: null })).not.toBe(seriesKey({ name: "a", unit: "s" }));
  });

  it("keeps ranks apart within one series", () => {
    const byAttempt = [[series("x", "s", [1], { rank: 0 }), series("x", "s", [2], { rank: 1 })]];
    const [choice] = seriesChoices(byAttempt) as [SeriesChoice];
    expect(chartLines(byAttempt, [1], choice).map((line) => line.label)).toEqual([
      "attempt 1 rank 0",
      "attempt 1 rank 1",
    ]);
  });
});
