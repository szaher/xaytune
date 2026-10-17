import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { type ChartLine, MetricChart, stepAxis, xOf } from "./chart";

const point = (step: number | null, value: number, sequence: number) => ({
  optimizer_step: step,
  value,
  stream_generation: 0,
  sequence,
});

describe("metric chart (Track B)", () => {
  it("draws one line per attempt, with every point", () => {
    const lines: ChartLine[] = [
      { label: "attempt 1", points: [point(1, 2.0, 0), point(2, 1.5, 1)] },
      { label: "attempt 2", points: [point(2, 1.4, 0), point(3, 1.1, 1), point(4, 0.9, 2)] },
    ];
    const { container } = render(<MetricChart title="training.loss" unit={null} lines={lines} />);
    const drawn = container.querySelectorAll("polyline");
    expect(drawn).toHaveLength(2);
    expect(drawn[1]?.getAttribute("points")?.split(" ")).toHaveLength(3);
    expect(screen.getByText(/training.loss by optimizer step/)).toBeInTheDocument();
  });

  it("never draws a missing step as step 0", () => {
    const lines: ChartLine[] = [{ label: "a", points: [point(null, 3, 0), point(5, 2, 1)] }];
    expect(stepAxis(lines)).toBe(false);
    const x = xOf(lines);
    expect(lines[0]?.points.map(x)).toEqual([0, 1]);
    render(<MetricChart title="training.loss" unit={null} lines={lines} />);
    expect(screen.getByText(/by position in the stream/)).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "training.loss points", hidden: true });
    expect(within(table).getByText("absent: no step reported")).toBeInTheDocument();
  });

  it("says nothing was recorded rather than drawing an empty axis", () => {
    const { container } = render(<MetricChart title="training.loss" unit={null} lines={[]} />);
    expect(container.querySelector("svg")).toBeNull();
    expect(screen.getByText("absent: no points recorded")).toBeInTheDocument();
  });

  it("keeps every recorded value in the table, a zero included", () => {
    const lines: ChartLine[] = [{ label: "a", points: [point(1, 0, 0), point(2, 0.25, 1)] }];
    render(<MetricChart title="training.loss" unit="nats" lines={lines} />);
    const table = screen.getByRole("table", { name: "training.loss points", hidden: true });
    expect(within(table).getAllByRole("row", { hidden: true })).toHaveLength(3);
    expect(within(table).getByText("0")).toBeInTheDocument();
    expect(screen.getByText(/\(nats\)/)).toBeInTheDocument();
  });
});
