import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Schemas } from "../api/client";
import { MetricTable } from "./compare";

const metric = (value: number): Schemas["MetricOut"] => ({
  name: "acc",
  value,
  sample_count: null,
  dataset_ref: null,
  slice: null,
  evaluator_name: "native",
  evaluator_version: null,
  seed: null,
  confidence_interval: null,
  standard_error: null,
  metadata: {},
});

const side = (value: number, fingerprint: string): Schemas["MetricSide"] => ({
  metric: metric(value),
  result_id: `result-${fingerprint}`,
  evaluation_fingerprint: fingerprint,
  unit: null,
});

const pair = (overrides: Partial<Schemas["MetricPair"]>): Schemas["MetricPair"] => ({
  evaluator_name: "native",
  name: "acc",
  slice: null,
  left: side(0.83, "sha256:same"),
  right: side(0.79, "sha256:same"),
  comparable: true,
  reasons: [],
  difference: 0.04,
  ...overrides,
});

describe("the comparison subtracts only what the API says is comparable (ADR-023 §S6)", () => {
  it("shows the difference for a pair measured under one evaluation", () => {
    render(<MetricTable pairs={[pair({})]} />);
    const cell = screen.getByTestId("difference-acc");
    expect(cell.textContent).toContain("+0.04");
    expect(cell.textContent).toContain("same evaluation");
  });

  it("never shows a number for a pair under different evaluations", () => {
    render(
      <MetricTable
        pairs={[
          pair({
            right: side(0.79, "sha256:other"),
            comparable: false,
            difference: null,
            reasons: ["measured under different evaluations"],
          }),
        ]}
      />,
    );
    const cell = screen.getByTestId("difference-acc");
    expect(cell.textContent).toContain("not comparable");
    expect(cell.textContent).toContain("measured under different evaluations");
    expect(cell.textContent).not.toMatch(/\d/);
  });

  it("shows a side measured on one candidate only as absent", () => {
    render(
      <MetricTable
        pairs={[
          pair({
            right: null,
            comparable: false,
            difference: null,
            reasons: ["measured on one candidate only"],
          }),
        ]}
      />,
    );
    const row = screen.getByTestId("difference-acc").closest("tr") as HTMLElement;
    expect(within(row).getByText("absent: not measured")).toBeInTheDocument();
  });
});
