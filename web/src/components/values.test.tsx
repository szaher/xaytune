import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DocumentTree } from "./document";
import { toneOf } from "./status";
import { Absent, format, Num, Text } from "./values";

describe("absent values are never zero (ADR-023 S-F7)", () => {
  it.each([null, undefined])("renders %s as an em dash with its reason", (value) => {
    const { container } = render(<Num value={value} reason="not reported" />);
    expect(container.textContent).not.toContain("0");
    expect(container.querySelector("[data-absent]")).not.toBeNull();
    expect(screen.getByText("absent: not reported")).toBeInTheDocument();
  });

  it("renders a recorded zero as zero", () => {
    const { container } = render(<Num value={0} />);
    expect(container.textContent).toBe("0");
    expect(container.querySelector("[data-absent]")).toBeNull();
  });

  it("renders empty text as absent, not as an empty string", () => {
    const { container } = render(<Text value="" reason="not recorded" />);
    expect(container.querySelector("[data-absent]")).not.toBeNull();
  });

  it("keeps the reason available to assistive technology", () => {
    render(<Absent reason="withheld by aggregate disclosure" />);
    expect(screen.getByText("absent: withheld by aggregate disclosure")).toBeInTheDocument();
  });
});

describe("numbers are shown as recorded", () => {
  it.each([
    [0.83, "0.83"],
    [0.12345678, "0.1235"],
    [42, "42"],
    [-0.04, "-0.04"],
  ])("formats %s as %s", (value, shown) => {
    expect(format(value)).toBe(shown);
  });
});

describe("recorded free text is never markup (ADR-023 §S10)", () => {
  const hostile = '<img src=x onerror="alert(1)"><b>bold</b>';

  it("renders a hypothesis as text", () => {
    const { container } = render(<Text value={hostile} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    expect(container.textContent).toBe(hostile);
  });

  it("renders every leaf of a recorded document as text", () => {
    const { container } = render(
      <DocumentTree value={{ reason: hostile, nested: { deep: hostile } }} open={3} />,
    );
    expect(container.querySelector("img")).toBeNull();
    expect(container.textContent).toContain(hostile);
  });
});

describe("statuses keep their recorded names", () => {
  it("knows the families", () => {
    expect(toneOf("succeeded")).toBe("success");
    expect(toneOf("approval_pending")).toBe("waiting");
    expect(toneOf("failed")).toBe("failure");
  });

  it("shows an unknown status from a newer server as neutral, not as an error", () => {
    expect(toneOf("some_future_status")).toBe("neutral");
  });
});
