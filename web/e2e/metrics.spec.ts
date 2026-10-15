/**
 * Track B acceptance: a real training run's loss history is visible in Studio
 * and survives a Studio restart.
 *
 * The run is recorded by `python -m tests.studio_record --training`: a native
 * worker trains a tiny causal LM for 24 optimizer steps under the local
 * runtime, and one controller host submits it and closes while it trains, so
 * a second host adopts the workload and replays its telemetry. Every point
 * reached the metric-point store (migration 020) through the controller.
 */
import { expect, type Page, test } from "@playwright/test";
import { openOnlyExperiment, recorded, Studio, watchConsole } from "./studio";

const PORT = 18_773;
const STEPS = 24;

test("a real training run's loss history survives a Studio restart", async ({ page }) => {
  const problems = watchConsole(page);
  const studio = new Studio(recorded().training, PORT);
  await studio.start();
  try {
    const experiment = await openOnlyExperiment(page, studio);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("tiny-sft-metrics");
    await page.goto(`${studio.url}${experiment}?tab=runs`);
    await page.getByRole("table", { name: "Training runs" }).getByRole("link").first().click();
    await expect(page).toHaveURL(/\/runs\//);

    const before = await lossHistory(page);
    expect(before.rows).toHaveLength(STEPS + 1);
    expect(before.rows[1]).toMatch(/^attempt 1\t1\t/);
    expect(before.rows.at(-1)).toMatch(new RegExp(`^attempt 1\\t${STEPS}\\t`));
    const runUrl = page.url();

    // The API serves the same series the page drew.
    const attemptId = await page
      .getByRole("group", { name: "attempt 1" })
      .getAttribute("data-attempt");
    const api = await page.request.get(
      `${studio.url}/api/v1/attempts/${attemptId}/metrics?name=training.loss`,
    );
    const body = await api.json();
    expect(body.series).toHaveLength(1);
    expect(body.series[0].count).toBe(STEPS);

    // Restart Studio: a new process over the same file.
    await studio.stop();
    await studio.start();
    await page.goto(runUrl);
    expect(await lossHistory(page)).toEqual(before);

    expect(problems).toEqual([]);
  } finally {
    await studio.stop();
  }
});

/** The loss chart's legend and every point in its table, as the page shows them. */
async function lossHistory(page: Page): Promise<{ legend: string; rows: string[] }> {
  const metrics = page.getByRole("region", { name: "Metrics" });
  await expect(metrics.getByTestId("metric-chart")).toBeVisible();
  await expect(metrics.getByRole("combobox")).toHaveValue(JSON.stringify(["training.loss", null]));
  const legend = await metrics.getByRole("list", { name: "Legend" }).innerText();
  expect(legend).toContain(`attempt 1 (${STEPS} points)`);
  await metrics.getByText("Points as a table").click();
  const rows = await metrics
    .getByRole("table", { name: "training.loss points" })
    .getByRole("row")
    .allInnerTexts();
  await metrics.getByText("Points as a table").click();
  return { legend, rows };
}
