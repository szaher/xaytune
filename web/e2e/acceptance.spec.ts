/**
 * UX-02 acceptance: open a real recorded experiment in the browser, inspect
 * its lineage, evaluation and recovery, restart Studio, and see the same data.
 *
 * The experiment is the spec 18 adaptive scenario, recorded by a real
 * controller host: node A (LoRA 16) trains, hits a CUDA OOM, recovers on a
 * second attempt with a resized micro-batch, scores 0.79 and branches; the
 * planner proposes node B (LoRA 32), which scores 0.83 and succeeds.
 */
import { expect, test } from "@playwright/test";
import { openOnlyExperiment, recorded, Studio, watchConsole } from "./studio";

const PORT = 18_771;

test("a recorded experiment survives a Studio restart, lineage, evaluation and recovery included", async ({
  page,
}) => {
  const problems = watchConsole(page);
  const studio = new Studio(recorded().state, PORT);
  await studio.start();
  try {
    // Explorer → experiment.
    const experiment = await openOnlyExperiment(page, studio);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("tiny-sft");
    await expect(page.getByText("succeeded").first()).toBeVisible();

    // Lineage: two candidates, root → child, the child the best.
    await page.getByRole("tab", { name: "candidates" }).click();
    const candidates = page.getByRole("table", { name: "Candidates" });
    await expect(candidates.getByRole("row")).toHaveCount(3);
    await expect(candidates.getByText("root")).toBeVisible();
    await expect(candidates.getByText("stop_succeeded")).toBeVisible();
    await expect(candidates.getByText("branch")).toBeVisible();
    await expect(page.getByRole("group", { name: "Candidate graph" })).toBeVisible();
    const before = await lineageAndResults(page);

    // Evaluation: the child's result, and the comparison with its parent.
    await candidates.getByRole("button", { name: "select" }).nth(1).click();
    await expect(page).toHaveURL(/node=/);
    await page.getByRole("link", { name: "Compare with parent" }).click();
    const comparison = page.getByRole("table", { name: "Metrics compared" });
    await expect(comparison.getByText("task_success")).toBeVisible();
    await expect(comparison.getByText("0.83")).toBeVisible();
    await expect(comparison.getByText("0.79")).toBeVisible();
    await expect(page.getByTestId("difference-task_success")).toContainText("+0.04");
    await expect(page.getByTestId("difference-task_success")).toContainText("same evaluation");
    await expect(page.getByText("candidate.training.adapter.rank")).toBeVisible();
    const compareUrl = page.url();

    // Recovery: the parent's run, its OOM and the resized second attempt.
    await page.goto(`${studio.url}${experiment}?tab=runs`);
    const runs = page.getByRole("table", { name: "Training runs" });
    await expect(runs.getByRole("row")).toHaveCount(3);
    await runs.getByRole("row").filter({ hasText: "2" }).getByRole("link").first().click();
    await expect(page.getByRole("group", { name: "attempt 1" })).toContainText("failed");
    await expect(page.getByRole("group", { name: "attempt 1" })).toContainText("CUDA_OOM");
    await expect(page.getByRole("group", { name: "attempt 2" })).toContainText("succeeded");
    await expect(page.getByRole("group", { name: "Overrides" })).toBeVisible();
    const runUrl = page.url();
    const runText = await page.getByRole("main").innerText();

    // Restart Studio: a new process over the same file.
    await studio.stop();
    await studio.start();

    await page.goto(runUrl);
    await expect(page.getByRole("group", { name: "attempt 1" })).toContainText("CUDA_OOM");
    expect(await page.getByRole("main").innerText()).toBe(runText);

    await page.goto(compareUrl);
    await expect(page.getByTestId("difference-task_success")).toContainText("+0.04");

    await page.goto(`${studio.url}${experiment}?tab=candidates`);
    await expect(page.getByRole("table", { name: "Candidates" }).getByRole("row")).toHaveCount(3);
    expect(await lineageAndResults(page)).toEqual(before);

    expect(problems).toEqual([]);
  } finally {
    await studio.stop();
  }
});

/** What the candidates table shows: identity, status, decision, lineage. */
async function lineageAndResults(page: import("@playwright/test").Page): Promise<string[]> {
  return page.getByRole("table", { name: "Candidates" }).getByRole("row").allInnerTexts();
}
