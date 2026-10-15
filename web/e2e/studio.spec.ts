import { expect, test } from "@playwright/test";
import { openOnlyExperiment, recorded, Studio, watchConsole } from "./studio";

const studio = new Studio(recorded().state, 18_772);

test.beforeAll(async () => studio.start());
test.afterAll(async () => studio.stop());

test("every section loads without a console error or a CSP violation", async ({ page }) => {
  const problems = watchConsole(page);
  for (const path of [
    "/",
    "/experiments",
    "/compare",
    "/monitoring",
    "/trajectories",
    "/optimization",
  ]) {
    await page.goto(`${studio.url}${path}`);
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  }
  const experiment = await openOnlyExperiment(page, studio);
  for (const tab of [
    "overview",
    "candidates",
    "runs",
    "decisions",
    "actions",
    "activity",
    "budget",
    "configuration",
  ]) {
    await page.goto(`${studio.url}${experiment}?tab=${tab}`);
    await expect(page.getByRole("tab", { name: tab })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("tabpanel")).toBeVisible();
  }
  expect(problems).toEqual([]);
});

test("a candidate can be selected in the graph from the keyboard", async ({ page }) => {
  const experiment = await openOnlyExperiment(page, studio);
  await page.goto(`${studio.url}${experiment}?tab=candidates`);
  const graph = page.getByRole("group", { name: "Candidate graph" });
  const best = graph.getByRole("button", { name: /best/ });
  await best.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/node=node_/);
  await expect(page.getByRole("group", { name: "Selected candidate" })).toContainText("best");
});

test("the URL is the state: a reload shows the same selection and tab", async ({ page }) => {
  const experiment = await openOnlyExperiment(page, studio);
  await page.goto(`${studio.url}${experiment}?tab=candidates`);
  await page
    .getByRole("table", { name: "Candidates" })
    .getByRole("button", { name: "select" })
    .first()
    .click();
  const url = page.url();
  await page.reload();
  expect(page.url()).toBe(url);
  await expect(page.getByRole("group", { name: "Selected candidate" })).toBeVisible();
});

test("the event journal pages by sequence", async ({ page }) => {
  const experiment = await openOnlyExperiment(page, studio);
  await page.goto(`${studio.url}${experiment}?tab=activity`);
  const events = page.getByRole("table", { name: "Events" });
  await expect(events.getByRole("row").nth(1)).toContainText("ExperimentCreated");
  await expect(page.getByRole("group", { name: "incident CUDA_OOM" })).toBeVisible();
});

test("an unknown address and an unknown record say so", async ({ page }) => {
  await page.goto(`${studio.url}/no/such/view`);
  await expect(page.getByRole("heading", { name: "Not found" })).toBeVisible();
  await page.goto(`${studio.url}/runs/run_does_not_exist`);
  await expect(page.getByRole("alert")).toContainText("no such run");
});

test("the API's own front door still guards the served client", async ({ request }) => {
  const response = await request.get(`${studio.url}/`);
  expect(response.headers()["content-security-policy"]).toContain("default-src 'self'");
  expect((await request.post(`${studio.url}/api/v1/experiments`)).status()).toBe(405);
  expect((await request.get(`${studio.url}/api/v1/no-such`)).status()).toBe(404);
});
