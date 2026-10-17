/**
 * A Run page left open sees a recovery attempt start, and stops polling once
 * the run settles (Track B review of #87).
 *
 * `python -m tests.studio_record --live` runs the spec 18 scenario through a
 * real controller host while the browser watches: attempt 1 reports four
 * losses and is held running until this test creates `release`; it then hits
 * a CUDA OOM, and the controller starts recovery attempt 2, which reports
 * four more. Nobody reloads the page.
 */
import { type ChildProcess, spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { expect, test } from "@playwright/test";
import { openOnlyExperiment, PYTHON, REPOSITORY, Studio, watchConsole } from "./studio";

const PORT = 18_774;

test("an open Run page discovers the recovery attempt and stops polling once settled", async ({
  page,
}) => {
  test.setTimeout(180_000);
  const problems = watchConsole(page);
  const directory = mkdtempSync(path.join(tmpdir(), "xaytune-studio-live-"));
  const controller = spawn(PYTHON, ["-m", "tests.studio_record", "--live", directory], {
    cwd: REPOSITORY,
    stdio: ["ignore", "pipe", "inherit"],
  });
  const exited = new Promise<number | null>((resolve) => controller.once("exit", resolve));
  let studio: Studio | undefined;
  try {
    studio = new Studio(await ready(controller), PORT);
    await studio.start();
    await openOnlyExperiment(page, studio);
    await page.goto(`${page.url()}?tab=runs`);
    await page.getByRole("table", { name: "Training runs" }).getByRole("link").first().click();
    await expect(page).toHaveURL(/\/runs\//);

    // Attempt 1 is training: its four points, and no other attempt.
    const metrics = page.getByRole("region", { name: "Metrics" });
    const legend = metrics.getByRole("list", { name: "Legend" });
    await expect(legend).toHaveText("attempt 1 (4 points)");
    await expect(page.getByRole("group", { name: /^attempt \d+$/ })).toHaveCount(1);

    // The OOM, then the recovery attempt -- on the page as it is, without a reload.
    writeFileSync(path.join(directory, "release"), "");
    await expect(page.getByRole("group", { name: "attempt 2" })).toBeVisible({ timeout: 30_000 });
    await expect(legend).toContainText("attempt 1 (4 points)");
    await expect(legend).toContainText("attempt 2 (4 points)", { timeout: 30_000 });
    await expect(
      page.getByRole("group", { name: "attempt 1" }).getByRole("group", { name: "Incidents" }),
    ).toBeVisible();
    await metrics.getByText("Points as a table").click();
    const steps = await metrics
      .getByRole("table", { name: "training.loss points" })
      .getByRole("row")
      .allInnerTexts();
    expect(steps.slice(1).map((row) => row.split("\t").slice(0, 2).join(" "))).toEqual([
      "attempt 1 1",
      "attempt 1 2",
      "attempt 1 3",
      "attempt 1 4",
      "attempt 2 5",
      "attempt 2 6",
      "attempt 2 7",
      "attempt 2 8",
    ]);

    // Settled: the run succeeded and both attempts ended. Then nothing on this page polls.
    expect(await exited).toBe(0);
    await expect(page.getByRole("group", { name: "attempt 1" })).toContainText("failed");
    await expect(page.getByRole("group", { name: "attempt 2" })).toContainText("succeeded", {
      timeout: 15_000,
    });
    await page.waitForTimeout(3_000);
    const polled: string[] = [];
    page.on("request", (request) => {
      if (/\/api\/v1\/(runs|attempts)\//.test(request.url())) polled.push(request.url());
    });
    await page.waitForTimeout(6_000);
    expect(polled).toEqual([]);

    expect(problems).toEqual([]);
  } finally {
    controller.kill("SIGTERM");
    await studio?.stop();
  }
});

/** The state database the live controller prints once attempt 1's points are recorded. */
function ready(controller: ChildProcess): Promise<string> {
  return new Promise((resolve, reject) => {
    let output = "";
    controller.stdout?.on("data", (chunk: Buffer) => {
      output += chunk.toString();
      const line = output.split("\n").find((one) => one.startsWith("ready "));
      if (line) resolve(line.slice("ready ".length).trim());
    });
    controller.once("exit", (code) => reject(new Error(`the live controller exited (${code})`)));
  });
}
