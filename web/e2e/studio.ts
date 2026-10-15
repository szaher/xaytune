/**
 * A real Studio over a real recorded experiment, for the browser tests.
 *
 * The state database is recorded by `python -m tests.studio_record` (the spec 18
 * adaptive scenario, through a real controller host). Studio is the real
 * `xaytune studio-web` process serving the built client; tests start, stop
 * and restart it themselves.
 */
import { type ChildProcess, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page } from "@playwright/test";

export const REPOSITORY = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
export const PYTHON = process.env.XAYTUNE_PYTHON ?? "python";
export const RECORD_FILE = path.join(tmpdir(), "xaytune-studio-e2e.json");

export type Recorded = { state: string };

export function recorded(): Recorded {
  return JSON.parse(readFileSync(RECORD_FILE, "utf8")) as Recorded;
}

export class Studio {
  private process: ChildProcess | undefined;

  constructor(
    readonly state: string,
    readonly port: number,
  ) {}

  get url(): string {
    return `http://127.0.0.1:${this.port}`;
  }

  async start(): Promise<void> {
    this.process = spawn(
      PYTHON,
      ["-m", "xaytune", "studio-web", "--state", this.state, "--port", String(this.port)],
      { cwd: REPOSITORY, stdio: ["ignore", "ignore", "pipe"] },
    );
    let stderr = "";
    this.process.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      if (this.process.exitCode !== null) {
        throw new Error(`xaytune studio-web exited (${this.process.exitCode}):\n${stderr}`);
      }
      try {
        const response = await fetch(`${this.url}/api/v1/meta`);
        if (response.ok) return;
      } catch {
        // not listening yet
      }
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    throw new Error(`xaytune studio-web did not start:\n${stderr}`);
  }

  async stop(): Promise<void> {
    const child = this.process;
    if (!child || child.exitCode !== null) return;
    const exited = new Promise((resolve) => child.once("exit", resolve));
    child.kill("SIGTERM");
    await exited;
    this.process = undefined;
  }
}

/** Fails the test on any console error, uncaught exception or CSP violation. */
export function watchConsole(page: Page): string[] {
  const problems: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(message.text());
  });
  page.on("pageerror", (error) => problems.push(error.message));
  return problems;
}

export async function openOnlyExperiment(page: Page, studio: Studio): Promise<string> {
  await page.goto(`${studio.url}/experiments`);
  const link = page.getByRole("table", { name: "Experiments" }).getByRole("link").first();
  await expect(link).toBeVisible();
  const href = (await link.getAttribute("href")) ?? "";
  await link.click();
  await expect(page).toHaveURL(new RegExp(`${href}$`));
  return href;
}
