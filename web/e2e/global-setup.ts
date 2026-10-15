import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { PYTHON, RECORD_FILE, REPOSITORY } from "./studio";

/**
 * Record the real experiments once; every test reads the same files, read-only:
 * the spec 18 adaptive scenario (UX-02) and one real native training run (Track B).
 */
export default function globalSetup(): void {
  const directory = mkdtempSync(path.join(tmpdir(), "xaytune-studio-e2e-"));
  const state = record([directory]);
  const training = record(["--training", path.join(directory, "training")]);
  writeFileSync(RECORD_FILE, JSON.stringify({ state, training }));
}

/** Run `python -m tests.studio_record ARGS`; the state database it printed last. */
function record(args: string[]): string {
  const output = execFileSync(PYTHON, ["-m", "tests.studio_record", ...args], {
    cwd: REPOSITORY,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
  return output.trim().split("\n").at(-1) ?? "";
}
