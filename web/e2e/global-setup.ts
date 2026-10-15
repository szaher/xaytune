import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { PYTHON, RECORD_FILE, REPOSITORY } from "./studio";

/** Record the real experiment once; every test reads the same file, read-only. */
export default function globalSetup(): void {
  const directory = mkdtempSync(path.join(tmpdir(), "xaytune-studio-e2e-"));
  const output = execFileSync(PYTHON, ["-m", "tests.studio_record", directory], {
    cwd: REPOSITORY,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
  const state = output.trim().split("\n").at(-1) ?? "";
  writeFileSync(RECORD_FILE, JSON.stringify({ state }));
}
