import { defineConfig, devices } from "@playwright/test";

// The browser tests drive the built client against a real recorded state
// database (UX-02 acceptance): e2e/studio.ts records it once and starts
// `xaytune studio-web` itself, so a test can restart Studio mid-test.
export default defineConfig({
  testDir: "e2e",
  globalSetup: "./e2e/global-setup.ts",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 60_000,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: { ...devices["Desktop Chrome"], viewport: { width: 1400, height: 1000 }, trace: "retain-on-failure" },
});
