/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The bundle is written into the Python package, where `xaytune studio-web`
// serves it from the API's own origin (ADR-023 §S2, §S11).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../xaytune/api/static",
    emptyOutDir: true,
    sourcemap: false,
    // ELK's layout engine is one 1.4 MB chunk, loaded only with the first graph.
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5173,
    strictPort: true,
    // No CORS on the API: the dev server forwards /api to it instead.
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
