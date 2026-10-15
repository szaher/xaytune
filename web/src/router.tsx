/**
 * Every view is addressable by URL, filters and selections included (ADR-023
 * §S11, chapter 24 §3): the URL plus the record is the whole state, so a
 * reload or a restart shows the same thing.
 */
import { createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { CandidatePage } from "./pages/candidate";
import { ComparePage } from "./pages/compare";
import { EvaluationRunPage } from "./pages/evaluation-run";
import { EXPERIMENT_TABS, ExperimentPage, type ExperimentTab } from "./pages/experiment";
import { ExperimentsPage } from "./pages/experiments";
import { NotFound } from "./pages/not-found";
import { OverviewPage } from "./pages/overview";
import { RunPage } from "./pages/run";
import { WaitingPage } from "./pages/waiting";
import { Shell } from "./shell";

const text = (value: unknown): string | undefined =>
  typeof value === "string" && value !== "" ? value : undefined;

const rootRoute = createRootRoute({ component: Shell, notFoundComponent: NotFound });

const overview = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: OverviewPage,
});

export const experimentsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/experiments",
  validateSearch: (search: Record<string, unknown>): { status?: string; before?: string } => ({
    status: text(search.status),
    before: text(search.before),
  }),
  component: ExperimentsPage,
});

export const experimentRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/experiments/$experimentId",
  validateSearch: (
    search: Record<string, unknown>,
  ): { tab?: ExperimentTab; node?: string; after?: number } => {
    const tab = text(search.tab);
    const after = Number(search.after);
    return {
      tab: EXPERIMENT_TABS.includes(tab as ExperimentTab) ? (tab as ExperimentTab) : undefined,
      node: text(search.node),
      after: Number.isInteger(after) && after > 0 ? after : undefined,
    };
  },
  component: ExperimentPage,
});

export const candidateRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/experiments/$experimentId/candidates/$nodeId",
  component: CandidatePage,
});

export const runRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  component: RunPage,
});

export const evaluationRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/evaluation-runs/$evaluationRunId",
  component: EvaluationRunPage,
});

export const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  validateSearch: (search: Record<string, unknown>): { left?: string; right?: string } => ({
    left: text(search.left),
    right: text(search.right),
  }),
  component: ComparePage,
});

const waiting = (path: "/monitoring" | "/trajectories" | "/optimization") =>
  createRoute({
    getParentRoute: () => rootRoute,
    path,
    component: () => <WaitingPage section={path.slice(1) as "monitoring"} />,
  });

const routeTree = rootRoute.addChildren([
  overview,
  experimentsRoute,
  experimentRoute,
  candidateRoute,
  runRoute,
  evaluationRunRoute,
  compareRoute,
  waiting("/monitoring"),
  waiting("/trajectories"),
  waiting("/optimization"),
]);

export const router = createRouter({ routeTree, defaultPreload: false });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
