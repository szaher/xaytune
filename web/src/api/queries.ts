/**
 * One TanStack Query hook per read. The record is the only state (ADR-023 §S2):
 * nothing is cached past a query, and a view that is live polls (§S7).
 */
import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";
import { api, read } from "./client";

/** How often a live view polls while its experiment is not at rest (§S7). */
export const LIVE_INTERVAL_MS = 2_000;

const live = (isLive: boolean | undefined) => (isLive ? LIVE_INTERVAL_MS : false);

export const useMeta = () =>
  useQuery({
    queryKey: ["meta"],
    queryFn: () => read(api.GET("/api/v1/meta")),
    refetchInterval: 10_000,
  });

export const useExperiments = (query: { status?: string; before?: string; limit?: number }) =>
  useQuery({
    queryKey: ["experiments", query],
    queryFn: () => read(api.GET("/api/v1/experiments", { params: { query } })),
    placeholderData: keepPreviousData,
  });

const experiment = (experiment_id: string) => ({ params: { path: { experiment_id } } });

export const useExperiment = (id: string) =>
  useQuery({
    queryKey: ["experiment", id],
    queryFn: () => read(api.GET("/api/v1/experiments/{experiment_id}", experiment(id))),
    refetchInterval: (query) => live(query.state.data ? !query.state.data.quiescent : false),
  });

export const useGraph = (id: string, isLive?: boolean) =>
  useQuery({
    queryKey: ["experiment", id, "graph"],
    queryFn: () => read(api.GET("/api/v1/experiments/{experiment_id}/graph", experiment(id))),
    refetchInterval: live(isLive),
  });

export const useEvents = (id: string, after: number, limit: number, isLive?: boolean) =>
  useQuery({
    queryKey: ["experiment", id, "events", after, limit],
    queryFn: () =>
      read(
        api.GET("/api/v1/experiments/{experiment_id}/events", {
          params: { path: { experiment_id: id }, query: { after, limit } },
        }),
      ),
    refetchInterval: live(isLive),
    placeholderData: keepPreviousData,
  });

export const useActions = (id: string, isLive?: boolean) =>
  useQuery({
    queryKey: ["experiment", id, "actions"],
    queryFn: () => read(api.GET("/api/v1/experiments/{experiment_id}/actions", experiment(id))),
    refetchInterval: live(isLive),
  });

export const useDecisions = (id: string, isLive?: boolean) =>
  useQuery({
    queryKey: ["experiment", id, "decisions"],
    queryFn: () => read(api.GET("/api/v1/experiments/{experiment_id}/decisions", experiment(id))),
    refetchInterval: live(isLive),
  });

export const useBudget = (id: string, isLive?: boolean) =>
  useQuery({
    queryKey: ["experiment", id, "budget"],
    queryFn: () => read(api.GET("/api/v1/experiments/{experiment_id}/budget", experiment(id))),
    refetchInterval: live(isLive),
  });

export const useIncidents = (id: string, isLive?: boolean) =>
  useQuery({
    queryKey: ["experiment", id, "incidents"],
    queryFn: () => read(api.GET("/api/v1/experiments/{experiment_id}/incidents", experiment(id))),
    refetchInterval: live(isLive),
  });

export const useAgentInvocations = (id: string) =>
  useQuery({
    queryKey: ["experiment", id, "agent-invocations"],
    queryFn: () =>
      read(api.GET("/api/v1/experiments/{experiment_id}/agent-invocations", experiment(id))),
  });

export const useNode = (node_id: string, isLive?: boolean) =>
  useQuery({
    queryKey: ["node", node_id],
    queryFn: () => read(api.GET("/api/v1/nodes/{node_id}", { params: { path: { node_id } } })),
    refetchInterval: live(isLive),
  });

/** A run's statuses past which it gains no attempt and none of its attempts changes. */
const SETTLED_RUN = new Set(["succeeded", "failed", "cancelled"]);

/**
 * Whether a run is settled: a final status, and every attempt ended. Until
 * then the run page polls, so a recovery attempt started while it is open
 * appears without a reload (§S7).
 */
export const runSettled = (run: { status: string; attempts: { ended_at: string | null }[] }) =>
  SETTLED_RUN.has(run.status) && run.attempts.every((attempt) => attempt.ended_at !== null);

export const useRun = (run_id: string) =>
  useQuery({
    queryKey: ["run", run_id],
    queryFn: () => read(api.GET("/api/v1/runs/{run_id}", { params: { path: { run_id } } })),
    refetchInterval: (query) => live(query.state.data ? !runSettled(query.state.data.run) : false),
  });

/**
 * An attempt's record. *ended* is part of the key: when the polled run says
 * the attempt ended, its record is read once more, then never polled again.
 */
export const useAttempt = (attempt_id: string, ended = true) =>
  useQuery({
    queryKey: ["attempt", attempt_id, { ended }],
    queryFn: () =>
      read(api.GET("/api/v1/attempts/{attempt_id}", { params: { path: { attempt_id } } })),
    refetchInterval: live(!ended),
    placeholderData: keepPreviousData,
  });

export const useEvaluationRun = (evaluation_run_id: string) =>
  useQuery({
    queryKey: ["evaluation-run", evaluation_run_id],
    queryFn: () =>
      read(
        api.GET("/api/v1/evaluation-runs/{evaluation_run_id}", {
          params: { path: { evaluation_run_id } },
        }),
      ),
  });

export const useCompare = (left: string | undefined, right: string | undefined) =>
  useQuery({
    queryKey: ["compare", left, right],
    enabled: Boolean(left && right),
    queryFn: () =>
      read(
        api.GET("/api/v1/compare", {
          params: { query: { left: left ?? "", right: right ?? "" } },
        }),
      ),
  });

/**
 * Each attempt's recorded metric series (Track B). An attempt that has not
 * ended is polled; when it ends, *ended* changes the key, so its series is
 * read once more -- the points it reported last -- and polling stops.
 */
export const useAttemptMetrics = (attempts: { id: string; ended: boolean }[]) =>
  useQueries({
    queries: attempts.map(({ id, ended }) => ({
      queryKey: ["attempt", id, "metrics", { ended }],
      queryFn: () =>
        read(
          api.GET("/api/v1/attempts/{attempt_id}/metrics", {
            params: { path: { attempt_id: id } },
          }),
        ),
      refetchInterval: live(!ended),
      placeholderData: keepPreviousData,
    })),
  });
