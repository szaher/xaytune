import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ApiProblem } from "./api/client";
import { router } from "./router";
import "./styles.css";

const queries = new QueryClient({
  defaultOptions: {
    queries: {
      // A 404 or a refused store is an answer, not a blip: do not retry it.
      retry: (count, error) => !(error instanceof ApiProblem && error.status < 500) && count < 2,
      refetchOnWindowFocus: true,
      staleTime: 1_000,
    },
  },
});

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <QueryClientProvider client={queries}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </StrictMode>,
  );
}
