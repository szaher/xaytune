import type { ReactNode } from "react";
import { ApiProblem } from "../api/client";

export function Page({
  title,
  subtitle,
  crumbs,
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  crumbs?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
      {crumbs ? (
        <nav aria-label="Breadcrumb" className="text-sm text-gray-500">
          {crumbs}
        </nav>
      ) : null}
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold text-gray-900">{title}</h1>
        {subtitle ? <div className="text-sm text-gray-600">{subtitle}</div> : null}
      </header>
      {children}
    </div>
  );
}

export function Section({
  title,
  children,
  aside,
}: {
  title: string;
  children: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <section className="space-y-3" aria-label={title}>
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Card({ children }: { children: ReactNode }) {
  return <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">{children}</div>;
}

export function Facts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
      {items.map(([term, detail]) => (
        <div key={term} className="contents">
          <dt className="text-gray-500">{term}</dt>
          <dd className="min-w-0 text-gray-900">{detail}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Loading, and the API's own problem text on failure (fixed text, never echoed input). */
export function Loading({
  query,
  children,
}: {
  query: { isPending: boolean; error: Error | null };
  children: () => ReactNode;
}) {
  if (query.error) {
    const problem = query.error instanceof ApiProblem ? query.error : null;
    return (
      <div
        role="alert"
        className="rounded border border-red-200 bg-red-50 p-4 text-sm text-red-800"
      >
        <p className="font-medium">{problem?.title ?? "Could not load"}</p>
        <p>{problem?.detail ?? "The Studio API could not be reached."}</p>
      </div>
    );
  }
  if (query.isPending) {
    return (
      <p className="text-sm text-gray-500" aria-live="polite">
        Loading…
      </p>
    );
  }
  return <>{children()}</>;
}
