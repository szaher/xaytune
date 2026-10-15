import { Link, Outlet } from "@tanstack/react-router";
import { useMeta } from "./api/queries";

const SECTIONS = [
  { to: "/", label: "Overview" },
  { to: "/experiments", label: "Experiments" },
  { to: "/compare", label: "Compare" },
  { to: "/monitoring", label: "Monitoring" },
  { to: "/trajectories", label: "Trajectories" },
  { to: "/optimization", label: "Optimization" },
] as const;

/** The left rail and the store banner: what Studio reads, and whether it can. */
export function Shell() {
  return (
    <div className="flex min-h-full">
      <nav
        aria-label="Sections"
        className="w-52 shrink-0 border-r border-gray-200 bg-white px-3 py-4"
      >
        <Link to="/" className="mb-6 block px-2 text-lg font-semibold tracking-tight">
          <span aria-hidden="true">◆</span> Xaytune Studio
        </Link>
        <ul className="space-y-1 text-sm">
          {SECTIONS.map((section) => (
            <li key={section.to}>
              <Link
                to={section.to}
                activeOptions={{ exact: section.to === "/" }}
                className="block rounded px-2 py-1.5 text-gray-700 hover:bg-gray-100"
                activeProps={{ className: "bg-gray-100 font-medium text-gray-900" }}
              >
                {section.label}
              </Link>
            </li>
          ))}
        </ul>
        <p className="mt-8 px-2 text-xs text-gray-400">Read-only. Changes go through the CLI.</p>
      </nav>
      <div className="flex min-w-0 flex-1 flex-col">
        <StoreBanner />
        <main className="flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

function StoreBanner() {
  const meta = useMeta();
  if (meta.error) {
    return (
      <div
        role="alert"
        className="border-b border-red-200 bg-red-50 px-6 py-2 text-sm text-red-800"
      >
        The Studio API cannot be reached.
      </div>
    );
  }
  if (!meta.data) return <div className="h-9 border-b border-gray-200 bg-white" />;
  const { store, controller_lease: lease, latest_event_sequence: sequence } = meta.data;
  return (
    <div
      role="status"
      className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-gray-200 bg-white px-6 py-2 text-xs text-gray-600"
      aria-label="State database"
    >
      {store.readable ? (
        <span>
          state database · migration {store.migration} · event {sequence}
        </span>
      ) : (
        <span role="alert" className="font-medium text-red-700">
          State database unreadable: {store.reason}
        </span>
      )}
      <span className="flex items-center gap-1">
        daemon{" "}
        {lease ? (
          <span className={lease.live ? "text-green-700" : "text-amber-700"}>
            {lease.live ? "● holds the lease" : "◐ lease expired"}
          </span>
        ) : (
          <span className="text-gray-400">no lease recorded</span>
        )}
      </span>
    </div>
  );
}
