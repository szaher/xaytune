import { useState } from "react";
import { Absent } from "./values";

/**
 * A recorded document (a spec or payload, ADR-023 §S5) as a key/value tree.
 * Nested objects and lists collapse below the first level, and long lists
 * show their first entries, so a large checkpoint manifest stays readable.
 * Every leaf is text: nothing in a recorded document is rendered as markup.
 */
export function DocumentTree({
  value,
  label,
  open = 1,
}: {
  value: unknown;
  label?: string;
  /** How many levels start expanded. */
  open?: number;
}) {
  return (
    <div role="group" className="font-mono text-xs" aria-label={label}>
      <Node value={value} depth={0} open={open} />
    </div>
  );
}

const LIST_PREVIEW = 20;

function Node({ value, depth, open }: { value: unknown; depth: number; open: number }) {
  if (value === null || value === undefined) return <Absent reason="not recorded" />;
  if (typeof value !== "object") {
    return <span className="whitespace-pre-wrap break-all">{String(value)}</span>;
  }
  const entries: [string, unknown][] = Array.isArray(value)
    ? value.map((item, index) => [String(index), item])
    : Object.entries(value as Record<string, unknown>);
  if (entries.length === 0) {
    return <span className="text-gray-400">{Array.isArray(value) ? "[]" : "{}"}</span>;
  }
  const body = <Entries entries={entries} depth={depth} open={open} list={Array.isArray(value)} />;
  if (depth < open) return body;
  return (
    <details>
      <summary className="cursor-pointer text-gray-500">
        {Array.isArray(value) ? `[${entries.length} items]` : `{${entries.length} keys}`}
      </summary>
      {body}
    </details>
  );
}

function Entries({
  entries,
  depth,
  open,
  list,
}: {
  entries: [string, unknown][];
  depth: number;
  open: number;
  list: boolean;
}) {
  const [all, setAll] = useState(false);
  const shown = all ? entries : entries.slice(0, LIST_PREVIEW);
  return (
    <dl className="ml-1 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 border-l border-gray-200 pl-3">
      {shown.map(([key, item]) => (
        <div key={key} className="contents">
          <dt className="text-gray-500">{list ? `${key}.` : key}</dt>
          <dd className="min-w-0">
            <Node value={item} depth={depth + 1} open={open} />
          </dd>
        </div>
      ))}
      {entries.length > shown.length ? (
        <div className="col-span-2">
          <button type="button" className="text-blue-700 underline" onClick={() => setAll(true)}>
            show all {entries.length}
          </button>
        </div>
      ) : null}
    </dl>
  );
}
