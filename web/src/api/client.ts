/**
 * The typed client for `/api/v1` (ADR-023 §S5). Every type comes from
 * `schema.d.ts`, generated from the committed OpenAPI document; nothing here
 * re-declares the wire.
 */
import createClient from "openapi-fetch";
import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

export const api = createClient<paths>({ baseUrl: "" });

/** An `application/problem+json` answer, as the API sent it (fixed text, never echoed input). */
export class ApiProblem extends Error {
  readonly status: number;
  readonly title: string;
  readonly detail: string;

  constructor(status: number, body: unknown) {
    const problem = (body ?? {}) as Partial<Schemas["Problem"]>;
    const title = problem.title ?? "Request failed";
    const detail = problem.detail ?? `the API answered ${status}`;
    super(`${title}: ${detail}`);
    this.status = status;
    this.title = title;
    this.detail = detail;
  }
}

/** The body of a successful answer, or an {@link ApiProblem}. */
export async function read<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await call;
  if (data === undefined) {
    throw new ApiProblem(response.status, error);
  }
  return data;
}
