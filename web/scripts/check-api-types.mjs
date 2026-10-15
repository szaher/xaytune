// Fails when src/api/schema.d.ts is not what the committed OpenAPI document
// generates (ADR-023 §S5): the client's types have one source, the API's contract.
import { readFileSync } from "node:fs";
import openapiTS, { astToString } from "openapi-typescript";

const document = new URL("../../xaytune/api/v1/openapi.json", import.meta.url);
const committed = new URL("../src/api/schema.d.ts", import.meta.url);

const generated = astToString(await openapiTS(document)).trim();
// The CLI prefixes a "do not edit" banner; everything after it must match.
const current = readFileSync(committed, "utf8")
  .replace(/^\/\*\*[\s\S]*?\*\/\s*/, "")
  .trim();

if (current !== generated) {
  console.error("src/api/schema.d.ts is stale: run `npm run api:generate` and commit the result.");
  process.exit(1);
}
console.log("src/api/schema.d.ts matches xaytune/api/v1/openapi.json");
