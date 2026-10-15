import { Link } from "@tanstack/react-router";
import { Page } from "../components/layout";

export function NotFound() {
  return (
    <Page title="Not found">
      <p className="text-sm text-gray-600">
        Studio has no view at this address.{" "}
        <Link to="/" className="text-blue-700 underline">
          Go to the overview
        </Link>
        .
      </p>
    </Page>
  );
}
