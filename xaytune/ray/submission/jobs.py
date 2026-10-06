"""``RayJobsBackend``: submission through the Ray Jobs API to an existing cluster.

Xaytune does not create the cluster. ``address`` names the job-submission
(dashboard) endpoint of one that already runs -- a local ``ray start --head``,
a cluster on VMs, or a RayCluster KubeRay manages -- and nothing here knows
which. Submitting a ``RayJob`` custom resource instead is a different
:class:`~xaytune.ray.submission.protocol.RaySubmissionBackend` (PR-033c).

This is the only module that imports ``ray``, and it does so lazily:
``xaytune.ray`` imports without Ray installed, and only a backend actually
talking to a cluster needs it (``pip install xaytune[ray]``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from xaytune.ray.submission.protocol import RayJob, RayUnavailableError

__all__ = ["REDIRECTING_VARIABLES", "RayJobsBackend"]

REDIRECTING_VARIABLES = ("RAY_API_SERVER_ADDRESS", "RAY_ADDRESS")
"""Ray's SDK prefers these to the address it is given, in this order."""


class RayJobsBackend:
    """A ``RaySubmissionBackend`` over ``ray.job_submission.JobSubmissionClient``.

    Args:
        address: The cluster's job-submission (dashboard) address, such as
            ``http://127.0.0.1:8265``. Authoritative: Ray's SDK would let
            ``RAY_API_SERVER_ADDRESS`` or ``RAY_ADDRESS`` replace it, so a
            process where either names another address is refused --
            :class:`RayUnavailableError`, before any request -- rather than
            submitting to a cluster the experiment does not record.

    Direct authentication and TLS settings for the Jobs API are not
    supported yet: the address carries no credentials.
    """

    def __init__(self, address: str) -> None:
        self.address = address
        self._client: Any = None

    def _jobs(self) -> Any:
        if self._client is None:
            # Refused before anything is imported or contacted: the cluster
            # the SDK would talk to must be the one the experiment records.
            _require_authoritative(self.address, _effective_address(self.address))
            try:
                from ray.job_submission import JobSubmissionClient
            except ImportError as missing:  # pragma: no cover - depends on the install
                raise RayUnavailableError(
                    "submitting to Ray needs Ray: install xaytune[ray]"
                ) from missing
            try:
                client = JobSubmissionClient(self.address)
            except Exception as error:
                raise RayUnavailableError(
                    f"cannot reach Ray at {self.address}: {type(error).__name__}"
                ) from None
            # And checked again on what the client resolved, before it is used.
            _require_authoritative(self.address, getattr(client, "_address", None))
            self._client = client
        return self._client

    def submit(
        self,
        submission_id: str,
        entrypoint: str,
        *,
        metadata: Mapping[str, str],
        resources: Mapping[str, Any],
        runtime_env: Mapping[str, Any],
    ) -> None:
        try:
            self._jobs().submit_job(
                entrypoint=entrypoint,
                submission_id=submission_id,
                metadata=dict(metadata),
                runtime_env=_plain(runtime_env) or None,
                **dict(resources),
            )
        except RayUnavailableError:
            raise
        except Exception as error:
            raise RayUnavailableError(
                f"Ray did not accept job {submission_id}: {type(error).__name__}"
            ) from None

    def info(self, submission_id: str) -> RayJob | None:
        try:
            details = self._jobs().get_job_info(submission_id)
        except RayUnavailableError:
            raise
        except Exception as error:
            if _is_not_found(error):
                return None
            raise RayUnavailableError(
                f"cannot ask Ray about job {submission_id}: {type(error).__name__}"
            ) from None
        return RayJob(
            submission_id=submission_id,
            status=str(getattr(details.status, "value", details.status)),  # type: ignore[arg-type]
            metadata=dict(details.metadata or {}),
            message=details.message,
            exit_code=getattr(details, "driver_exit_code", None),
        )

    def stop(self, submission_id: str) -> None:
        try:
            self._jobs().stop_job(submission_id)
        except RayUnavailableError:
            raise
        except Exception as error:
            if _is_not_found(error):
                return
            raise RayUnavailableError(
                f"cannot ask Ray to stop job {submission_id}: {type(error).__name__}"
            ) from None


def _is_not_found(error: BaseException) -> bool:
    """Whether Ray answered that the job does not exist -- and only that.

    The client reports HTTP failures as ``RuntimeError("Request failed with
    status code 404: Job ... does not exist.")``. Matched narrowly: any other
    failure is an inability to answer, which must not read as absence.
    """
    text = str(error)
    return "status code 404" in text and "does not exist" in text


def _plain(value: Any) -> Any:
    """A frozen configuration value as the plain JSON Ray serializes."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _effective_address(address: str) -> str:
    """The address Ray's SDK would use, by its own precedence, contacting nothing."""
    for name in REDIRECTING_VARIABLES:
        value = os.environ.get(name)
        if value:
            return value
    return address


def _require_authoritative(address: str, effective: object) -> None:
    if isinstance(effective, str) and _normal(effective) == _normal(address):
        return
    named = [name for name in REDIRECTING_VARIABLES if os.environ.get(name)]
    # The variables are named, never their values: an address may carry credentials.
    because = f"{' / '.join(named)} is set to another address" if named else "it resolved elsewhere"
    raise RayUnavailableError(
        f"Ray's SDK would not submit to the configured address {address}: {because}. "
        f"The configured address is authoritative; unset the variable for this process"
    )


def _normal(address: str) -> str:
    return address.strip().rstrip("/")
