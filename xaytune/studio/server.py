from __future__ import annotations

import ipaddress
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"

SHARE_DISABLED = (
    "Public sharing is disabled: Studio has no authentication, and a public "
    "link would let anyone on the internet launch jobs on this machine."
)


class StudioExposureError(ValueError):
    """Studio was asked to listen somewhere it cannot be served safely."""


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` binds only to this machine.

    ``localhost`` and loopback addresses (``127.0.0.0/8``, ``::1``) are local.
    Everything else -- ``0.0.0.0``, ``::``, a LAN address or any other host
    name -- is treated as reachable from the network.
    """
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip().strip("[]")).is_loopback
    except ValueError:
        return False


def launch(
    *,
    host: str = DEFAULT_HOST,
    port: int = 7860,
    share: bool = False,
) -> None:
    """Serve Studio on ``host``.

    Studio has no authentication, so anything it serves is available to
    whoever can reach ``host``. Public sharing is refused. On a non-loopback
    host, jobs that would run a model repository's own code
    (``trust_remote_code``) are refused by the server.
    """
    if share:
        raise StudioExposureError(SHARE_DISABLED)
    local = is_loopback_host(host)
    if not local:
        _LOG.warning(
            "Studio is listening on %s, which is reachable from the network. "
            "Studio has no authentication: anyone who can reach this port can "
            "launch training jobs. Trust Remote Code is disabled.",
            host,
        )

    import gradio as gr

    from xaytune.studio.app import create_app
    from xaytune.studio.jobs import JobManager

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    )
    persist_dir = Path("~/.xaytune/studio/jobs").expanduser()
    mgr = JobManager(persist_dir=persist_dir)
    app = create_app(job_manager=mgr, theme=theme, allow_remote_code=local)
    app.launch(server_name=host, server_port=port, share=False)
