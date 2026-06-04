from __future__ import annotations

from pathlib import Path


def launch(
    *,
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
) -> None:
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
    app = create_app(job_manager=mgr, theme=theme)
    app.launch(server_name=host, server_port=port, share=share)
