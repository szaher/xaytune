from __future__ import annotations


def launch(
    *,
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
) -> None:
    import gradio as gr

    from trainlib.studio.app import create_app

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
    )
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share, theme=theme)
