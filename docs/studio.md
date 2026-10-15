# Training Studio

xaytune Studio is a web UI for configuring, launching, and monitoring training jobs. Launch it with:

```bash
xaytune studio                # http://127.0.0.1:7860
xaytune studio --port 8080
```

!!! tip "Xaytune Studio on the control-plane record"
    `xaytune studio-web --state state.db` opens a read-only Studio at
    `http://127.0.0.1:8765/` built on the control-plane record: experiments,
    the candidate graph, runs with their recovery, evaluations, decisions and
    comparisons, for anything the daemon or a controller host has recorded.
    See [Studio web](#studio-web) below. The Gradio Studio described on the
    rest of this page stays available as a legacy entry point.

!!! note "Requires Gradio"
    Install with `pip install xaytune[studio]` or `pip install gradio plotly`.

## Security

Studio has **no authentication**. Anyone who can reach its port can use it: launch training jobs, read dataset files the server can read, and log the server in to the Hugging Face Hub. Every button is also an HTTP API endpoint, so the page's layout is not a boundary.

- Studio listens on `127.0.0.1` by default, so only this machine can reach it. To use it from another machine, prefer an SSH tunnel (`ssh -L 7860:127.0.0.1:7860 host`) over binding another address.
- `--host 0.0.0.0` (or any non-loopback address) is allowed, logs a warning, and disables **Trust Remote Code**: a job that would run a model repository's own Python is refused by the server, not only greyed out in the form.
- `--share` is refused. A public Gradio link would expose Studio to the internet.
- Studio does not run Python you type. The former Code tab is removed; **Generate Code** writes a script for you to run yourself.

## Train Tab

Configure and launch training jobs with a form-based UI.

### Simple vs Advanced Mode

- **Simple** — shows only essential options: recipe, method, model, data path, and alignment parameters.
- **Advanced** — expands all training options: batch size, learning rate, scheduler, evaluation, checkpointing, and more.

### Model Search

The **Model Search** accordion queries HuggingFace Hub for text-generation models. Type a query (e.g., "llama", "mistral"), click **Search Models**, and click a result row to populate the model name field.

### Data Preview

The **Data Preview** accordion loads and displays the first 5 samples from your dataset file. Click **Preview Data** to inspect your JSONL before training. Long values are truncated at 200 characters.

### Validation

The form validates inputs before submission:

- Model name and data path are required
- Recipe/method combinations must be valid (e.g., `align` requires `dpo`/`grpo`/etc.)
- QLoRA requires 4-bit quantization
- Warmup steps and warmup ratio are mutually exclusive

## Monitor Tab

Real-time monitoring of running training jobs.

### Job Selection

Select a job from the dropdown and click **Refresh Jobs** to update the list. The timer polls every 2 seconds while a job is selected.

### Live Metrics

- **Status badge** — colored badge showing PENDING, RUNNING, COMPLETED, FAILED, or CANCELLED
- **Elapsed time** — running clock for active jobs
- **Step counter** — current global step with loss and other metrics
- **Throughput** — samples/sec computed from step timestamps
- **Loss curve** — Plotly line chart updated in real time

### Cancel Button

A **Cancel Job** button appears while a job is running. Cancellation is cooperative — it sets a flag that the training loop checks at each step end.

### GPU Metrics

When running on CUDA, the monitor displays:

- GPU memory allocated (MB)
- GPU memory reserved (MB)
- Peak GPU memory (MB)

On CPU, this section is hidden.

### Training Logs

A scrollable log panel shows the last 200 lines from the training log buffer. Logs are captured from Python's root logger during the job.

## History Tab

Browse all past and current training jobs in a table showing:

| Column | Description |
|--------|-------------|
| Job ID | First 8 characters of the UUID |
| Recipe | `finetune`, `pretrain`, or `align` |
| Status | Current job status |
| Created | Job submission timestamp |
| Completed | Completion timestamp (if finished) |
| Final Loss | Last recorded loss value |
| Tags | Comma-separated user tags |

### Tagging Jobs

Select a job from the dropdown, type a tag name, and click **Add Tag** or **Remove Tag**. Tags are useful for organizing experiments (e.g., `baseline`, `v2`, `ablation-lr`).

## Compare Tab

Compare training runs side by side.

1. Click **Refresh Jobs** to load the job list
2. Select two or more jobs from the multi-select dropdown
3. Click **Compare**

The comparison view shows:

- **Loss curves** — overlaid Plotly chart with one trace per job, color-coded
- **Metrics table** — side-by-side summary with job ID, recipe, total steps, final loss, and tags

## Python API

```python
from xaytune.studio.app import create_app
from xaytune.studio.jobs import JobManager

# Create with a shared job manager
mgr = JobManager()
app = create_app(job_manager=mgr)
app.launch(server_name="127.0.0.1", server_port=7860)
```

`create_app()` refuses jobs with Trust Remote Code unless you pass `allow_remote_code=True`. Pass it only when the app listens on a loopback address.

### Programmatic Job Submission

```python
from xaytune.config.schema import TrainConfig, ModelConfig, DataConfig, TrainerConfig
from xaytune.studio.jobs import JobManager

mgr = JobManager()

config = TrainConfig(
    recipe="finetune",
    method="full",
    model=ModelConfig(name="meta-llama/Llama-3-8B"),
    data=DataConfig(path="data/train.jsonl", format="alpaca"),
    trainer=TrainerConfig(num_epochs=1, max_steps=100),
)

job_id = mgr.submit(config)
status = mgr.get_status(job_id)
print(status.status)  # "running" or "completed"
```

## Studio web

```bash
pip install "xaytune[studio-web]"
xaytune studio-web --state state.db        # http://127.0.0.1:8765/
```

A read-only view of the control-plane record (ADR-023). It reads the same
state database as `xaytune status`, opened read-only, so it can run with or
without a daemon, and restarting it loses nothing.

| Section | Shows |
|---|---|
| Overview | Active experiments, actions waiting for approval (with the CLI command that answers them), recent failures |
| Experiments | Every experiment, filterable by status, newest first |
| Experiment | Tabs: overview, the **candidate graph** with lineage and decisions, runs, decisions, actions with their policy decisions, the event journal with incidents and recovery, budget, and the recorded configuration |
| Candidate | Its spec, lineage, runs, evaluations and decisions |
| Run | Each attempt with its incidents, overrides and the checkpoint it restored from |
| Compare | Two candidates: their lineage, spec differences and metrics side by side. A difference is shown only when both were measured under the same evaluation and unit |

Every view's address includes its selection, so a link reproduces what you
saw. Changes (approvals, cancellation) are still made with the CLI.
Monitoring charts, harness evaluations and trajectories arrive with the
backend work ADR-023 §S12 names.

A release wheel carries the built client. From a source checkout, build it
once with `npm ci && npm run build` in `web/`.
