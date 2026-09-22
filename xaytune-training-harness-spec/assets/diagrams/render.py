"""Generate the spec's architecture SVGs using only the Python standard library.

Run: python xaytune-training-harness-spec/assets/diagrams/render.py
SVGs contain accessible descriptions, embedded styling, and no external assets.
"""

from html import escape
from pathlib import Path

ROOT = Path(__file__).parent
INK = "#172b40"
MUTED = "#52677b"
TEAL = "#087f79"
PURPLE = "#694bb1"
BLUE = "#2567ad"
ORANGE = "#ad591f"


class Diagram:
    def __init__(self, name, height, title, subtitle, description):
        self.name = name
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
            f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>',
            "<style>text{font-family:Arial,Helvetica,sans-serif} "
            ".mono{font-family:Menlo,Consolas,monospace}</style>",
            f'<rect width="1200" height="{height}" fill="#f5f8fb"/>',
        ]
        self.text(48, 40, "XAYTUNE  /  ARCHITECTURE SPECIFICATION", 15, TEAL, bold=True)
        self.text(48, 86, title, 34, bold=True)
        self.text(48, 119, subtitle, 19, MUTED)

    def text(self, x, y, value, size=22, color=INK, bold=False, center=False, mono=False):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'font-weight="{700 if bold else 400}" '
            f'text-anchor="{"middle" if center else "start"}" '
            f'class="{"mono" if mono else ""}">{escape(value)}</text>'
        )

    def rect(self, x, y, width, height, fill="white", stroke="#d5e0ea", radius=16):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )

    def card(self, x, y, width, height, heading, lines=(), color=BLUE, fill="white"):
        self.rect(x, y, width, height, fill)
        self.rect(x + 20, y + 24, 5, height - 48, color, color, 2)
        self.text(x + 42, y + 42, heading, 24, color, bold=True)
        for index, line in enumerate(lines):
            self.text(x + 42, y + 74 + index * 29, line, 19, MUTED)

    def arrow(self, points, dashed=False):
        path = "M " + " L ".join(f"{x} {y}" for x, y in points)
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.parts.append(
            f'<path d="{path}" fill="none" stroke="{MUTED}" stroke-width="2" '
            f'marker-end="url(#arrow)"{dash}/>'
        )

    def bridge(self, top, bottom, label):
        self.arrow([(600, top), (600, bottom)])
        width = max(270, len(label) * 12 + 40)
        self.rect(600 - width / 2, (top + bottom) / 2 - 17, width, 34, "#f5f8fb", "#f5f8fb", 4)
        self.text(600, (top + bottom) / 2 + 6, label, 18, MUTED, center=True, mono=True)

    def save(self):
        self.text(48, self.height - 24, "Xaytune 2.0 · Target architecture", 14, MUTED)
        self.text(
            1152,
            self.height - 24,
            "Contracts and rollout status: see linked specification",
            14,
            MUTED,
            center=False,
        )
        # Right-align the footer without relying on a particular installed font.
        self.parts[-1] = self.parts[-1].replace('text-anchor="start"', 'text-anchor="end"')
        self.parts.append("</svg>")
        svg = "\n".join(self.parts) + "\n"
        # Keep accessible labels and markers distinct when several SVGs are inlined.
        for identifier in ("title", "desc", "arrow"):
            svg = svg.replace(f'id="{identifier}"', f'id="{self.name}-{identifier}"')
            svg = svg.replace(f"url(#{identifier})", f"url(#{self.name}-{identifier})")
        svg = svg.replace(
            'aria-labelledby="title desc"',
            f'aria-labelledby="{self.name}-title {self.name}-desc"',
        )
        (ROOT / f"{self.name}.svg").write_text(svg, encoding="utf-8")


def overview():
    d = Diagram(
        "architecture-overview",
        1390,
        "One control plane. Multiple execution backends.",
        "Experiment intent stays separate from trainer compilation and runtime execution.",
        "Clients enter the experiment control plane. CandidateSpec passes to trainer compilers, "
        "TrainingExecutionSpec to capability and resilience resolution, and ResolvedExecutionPlan "
        "to runtime backends using submit_or_get. Runtime backends delegate to infrastructure. "
        "Training Hub is the preferred path to Kubeflow or KubeRay and Kueue.",
    )
    d.card(
        48, 153, 1104, 94, "Interfaces", ["Python SDK  ·  CLI  ·  Studio  ·  Coding agents"], BLUE
    )
    d.arrow([(600, 247), (600, 281)])
    d.rect(48, 285, 1104, 266, "#eaf6f3", "#b7dcd5")
    d.text(76, 324, "Experiment control plane", 27, TEAL, bold=True)
    d.text(76, 354, "ExperimentController coordinates durable, governed experiments", 20, MUTED)
    for x, head, rows in [
        (76, "Scientific lineage", ["ExperimentGraph", "Provenance / Memory"]),
        (438, "Adaptive decisions", ["Planner / SearchProvider", "Evaluation / DecisionEngine"]),
        (
            800,
            "Governance & recovery",
            ["PolicyEngine / BudgetLedger", "Incident / RecoveryCoordinator"],
        ),
    ]:
        d.text(x, 400, head, 21, TEAL, bold=True)
        for index, row in enumerate(rows):
            d.text(x, 430 + index * 29, row, 18)
    d.text(
        76,
        520,
        "Planning: rules · LLM proposals · search adapters (Ray Tune / Katib / Optuna)",
        19,
        MUTED,
    )
    d.bridge(551, 616, "CandidateSpec")
    d.card(
        48,
        620,
        1104,
        102,
        "Trainer compilers",
        ["NativeCompiler  ·  TRLCompiler  ·  TorchtuneCompiler  ·  VerlCompiler"],
        PURPLE,
    )
    d.bridge(722, 792, "TrainingExecutionSpec")
    d.card(
        48,
        796,
        1104,
        102,
        "Capability & resilience resolution",
        ["CapabilityResolver  ·  ResilienceProvider  ·  Checkpoint contract"],
        PURPLE,
    )
    d.bridge(898, 968, "ResolvedExecutionPlan")
    d.card(
        48,
        972,
        1104,
        120,
        "Runtime backends",
        [
            "LocalRuntime  ·  RayTrainRuntime  ·  TrainingHubRuntime",
            "RuntimeBackend.submit_or_get(operation_id, plan)",
        ],
        BLUE,
    )
    d.arrow([(600, 1092), (600, 1130)])
    d.rect(48, 1134, 1104, 188, "#fff6ed", "#ecd8c5")
    d.text(76, 1174, "Platform / infrastructure", 25, ORANGE, bold=True)
    d.text(76, 1210, "Local: one subprocess  ·  Ray: Ray Train / Jobs", 21)
    d.text(76, 1247, "Training Hub path: Kubeflow Trainer / KubeRay → Kueue", 21)
    d.text(
        76,
        1287,
        "Integrations land incrementally; this diagram describes the target architecture.",
        18,
        MUTED,
    )
    d.save()


def execution():
    d = Diagram(
        "execution-path",
        1330,
        "From candidate to runtime workload",
        "Serializable contracts cross boundaries; runtime operations retain durable identity.",
        "ExperimentNode owns CandidateSpecSnapshot. TrainerCompiler.compile produces "
        "TrainingExecutionSpec. CapabilityResolver.resolve and ResilienceProvider.augment "
        "produce ResolvedExecutionPlan. Attempt and INTENDED operation are committed before "
        "RuntimeBackend.submit_or_get returns RuntimeRef. Status, events, logs and operation "
        "lookup support observation and reconciliation.",
    )
    d.card(
        100,
        155,
        1000,
        106,
        "01  Declare the candidate",
        ["ExperimentNode owns an immutable CandidateSpecSnapshot"],
        TEAL,
    )
    d.bridge(261, 321, "CandidateSpecSnapshot")
    d.card(
        100, 325, 1000, 104, "02  Compile training intent", ["TrainerCompiler.compile()"], PURPLE
    )
    d.bridge(429, 489, "TrainingExecutionSpec")
    d.card(
        100,
        493,
        1000,
        132,
        "03  Resolve execution requirements",
        ["CapabilityResolver.resolve()", "ResilienceProvider.augment()"],
        PURPLE,
    )
    d.bridge(625, 685, "ResolvedExecutionPlan")
    d.card(
        100,
        689,
        1000,
        134,
        "04  Commit operation intent",
        [
            "RunAttempt + INTENDED RuntimeOperation + request_digest",
            "One transaction with events and outbox; commit before the runtime call",
        ],
        ORANGE,
        "#fff8f0",
    )
    d.arrow([(600, 823), (600, 865)])
    d.card(
        100,
        869,
        1000,
        110,
        "05  Submit or recover the existing operation",
        ["RuntimeBackend.submit_or_get(operation_id, plan)"],
        BLUE,
    )
    d.bridge(979, 1039, "RuntimeRef")
    d.card(
        100,
        1043,
        1000,
        118,
        "06  Observe and reconcile",
        [
            "get_status()  ·  watch()  ·  get_logs()  ·  lookup_operation()",
            "cancel() records an external effect through the operation journal",
        ],
        TEAL,
    )
    d.text(
        600,
        1210,
        "Same operation ID + same request = the original workload",
        23,
        TEAL,
        bold=True,
        center=True,
    )
    d.text(
        600,
        1246,
        "The outbox publishes events; it does not dispatch runtime operations.",
        20,
        MUTED,
        center=True,
    )
    d.save()


def evaluation():
    d = Diagram(
        "evaluation-path",
        1140,
        "Evaluation has its own durable lifecycle",
        "Changing the evaluator does not change candidate identity or require retraining.",
        "ModelArtifact and EvaluationSpec feed evaluator preparation and execution. "
        "EvaluationRun owns seed and replicate; EvaluationAttempt records execution and retry. "
        "Versioned MetricResult values become EvaluationResult linked to EvaluationRun and "
        "the artifact, then DecisionEngine evaluates objective, constraints and uncertainty.",
    )
    d.card(
        48,
        158,
        530,
        120,
        "ModelArtifact",
        ["Immutable artifact reference", "Subject being measured"],
        TEAL,
    )
    d.card(
        622,
        158,
        530,
        120,
        "EvaluationSpec",
        ["Evaluators · dataset · slices", "Independent EvaluationFingerprint"],
        PURPLE,
    )
    d.arrow([(313, 278), (313, 309), (600, 309), (600, 348)])
    d.arrow([(887, 278), (887, 309), (600, 309), (600, 348)])
    d.card(
        100,
        352,
        1000,
        111,
        "Prepare evaluation",
        ["Evaluator.prepare(artifact, spec) → EvaluationExecutionSpec"],
        PURPLE,
    )
    d.arrow([(600, 463), (600, 505)])
    d.rect(100, 509, 1000, 182, "#eef5fc", "#c7daec")
    d.text(128, 550, "Execute through the evaluation coordinator", 25, BLUE, bold=True)
    d.text(128, 590, "EvaluationRun", 22, BLUE, bold=True)
    d.text(128, 620, "Seed / replicate ownership", 20)
    d.text(632, 590, "EvaluationAttempt", 22, BLUE, bold=True)
    d.text(632, 620, "Queue · execute · retry · reconcile", 20)
    d.text(
        128,
        662,
        "Runtime-operation journal persists submission intent before execution.",
        19,
        MUTED,
    )
    d.bridge(691, 751, "MetricResult[]")
    d.card(
        100,
        755,
        1000,
        106,
        "EvaluationResult",
        ["evaluation_run_id · artifact_ref · versioned metrics · constraints"],
        TEAL,
    )
    d.arrow([(600, 861), (600, 903)])
    d.card(
        100,
        907,
        1000,
        112,
        "DecisionEngine",
        ["Objective + constraints + uncertainty + budget + policy"],
        ORANGE,
    )
    d.text(
        600,
        1070,
        "Continue  ·  Evaluate more  ·  Branch  ·  Promote  ·  Stop",
        22,
        ORANGE,
        bold=True,
        center=True,
    )
    d.save()


def dependencies():
    d = Diagram(
        "dependency-boundaries",
        930,
        "Dependencies point toward contracts",
        "Arrows mean imports / dependencies, not workload execution.",
        "API depends on experiment orchestration. Experiment orchestration depends on domain "
        "and compiler, runtime, evaluation and persistence protocols. Plugins depend on protocols. "
        "Core does not import ML runtimes. Direct domain-to-ML, policy-to-TRL, "
        "recipe-to-Kubernetes, "
        "experiment-to-MLflow and core-to-Training-Hub implementation dependencies are forbidden.",
    )
    d.card(48, 164, 340, 100, "API", ["SDK · CLI · Studio"], BLUE)
    d.card(48, 334, 340, 115, "Experiment", ["Control-plane orchestration"], TEAL)
    d.arrow([(218, 264), (218, 330)])
    d.card(720, 270, 432, 105, "Domain", ["Immutable values / aggregates"], TEAL)
    d.card(
        720,
        424,
        432,
        177,
        "Protocols",
        ["Compiler · runtime", "Evaluation · persistence", "Typed integration contracts"],
        PURPLE,
    )
    d.arrow([(388, 377), (545, 377), (545, 322), (716, 322)])
    d.arrow([(388, 408), (545, 408), (545, 510), (716, 510)])
    d.card(48, 516, 340, 100, "Plugins", ["Implement protocol contracts"], PURPLE)
    d.arrow([(388, 566), (590, 566), (590, 558), (716, 558)])
    d.rect(48, 677, 1104, 183, "#fff4f1", "#eccfc7")
    d.text(76, 718, "Forbidden dependency directions", 24, "#a14536", bold=True)
    rows = [
        (76, 757, "domain → torch / TRL / Ray"),
        (622, 757, "policy → TRL"),
        (76, 793, "recipe → Kubernetes"),
        (622, 793, "experiment → MLflow"),
        (76, 829, "core → Training Hub implementation"),
    ]
    for x, y, label in rows:
        d.text(x, y, label, 21)
    d.save()


def roadmap():
    d = Diagram(
        "implementation-order",
        1160,
        "Build in the order contracts become fixed",
        "The numbered implementation plan remains authoritative for individual PRs and gates.",
        "Bands A through J: domain hardening, transactional persistence, compile/runtime, "
        "durable evaluation, action/policy/budget, recovery/interventions, planner/branching, "
        "daemon and restart MVP, LLM planner, and platform integrations. ADR-005 is "
        "accepted, so Band B is open. The operation journal precedes LocalRuntime; "
        "policy precedes recovery.",
    )
    stages = [
        ("A", "Domain contracts", "IDs · state machines · immutable scientific identity", TEAL),
        (
            "B",
            "Transactional persistence",
            "SQLite · events · outbox · operation journal · graph",
            ORANGE,
        ),
        (
            "C",
            "Compile & execute",
            "CandidateSpec · LocalRuntime · Native / TRL · reconciliation",
            PURPLE,
        ),
        (
            "D",
            "Durable evaluation",
            "EvaluationRun / EvaluationAttempt · metrics · decisions",
            BLUE,
        ),
        ("E", "Action / policy / budget", "Typed actions · approvals · reservations", TEAL),
        ("F", "Recovery & interventions", "Checkpoints · incidents · governed adaptation", ORANGE),
        ("G", "Planner & branching", "Rule-based planning · candidate comparisons", PURPLE),
        ("H", "Daemon & restart MVP", "Leases · startup reconciliation · attach / watch", BLUE),
        ("I", "LLM planner", "Structured proposals · policy enforcement · audit", TEAL),
        ("J", "Platform integrations", "Ray / TorchFT / Training Hub", PURPLE),
    ]
    for index, (letter, heading, detail, color) in enumerate(stages):
        y = 157 + index * 87
        if index:
            d.arrow([(86, y - 22), (86, y + 4)])
        d.rect(58, y + 6, 56, 56, color, color, 15)
        d.text(86, y + 43, letter, 27, "white", True, True)
        d.text(145, y + 30, heading, 24, color, bold=True)
        d.text(145, y + 60, detail, 20, MUTED)
    d.rect(48, 1051, 1104, 53, "#fff2df", "#ead6b5", 12)
    d.text(
        600,
        1084,
        "ADR-005 accepted 2026-09-21 \u2014 Band B is open and PR-004 may start.",
        21,
        ORANGE,
        bold=True,
        center=True,
    )
    d.save()


if __name__ == "__main__":
    for render in (overview, execution, evaluation, dependencies, roadmap):
        render()
