from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepReport:
    name: str
    input_rows: int
    output_rows: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def rows_removed(self) -> int:
        return self.input_rows - self.output_rows


@dataclass
class PrepReport:
    input_rows: int
    output_rows: int
    steps: list[StepReport] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Input: {self.input_rows} rows → Output: {self.output_rows} rows"]
        for step in self.steps:
            lines.append(
                f"  {step.name}: {step.input_rows} → {step.output_rows} "
                f"(removed {step.rows_removed})"
            )
        return "\n".join(lines)


@dataclass
class PrepResult:
    dataset: list[dict[str, Any]]
    report: PrepReport

    def save(self, path: str, format: str = "jsonl") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            p.write_text(json.dumps(self.dataset, ensure_ascii=False, indent=2))
        else:
            with open(p, "w") as f:
                for item in self.dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
