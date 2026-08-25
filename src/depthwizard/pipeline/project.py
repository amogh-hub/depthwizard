from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from depthwizard.pipeline.stages import ProcessingStage


@dataclass
class ProjectManifest:
    project_dir: Path
    source_path: Path
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.project_dir / "project-manifest.json"

    def record_stage(
        self,
        stage: ProcessingStage,
        *,
        status: str,
        artifacts: dict[str, str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.stages[stage.value] = {
            "status": status,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "artifacts": artifacts or {},
            "details": details or {},
        }
        self.save()

    def save(self) -> None:
        payload = {
            "schema_version": 1,
            "source_path": str(self.source_path),
            "stages": self.stages,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, project_dir: str | Path) -> ProjectManifest:
        directory = Path(project_dir)
        payload = json.loads((directory / "project-manifest.json").read_text(encoding="utf-8"))
        return cls(
            project_dir=directory,
            source_path=Path(payload["source_path"]),
            stages=payload.get("stages", {}),
        )
