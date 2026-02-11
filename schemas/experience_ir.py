from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field

from schemas.strict_base import StrictBaseModel


class ExperienceRecord(StrictBaseModel):
    action_id: str
    family: str
    outcome: str
    improvement_pct: float = 0.0
    speedup_vs_baseline: float = 0.0
    variance_cv: Optional[float] = None
    case_id: str = ""
    app: str = ""
    backend: Optional[str] = None
    target_file: Optional[str] = None
    patch_family: Optional[str] = None
    run_id: str = ""
    timestamp: str = ""
    strength: str = "weak"
    weight: float = 0.0
    evidence: Dict[str, str] = Field(default_factory=dict)
    # Deep analysis context — persisted so future sessions benefit from insights
    origin: Optional[str] = None
    category: Optional[str] = None
    diagnosis: Optional[str] = None
    mechanism: Optional[str] = None
    compiler_gap: Optional[str] = None
    target_functions: Optional[List[str]] = None
