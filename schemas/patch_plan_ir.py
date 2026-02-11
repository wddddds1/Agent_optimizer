from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from schemas.strict_base import LLMStatus, StrictBaseModel


class PatchAction(StrictBaseModel):
    """A concrete patch action proposed by PatchPlanner.

    Unlike the old Idea -> Survey -> Synth chain, ``code_context`` carries
    the actual source code the LLM identified as the optimisation target,
    so that downstream CodePatchAgent can work from real code rather than
    a vague anchor string.
    """

    action_id: str
    patch_family: str
    target_file: str
    target_anchor: str = ""
    mechanism: str = ""
    expected_effect: str = ""
    risk_level: str = "medium"
    wrapper_id: Optional[str] = None
    rationale: str = ""
    evidence: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    code_context: str = ""


class PatchPlan(StrictBaseModel):
    actions: List[PatchAction] = Field(default_factory=list)
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
