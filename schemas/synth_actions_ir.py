from __future__ import annotations

from typing import List

from pydantic import Field

from schemas.action_ir import ActionIR
from schemas.strict_base import LLMStatus, StrictBaseModel


class SynthesizedActions(StrictBaseModel):
    actions: List[ActionIR] = Field(default_factory=list)
    status: LLMStatus = "OK"
    missing_fields: List[str] = Field(default_factory=list)
