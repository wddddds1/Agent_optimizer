from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

LLMStatus = Literal["OK", "NEED_MORE_EVIDENCE", "NEED_MORE_CONTEXT"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
