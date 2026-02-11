from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from pydantic import ValidationError

from orchestrator.llm_client import LLMClient
from schemas.code_survey_ir import CodeSurveyResult


class CodeSurveyAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client
        self.last_llm_trace: Optional[Dict[str, object]] = None

    def survey(self, context: Dict[str, object]) -> Optional[CodeSurveyResult]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("code_survey")
        data = self.llm_client.request_json(prompt, context)
        self.last_llm_trace = {"payload": context, "response": data}
        if not data:
            return None
        try:
            result = CodeSurveyResult(**data)
        except ValidationError:
            return None
        if result.status != "OK":
            return None
        return result


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
