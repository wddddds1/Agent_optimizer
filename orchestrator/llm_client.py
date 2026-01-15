from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import openai

from schemas.action_ir import ActionIR
from schemas.profile_report import ProfileReport


@dataclass
class LLMConfig:
    enabled: bool
    api_key_env: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = None
        if not config.enabled:
            return
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {config.api_key_env}")
        client_cls = getattr(openai, "Client", None) or getattr(openai, "OpenAI", None)
        if client_cls is None:
            raise RuntimeError("OpenAI client class not available in openai package")
        self.client = client_cls(api_key=api_key, base_url=config.base_url)

    def rank_actions(
        self,
        actions: List[ActionIR],
        profile: ProfileReport,
        context: Dict[str, object],
    ) -> List[str]:
        if not self.client:
            return []
        payload = {
            "profile": {
                "timing_breakdown": profile.timing_breakdown,
                "system_metrics": profile.system_metrics,
            },
            "context": context,
            "actions": [
                {
                    "action_id": a.action_id,
                    "family": a.family,
                    "description": a.description,
                    "expected_effect": a.expected_effect,
                    "risk_level": a.risk_level,
                }
                for a in actions
            ],
        }
        prompt = (
            "You are an HPC performance optimizer. Return only a JSON array of action_id "
            "sorted from best to worst. Only use action_id values provided.\n\n"
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": "Return only JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [str(item) for item in data]
        except json.JSONDecodeError:
            return []
        return []

    def explain_decision(self, context: Dict[str, object]) -> str:
        if not self.client:
            return ""
        prompt = (
            "Explain the action ranking rationale in 3-5 bullet points. "
            "Do not mention missing info."
        )
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": "Return plain text only."},
                {"role": "user", "content": prompt + "\n\nContext:\n" + json.dumps(context)},
            ],
        )
        return response.choices[0].message.content.strip()
