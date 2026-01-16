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

    def summarize_report_zh(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not self.client:
            return None
        prompt = (
            "你是性能优化审计员。请根据给定实验数据，输出一个 JSON 对象，包含以下键：\n"
            "- experiment_reasons: {run_id: 中文简要原因}\n"
            "- experiment_details: {run_id: 中文详细分析段落（不少于4句）}\n"
            "- overall_analysis: [\"中文分析句子\", ...] (3-6 条)\n"
            "- selection_reason: \"中文简要说明为什么最终选择该优化\"\n"
            "要求：只输出 JSON，不要 markdown，不要额外文字。原因要简短、具体、可解释。"
        )
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": prompt + "\n\nPayload:\n" + json.dumps(payload)},
            ],
        )
        content = response.choices[0].message.content.strip()
        data = _safe_json_loads(content)
        if not isinstance(data, dict):
            return None
        return data

    def summarize_iteration_zh(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not self.client:
            return None
        prompt = (
            "你是性能优化审计员。请根据给定本轮迭代数据，输出一个 JSON 对象，包含以下键：\n"
            "- experiment_reasons: {run_id: 中文简要原因}\n"
            "- experiment_details: {run_id: 中文详细分析段落（不少于4句）}\n"
            "- summary_lines: [\"中文分析句子\", ...] (2-5 条)\n"
            "- selection_reason: \"中文简要说明本轮为什么选择该最优配置\"\n"
            "要求：只输出 JSON，不要 markdown，不要额外文字。原因要简短、具体、可解释。"
        )
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": "Return only JSON."},
                {"role": "user", "content": prompt + "\n\nPayload:\n" + json.dumps(payload)},
            ],
        )
        content = response.choices[0].message.content.strip()
        data = _safe_json_loads(content)
        if not isinstance(data, dict):
            return None
        return data


def _safe_json_loads(content: str) -> Optional[object]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = content[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
