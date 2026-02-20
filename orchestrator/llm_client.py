from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import openai

from orchestrator.errors import LLMUnavailableError
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
    strict_availability: bool = True
    request_timeout_sec: float = 60.0
    api_timeout_retries: int = 2


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.client = None
        self._prefer_max_completion_tokens = False
        self._prefer_default_temperature = False
        self._request_timeout_sec = max(0.0, float(config.request_timeout_sec or 0.0))
        self._api_timeout_retries = max(0, int(config.api_timeout_retries or 0))
        if not config.enabled:
            return
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise LLMUnavailableError(f"Missing API key in env var {config.api_key_env}")
        client_cls = getattr(openai, "Client", None) or getattr(openai, "OpenAI", None)
        if client_cls is None:
            raise LLMUnavailableError("OpenAI client class not available in openai package")
        self.client = client_cls(
            api_key=api_key,
            base_url=config.base_url,
            max_retries=0,
            timeout=(self._request_timeout_sec if self._request_timeout_sec > 0 else None),
        )

    def _chat_create(self, **kwargs: Any) -> Any:
        if not self.client:
            raise RuntimeError("LLM client not initialized")
        req = dict(kwargs)
        if self._request_timeout_sec > 0 and "timeout" not in req:
            req["timeout"] = self._request_timeout_sec
        if self._prefer_max_completion_tokens:
            _promote_max_completion_tokens(req)
        if self._prefer_default_temperature:
            _promote_default_temperature(req)
        last_exc: Optional[Exception] = None
        attempts = 3 + self._api_timeout_retries
        for attempt in range(attempts):
            try:
                return self.client.chat.completions.create(**req)
            except Exception as exc:
                last_exc = exc
                changed = False
                if _needs_max_completion_tokens(exc):
                    before = dict(req)
                    _promote_max_completion_tokens(req)
                    if req != before:
                        self._prefer_max_completion_tokens = True
                        changed = True
                if _needs_default_temperature(exc):
                    if _promote_default_temperature(req):
                        self._prefer_default_temperature = True
                        changed = True
                if changed:
                    continue
                if _is_retryable_llm_error(exc) and attempt < attempts - 1:
                    time.sleep(1.0 + 0.5 * attempt)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")

    def preflight_check(self) -> None:
        """Fail fast if the configured model is not accessible."""
        if not self.config.enabled:
            return
        if not self.client:
            raise LLMUnavailableError("LLM client not initialized")
        try:
            # Prefer model metadata probe (cheap, no generation tokens).
            self.client.models.retrieve(self.config.model)
            return
        except Exception as exc:
            msg = str(exc)
            # Some compatible providers may not implement models.retrieve.
            # Fallback to a tiny completion probe in that case.
            if "404" not in msg and "not found" not in msg.lower():
                raise LLMUnavailableError(
                    f"Model preflight failed for {self.config.model}: {exc}"
                ) from exc
        try:
            self._chat_create(
                model=self.config.model,
                temperature=0.0,
                max_tokens=1,
                messages=[
                    {"role": "system", "content": "Reply with OK."},
                    {"role": "user", "content": "OK"},
                ],
            )
        except Exception as exc:
            hint = _chat_api_compat_hint(exc, self.config.model)
            raise LLMUnavailableError(
                f"Model preflight failed for {self.config.model}: {hint}"
            ) from exc

    def rank_actions(
        self,
        actions: List[ActionIR],
        profile: ProfileReport,
        context: Dict[str, object],
    ) -> List[str]:
        if not self.client:
            if self.config.enabled and self.config.strict_availability:
                raise LLMUnavailableError("LLM client not initialized")
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
        try:
            response = self._chat_create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": "Return only JSON. No markdown."},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            if self.config.strict_availability:
                hint = _chat_api_compat_hint(exc, self.config.model)
                raise LLMUnavailableError(f"LLM rank_actions request failed: {hint}") from exc
            return []
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
            if self.config.enabled and self.config.strict_availability:
                raise LLMUnavailableError("LLM client not initialized")
            return ""
        prompt = (
            "Explain the action ranking rationale in 3-5 bullet points. "
            "Do not mention missing info."
        )
        try:
            response = self._chat_create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": "Return plain text only."},
                    {"role": "user", "content": prompt + "\n\nContext:\n" + json.dumps(context)},
                ],
            )
        except Exception as exc:
            if self.config.strict_availability:
                hint = _chat_api_compat_hint(exc, self.config.model)
                raise LLMUnavailableError(f"LLM explain_decision request failed: {hint}") from exc
            return ""
        return response.choices[0].message.content.strip()

    def summarize_report_zh(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not self.client:
            if self.config.enabled and self.config.strict_availability:
                raise LLMUnavailableError("LLM client not initialized")
            return None
        prompt = (
            "你是性能优化审计员。请根据给定实验数据，输出一个 JSON 对象，包含以下键：\n"
            "- experiment_reasons: {run_id: 中文简要原因}\n"
            "- experiment_details: {run_id: 中文详细分析段落（不少于4句，需包含：做了什么、为什么、预期、实测结果与简短结论）}\n"
            "- overall_analysis: [\"中文分析句子\", ...] (3-6 条)\n"
            "- selection_reason: \"中文简要说明为什么最终选择该优化\"\n"
            "要求：只输出 JSON，不要 markdown，不要额外文字。原因要简短、具体、可解释；不要编造未给出的数据。"
        )
        try:
            response = self._chat_create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": "Return only JSON."},
                    {"role": "user", "content": prompt + "\n\nPayload:\n" + json.dumps(payload)},
                ],
            )
        except Exception as exc:
            if self.config.strict_availability:
                hint = _chat_api_compat_hint(exc, self.config.model)
                raise LLMUnavailableError(f"LLM summarize_report_zh request failed: {hint}") from exc
            return None
        content = response.choices[0].message.content.strip()
        data = _safe_json_loads(content)
        if not isinstance(data, dict):
            return None
        return data

    def summarize_iteration_zh(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not self.client:
            if self.config.enabled and self.config.strict_availability:
                raise LLMUnavailableError("LLM client not initialized")
            return None
        prompt = (
            "你是性能优化审计员。请根据给定本轮迭代数据，输出一个 JSON 对象，包含以下键：\n"
            "- experiment_reasons: {run_id: 中文简要原因}\n"
            "- experiment_details: {run_id: 中文详细分析段落（不少于4句，需包含：做了什么、为什么、预期、实测结果与简短结论）}\n"
            "- summary_lines: [\"中文分析句子\", ...] (2-5 条)\n"
            "- selection_reason: \"中文简要说明本轮为什么选择该最优配置\"\n"
            "要求：只输出 JSON，不要 markdown，不要额外文字。原因要简短、具体、可解释；不要编造未给出的数据。"
        )
        try:
            response = self._chat_create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": "Return only JSON."},
                    {"role": "user", "content": prompt + "\n\nPayload:\n" + json.dumps(payload)},
                ],
            )
        except Exception as exc:
            if self.config.strict_availability:
                hint = _chat_api_compat_hint(exc, self.config.model)
                raise LLMUnavailableError(
                    f"LLM summarize_iteration_zh request failed: {hint}"
                ) from exc
            return None
        content = response.choices[0].message.content.strip()
        data = _safe_json_loads(content)
        if not isinstance(data, dict):
            return None
        return data

    def request_json(self, prompt: str, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        if not self.client:
            if self.config.enabled and self.config.strict_availability:
                raise LLMUnavailableError("LLM client not initialized")
            return None
        message = prompt.rstrip() + "\n\nPayload:\n" + json.dumps(payload, ensure_ascii=False)

        def _request_once(user_content: str) -> str:
            attempts = 3
            last_exc: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    response = self._chat_create(
                        model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        messages=[
                            {"role": "system", "content": "Return only JSON. No markdown or extra text."},
                            {"role": "user", "content": user_content},
                        ],
                    )
                    return (response.choices[0].message.content or "").strip()
                except Exception as exc:
                    last_exc = exc
                    if not _is_retryable_llm_error(exc) or attempt >= attempts - 1:
                        raise
                    time.sleep(1.0 + attempt)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("LLM request failed without exception details")

        try:
            content = _request_once(message)
        except Exception as exc:
            if self.config.strict_availability:
                hint = _chat_api_compat_hint(exc, self.config.model)
                raise LLMUnavailableError(f"LLM request_json failed: {hint}") from exc
            return None
        data = _safe_json_loads(content)
        if isinstance(data, dict):
            return data

        # Strict mode: force one structured retry for malformed/truncated JSON.
        if self.config.strict_availability:
            retry_prompt = (
                "Your previous response was invalid or truncated JSON.\n"
                "Return exactly ONE valid JSON object now. No markdown fences.\n"
                "Keep the same schema requested by the prompt.\n\n"
                f"Original request:\n{message}\n\n"
                f"Previous invalid response (for reference):\n{content[:1200]}"
            )
            try:
                retry_content = _request_once(retry_prompt)
            except Exception as exc:
                hint = _chat_api_compat_hint(exc, self.config.model)
                raise LLMUnavailableError(f"LLM request_json retry failed: {hint}") from exc
            retry_data = _safe_json_loads(retry_content)
            if isinstance(retry_data, dict):
                return retry_data
            preview = retry_content[:240].replace("\n", " ")
            raise LLMUnavailableError(
                f"LLM request_json returned non-JSON response for model {self.config.model}: {preview}"
            )

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


def _is_retryable_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_markers = (
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "service unavailable",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    )
    return any(marker in text for marker in retry_markers)


def _chat_api_compat_hint(exc: Exception, model: str) -> str:
    text = str(exc)
    lower = text.lower()
    if _needs_default_temperature(exc):
        return (
            f"{text} | model '{model}' only supports default temperature=1 "
            "for chat.completions."
        )
    if _needs_max_completion_tokens(exc):
        return (
            f"{text} | model '{model}' expects max_completion_tokens for "
            "chat.completions. Retrying with max_completion_tokens may fix this."
        )
    if (
        "only supported in v1/responses" in lower
        or ("v1/responses" in lower and "chat/completions" in lower)
    ):
        return (
            f"{text} | model '{model}' is Responses-API-only. "
            "Use a chat.completions-compatible model (e.g. gpt-4.1-mini) "
            "or migrate client calls to Responses API."
        )
    return text


def _needs_max_completion_tokens(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "unsupported parameter" in text
        and "max_tokens" in text
        and "max_completion_tokens" in text
    )


def _promote_max_completion_tokens(payload: Dict[str, Any]) -> None:
    if "max_tokens" in payload and "max_completion_tokens" not in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")


def _needs_default_temperature(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "temperature" in text
        and "unsupported" in text
        and "default (1)" in text
    )


def _promote_default_temperature(payload: Dict[str, Any]) -> bool:
    if "temperature" in payload and payload.get("temperature") != 1:
        payload["temperature"] = 1
        return True
    return False
