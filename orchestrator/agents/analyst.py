from __future__ import annotations

from typing import Dict, List

from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from schemas.analysis_ir import AnalysisResult
from schemas.profile_report import ProfileReport
from orchestrator.llm_client import LLMClient


class AnalystAgent:
    def __init__(self, llm_client: Optional[LLMClient]) -> None:
        self.llm_client = llm_client

    def analyze(
        self,
        profile: ProfileReport,
        history: Dict[str, object],
        policy: Dict[str, object],
        case_tags: List[str],
    ) -> AnalysisResult:
        llm_result = self._try_llm(profile, history, policy, case_tags)
        if llm_result is not None:
            return llm_result
        timing = profile.timing_breakdown or {}
        total = timing.get("total", 0.0) or 0.0
        comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
        output_ratio = (timing.get("output", 0.0) / total) if total else 0.0
        bottleneck = "compute"
        if comm_ratio > 0.2:
            bottleneck = "comm"
        if output_ratio > 0.2:
            bottleneck = "io"
        allowed = set(["run_config"])
        if "allow_build" in case_tags:
            allowed.add("build_config")
        if "allow_source_patch" in case_tags:
            allowed.add("source_patch")
        confidence = _analysis_confidence(profile)
        rationale = _build_rationale(bottleneck, comm_ratio, output_ratio, confidence)
        return AnalysisResult(
            bottleneck=bottleneck,
            allowed_families=sorted(allowed),
            allowed_transforms=[],
            forbidden_transforms=[],
            risk_overrides={},
            confidence=confidence,
            rationale=rationale,
        )

    def _try_llm(
        self,
        profile: ProfileReport,
        history: Dict[str, object],
        policy: Dict[str, object],
        case_tags: List[str],
    ) -> Optional[AnalysisResult]:
        if not self.llm_client or not self.llm_client.config.enabled:
            return None
        prompt = _load_prompt("analyst")
        payload = {
            "profile": {
                "timing_breakdown": profile.timing_breakdown,
                "system_metrics": profile.system_metrics,
            },
            "history": history,
            "policy": policy,
            "case_tags": case_tags,
        }
        data = self.llm_client.request_json(prompt, payload)
        if not data:
            return None
        try:
            result = AnalysisResult(**data)
        except ValidationError:
            return None
        if result.status != "OK":
            return None
        return result


def _analysis_confidence(profile: ProfileReport) -> float:
    timing = profile.timing_breakdown or {}
    total = timing.get("total", 0.0) or 0.0
    if total <= 0.0:
        return 0.2
    keys = ["pair", "kspace", "neigh", "comm", "modify", "output"]
    present = [key for key in keys if timing.get(key) is not None]
    positive = [key for key in keys if (timing.get(key) or 0.0) > 0.0]
    if len(positive) >= 2:
        return 0.9
    if len(present) >= 2:
        return 0.6
    return 0.3


def _build_rationale(
    bottleneck: str,
    comm_ratio: float,
    output_ratio: float,
    confidence: float,
) -> str:
    if confidence < 0.5:
        return "profiling signal weak; allow low-risk run_config only"
    if bottleneck == "comm":
        return f"comm ratio {comm_ratio:.2f} suggests communication tuning"
    if bottleneck == "io":
        return f"output ratio {output_ratio:.2f} suggests IO tuning"
    return "compute-dominant; prioritize parallel strategy and affinity"


def _load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    return path.read_text(encoding="utf-8")
