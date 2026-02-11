from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Callable

from schemas.action_ir import ActionIR
from schemas.experiment_ir import ExperimentIR
from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from schemas.result_ir import ResultIR
from orchestrator.llm_client import LLMClient
from skills.verify import verify_run


class VerifierAgent:
    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client

    def verify(
        self,
        job: JobIR,
        action: Optional[ActionIR],
        result: ResultIR,
        profile: ProfileReport,
        gates: Dict[str, object],
        baseline_exp: Optional[ExperimentIR],
        is_final_validation: bool = False,
    ):
        correctness_cfg = gates.get("correctness", {})
        agentic_cfg = correctness_cfg.get("agentic", {}) if isinstance(correctness_cfg, dict) else {}
        use_agent = bool(agentic_cfg.get("enabled", False)) and self.llm_client and self.llm_client.config.enabled

        agentic_decider: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None
        if use_agent:
            agentic_decider = lambda payload: self._agentic_decide(payload)
        return verify_run(
            job,
            action,
            result,
            profile,
            gates,
            baseline_exp,
            is_final_validation=is_final_validation,
            agentic_decider=agentic_decider,
            agentic_cfg=agentic_cfg if use_agent else None,
        )

    def _agentic_decide(self, payload: Dict[str, object]) -> Dict[str, object]:
        prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / "correctness_agent.md"
        prompt = prompt_path.read_text(encoding="utf-8")
        data = self.llm_client.request_json(prompt, payload) if self.llm_client else None
        if not isinstance(data, dict):
            return {
                "verdict": "FAIL",
                "rationale": "agent returned no decision",
                "confidence": 0.0,
                "allowed_drift": {"policy": "none", "notes": "agent failure"},
            }
        verdict = str(data.get("verdict", "")).upper()
        if verdict not in {"PASS", "FAIL", "NEED_MORE_CONTEXT"}:
            return {
                "verdict": "FAIL",
                "rationale": "agent decision invalid",
                "confidence": 0.0,
                "allowed_drift": {"policy": "none", "notes": "invalid verdict"},
            }
        data["verdict"] = verdict
        if "confidence" not in data:
            data["confidence"] = 0.5
        if "rationale" not in data:
            data["rationale"] = "agent decision"
        if "allowed_drift" not in data:
            data["allowed_drift"] = {"policy": "unspecified", "notes": ""}
        return data
