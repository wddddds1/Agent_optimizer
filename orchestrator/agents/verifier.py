from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
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
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        contract_store_path: Optional[Path] = None,
    ) -> None:
        self.llm_client = llm_client
        self._contract_store_path: Optional[Path] = None
        self._contract_store: Dict[str, object] = {"version": 1, "contracts": {}}
        if contract_store_path is not None:
            self.configure_contract_store(contract_store_path)

    def configure_contract_store(self, path: Path) -> None:
        self._contract_store_path = path
        self._load_contract_store()

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
            contract_getter=self._get_contract,
            contract_putter=self._put_contract,
        )

    def _job_fingerprint(self, job: JobIR) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "app": job.app,
            "case_id": job.case_id,
            "run_args": list(job.run_args or []),
            "env_keys": sorted((job.env or {}).keys()),
            "app_bin": str(job.app_bin or ""),
            "input_script": str(job.input_script or ""),
        }
        app_bin = Path(job.app_bin) if job.app_bin else None
        if app_bin and app_bin.exists():
            stat = app_bin.stat()
            payload["app_bin_size"] = int(stat.st_size)
            payload["app_bin_mtime_ns"] = int(stat.st_mtime_ns)
        input_script = Path(job.input_script) if job.input_script else None
        if input_script and input_script.exists():
            stat = input_script.stat()
            payload["input_size"] = int(stat.st_size)
            payload["input_mtime_ns"] = int(stat.st_mtime_ns)
        return payload

    def _job_key(self, job: JobIR) -> str:
        payload = self._job_fingerprint(job)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def _load_contract_store(self) -> None:
        path = self._contract_store_path
        if path is None:
            return
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("contracts"), dict):
                    self._contract_store = data
                    return
        except Exception:
            pass
        self._contract_store = {"version": 1, "contracts": {}}

    def _save_contract_store(self) -> None:
        path = self._contract_store_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._contract_store, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            pass

    def _get_contract(self, job: JobIR) -> Optional[Dict[str, object]]:
        contracts = self._contract_store.get("contracts", {})
        if not isinstance(contracts, dict):
            return None
        item = contracts.get(self._job_key(job))
        if isinstance(item, dict):
            contract = item.get("contract")
            if isinstance(contract, dict):
                return contract
        return None

    def _put_contract(self, job: JobIR, contract: Dict[str, object]) -> None:
        contracts = self._contract_store.get("contracts")
        if not isinstance(contracts, dict):
            contracts = {}
            self._contract_store["contracts"] = contracts
        key = self._job_key(job)
        contracts[key] = {
            "fingerprint": self._job_fingerprint(job),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "contract": contract,
        }
        self._save_contract_store()

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
