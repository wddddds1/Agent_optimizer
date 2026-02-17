from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import ValidationError

from orchestrator.agent_llm import AgentConfig, AgentLLMClient, AgentSession
from schemas.decision_ir import DecisionIR
from skills.parameter_explorer_tools import ParameterExplorerTools


@dataclass
class DecisionResult:
    status: str
    decision: Optional[DecisionIR]
    conversation_log: List[Dict[str, Any]]
    total_turns: int
    total_tokens: int


class OrchestratorAgent:
    def __init__(
        self,
        config: AgentConfig,
        repo_root: Path,
        input_script_path: Optional[Path] = None,
        experience_db: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root
        self.llm_client = AgentLLMClient(config)
        self.tools = ParameterExplorerTools(
            repo_root, input_script_path, experience_db
        )
        self.last_decision: Optional[DecisionIR] = None

    def decide(
        self,
        profile: Dict[str, Any],
        action_space: Dict[str, Any],
        context: Dict[str, Any],
    ) -> DecisionResult:
        self.tools.set_profile_data(profile)
        self.tools.set_action_space(action_space)

        system_prompt = _load_prompt("orchestrator")
        session = self.llm_client.create_session(system_prompt)
        self.llm_client.register_tools(session, self.tools.get_all_tools())

        user_message = _build_user_message(context)

        try:
            response = self.llm_client.chat(
                session, user_message, auto_execute_tools=True
            )
            decision = _parse_decision(response, session)
            allowed_cids = _extract_allowed_cids(context)
            if decision and allowed_cids and not decision.stop:
                invalid = _find_invalid_cids(decision, allowed_cids)
                missing_cids = not (decision.candidate_cids or decision.ranking_cids)
                if invalid or missing_cids:
                    repair_message = _build_cid_repair_message(invalid, allowed_cids, missing_cids)
                    repair_response = self.llm_client.chat(
                        session,
                        repair_message,
                        auto_execute_tools=False,
                    )
                    repaired = _parse_decision(repair_response, session)
                    if repaired:
                        decision = repaired
                    if (
                        not decision
                        or not (decision.candidate_cids or decision.ranking_cids)
                        or _find_invalid_cids(decision, allowed_cids)
                    ):
                        decision = None
            self.last_decision = decision
            return DecisionResult(
                status=decision.status if decision else "ERROR",
                decision=decision,
                conversation_log=_extract_conversation_log(session),
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )
        except Exception:
            if self.config.strict_availability:
                raise
            return DecisionResult(
                status="ERROR",
                decision=None,
                conversation_log=_extract_conversation_log(session),
                total_turns=session.turn_count,
                total_tokens=session.total_tokens,
            )


def _load_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "agents" / f"{name}.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def _build_user_message(context: Dict[str, Any]) -> str:
    payload = json.dumps(context, ensure_ascii=False)
    return "Payload:\n" + payload


def _extract_decision_payload(session: AgentSession) -> Optional[Dict[str, Any]]:
    for msg in reversed(session.messages):
        if msg.role != "assistant" or not msg.content:
            continue
        content = msg.content.strip()
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(content[start : end + 1])
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
    return None


def _extract_allowed_cids(context: Dict[str, Any]) -> Set[int]:
    allowed: Set[int] = set()
    actions = context.get("available_actions", [])
    if not isinstance(actions, list):
        return allowed
    for item in actions:
        if not isinstance(item, dict):
            continue
        cid = item.get("cid")
        try:
            if cid is not None:
                allowed.add(int(cid))
        except (TypeError, ValueError):
            continue
    return allowed


def _find_invalid_cids(decision: DecisionIR, allowed_cids: Set[int]) -> List[int]:
    seen: Set[int] = set()
    invalid: List[int] = []
    for cid in list(decision.candidate_cids or []) + list(decision.ranking_cids or []):
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        if cid_int in seen:
            continue
        seen.add(cid_int)
        if cid_int not in allowed_cids:
            invalid.append(cid_int)
    return invalid


def _build_cid_repair_message(
    invalid: List[int],
    allowed: Set[int],
    missing_cids: bool,
) -> str:
    allowed_sorted = sorted(allowed)
    issue_line = "Missing candidate_cids/ranking_cids in previous JSON." if missing_cids else ""
    return (
        "Validation error: your previous JSON used invalid cids.\n"
        f"{issue_line}\n"
        f"Invalid cids: {invalid}\n"
        f"Allowed cid values: {allowed_sorted}\n"
        "Return a corrected full DecisionIR JSON now.\n"
        "Rules:\n"
        "1) Use only candidate_cids/ranking_cids from allowed cid values.\n"
        "2) If unsure, return status=\"PARTIAL\" with conservative candidate_cids.\n"
        "3) Do not include markdown.\n"
    )


def _parse_decision(response: str, session: AgentSession) -> Optional[DecisionIR]:
    raw = _extract_decision_payload(session)
    if not raw:
        return None
    try:
        return DecisionIR(**raw)
    except ValidationError:
        return None


def _extract_conversation_log(session: AgentSession) -> List[Dict[str, Any]]:
    log: List[Dict[str, Any]] = []
    for msg in session.messages:
        entry: Dict[str, Any] = {"role": msg.role}
        if msg.content:
            entry["content"] = msg.content
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        log.append(entry)
    return log
