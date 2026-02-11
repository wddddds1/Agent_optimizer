from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from schemas.action_ir import ActionIR
from schemas.job_ir import JobIR
from schemas.profile_report import ProfileReport
from orchestrator.llm_client import LLMClient


def load_yaml(path: Path) -> Dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_action_space(path: Path) -> List[ActionIR]:
    data = load_yaml(path)
    actions = [ActionIR(**item) for item in data.get("actions", [])]
    return actions


def load_direction_space(path: Path) -> List[Dict[str, object]]:
    data = load_yaml(path)
    directions = data.get("directions", [])
    return [direction for direction in directions if isinstance(direction, dict)]


def load_policy(path: Path) -> Dict[str, object]:
    return load_yaml(path)


def load_gates(path: Path) -> Dict[str, object]:
    return load_yaml(path)


@dataclass
class RuleContext:
    job: JobIR
    input_text: str
    run_args: List[str]
    env: Dict[str, str]


def evaluate_rule(rule: str, ctx: RuleContext) -> bool:
    if ":" not in rule:
        return False
    key, value = rule.split(":", 1)
    value = value.strip()
    if key == "input_contains":
        return bool(_regex_search(value, ctx.input_text))
    if key == "input_not_contains":
        return not _regex_search(value, ctx.input_text)
    if key == "args_contains":
        return value in ctx.run_args
    if key == "args_not_contains":
        return value not in ctx.run_args
    if key == "env_contains":
        return value in ctx.env
    if key == "env_not_contains":
        return value not in ctx.env
    if key == "app_is":
        return ctx.job.app == value
    if key == "case_tag":
        return value in ctx.job.tags
    return False


def _regex_search(pattern: str, text: str) -> bool:
    try:
        import re

        return re.search(pattern, text, re.MULTILINE) is not None
    except re.error:
        return False


def filter_actions(
    actions: List[ActionIR],
    ctx: RuleContext,
    allowed_families: Optional[List[str]],
    policy: Dict[str, object],
) -> List[ActionIR]:
    filtered: List[ActionIR] = []
    conditional_rules = policy.get("conditional_rules", [])
    global_rules = {rule.get("rule") for rule in policy.get("global_rules", [])}
    forbid_env = policy.get("forbid_env", {})

    for action in actions:
        if "no_mixed_targets" in global_rules:
            if "run_config" in action.applies_to and "input_script" in action.applies_to:
                continue
        if allowed_families and action.family not in allowed_families:
            continue
        if not all(evaluate_rule(rule, ctx) for rule in action.preconditions):
            continue
        if any(evaluate_rule(rule, ctx) for rule in action.constraints):
            continue
        if forbid_env and isinstance(action.parameters, dict):
            env = action.parameters.get("env") or {}
            if isinstance(env, dict):
                blocked = False
                for key, values in forbid_env.items():
                    if key not in env:
                        continue
                    val = str(env.get(key)).strip().lower()
                    deny = {str(v).strip().lower() for v in (values or [])}
                    if val in deny:
                        blocked = True
                        break
                if blocked:
                    continue
        if _violates_conditional_rules(action, ctx, conditional_rules):
            continue
        filtered.append(action)
    return filtered


def _violates_conditional_rules(
    action: ActionIR, ctx: RuleContext, rules: List[Dict[str, object]]
) -> bool:
    for rule in rules:
        when = rule.get("when")
        if when and evaluate_rule(str(when), ctx):
            families = rule.get("forbid_action_families", [])
            ids = rule.get("forbid_action_ids", [])
            if action.family in families or action.action_id in ids:
                return True
    return False


def rank_actions(actions: List[ActionIR], profile: ProfileReport) -> List[ActionIR]:
    def score(action: ActionIR) -> float:
        timing = profile.timing_breakdown
        total = timing.get("total", 0.0) or 0.0
        comm_ratio = (timing.get("comm", 0.0) / total) if total else 0.0
        output_ratio = (timing.get("output", 0.0) / total) if total else 0.0
        cpu = profile.system_metrics.get("cpu_percent_avg", 100.0)
        score_val = 0.0
        if comm_ratio > 0.2 and "comm_reduce" in action.expected_effect:
            score_val += 2.0
        if output_ratio > 0.2 and "io_reduce" in action.expected_effect:
            score_val += 2.0
        if cpu < 70.0 and (
            "compute_opt" in action.expected_effect or "mem_locality" in action.expected_effect
        ):
            score_val += 1.0
        if action.risk_level == "low":
            score_val += 0.5
        elif action.risk_level == "high":
            score_val -= 0.5
        return score_val

    return sorted(actions, key=lambda a: (-score(a), a.action_id))


def rank_actions_with_llm(
    actions: List[ActionIR],
    profile: ProfileReport,
    context: Dict[str, object],
    llm_client: LLMClient,
) -> List[ActionIR]:
    llm_ids = llm_client.rank_actions(actions, profile, context)
    if not llm_ids:
        return rank_actions(actions, profile)
    order = {action_id: idx for idx, action_id in enumerate(llm_ids)}
    return sorted(actions, key=lambda a: (order.get(a.action_id, len(order)), a.action_id))


def llm_rank_actions(context: Dict[str, object]) -> List[str]:
    return []


def llm_explain_decision(context: Dict[str, object]) -> str:
    return ""
