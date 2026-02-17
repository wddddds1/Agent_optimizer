from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from schemas.action_ir import ActionIR
    from schemas.job_ir import JobIR


def _app_module_name(app: str) -> str:
    # Convert arbitrary app names to python module-friendly suffixes.
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", str(app or "").strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return f"{normalized}_app" if normalized else ""


def _load_app_module(app: str):
    module_name = _app_module_name(app)
    if not module_name:
        return None
    try:
        return importlib.import_module(f"skills.applications.{module_name}")
    except ModuleNotFoundError:
        return None


def apply_adapter(
    action: "ActionIR",
    job: "JobIR",
    adapter_cfg: Optional[Dict[str, object]] = None,
) -> "ActionIR":
    module = _load_app_module(job.app)
    if module is not None and hasattr(module, "apply_adapter"):
        return module.apply_adapter(action, job, adapter_cfg)
    return action


def input_edit_allowlist(app: str) -> List[str]:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "input_edit_allowlist"):
        result = module.input_edit_allowlist()
        if isinstance(result, list):
            return result
    return []


def parse_timing_breakdown(app: str, log_text: str) -> Dict[str, float]:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "parse_timing_breakdown"):
        result = module.parse_timing_breakdown(log_text)
        if isinstance(result, dict):
            return result
    return {}


def requires_structured_correctness(app: str) -> bool:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "requires_structured_correctness"):
        return bool(module.requires_structured_correctness())
    return False


def supports_agentic_correctness(app: str) -> bool:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "supports_agentic_correctness"):
        return bool(module.supports_agentic_correctness())
    return True


def ensure_log_path(app: str, run_args: List[str], run_dir: Path) -> List[str]:
    module = _load_app_module(app)
    if module is not None and hasattr(module, "ensure_log_path"):
        result = module.ensure_log_path(run_args, run_dir)
        if isinstance(result, list):
            return [str(arg) for arg in result]
    return list(run_args)


__all__ = [
    "apply_adapter",
    "ensure_log_path",
    "input_edit_allowlist",
    "parse_timing_breakdown",
    "requires_structured_correctness",
    "supports_agentic_correctness",
]
