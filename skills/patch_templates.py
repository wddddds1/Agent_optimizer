from __future__ import annotations

import difflib
from pathlib import Path
from typing import Dict, Optional

from schemas.action_ir import ActionIR


def materialize_patch_template(
    action: ActionIR,
    repo_root: Path,
    run_dir: Path,
    adapter_cfg: Optional[Dict[str, object]] = None,
) -> Optional[Path]:
    if "source_patch" not in (action.applies_to or []):
        return None
    params = action.parameters or {}
    if params.get("patch_path"):
        return None

    patch_family = str(params.get("patch_family") or action.family or "")
    cfg = adapter_cfg.get("patch_templates", {}) if isinstance(adapter_cfg, dict) else {}
    family_overrides = cfg.get("family_overrides", {}) if isinstance(cfg, dict) else {}
    target = family_overrides.get(patch_family) if isinstance(family_overrides, dict) else None
    if not target:
        target = cfg.get("default_target")
    if not target:
        return None

    target_path = (repo_root / str(target)).resolve()
    if not target_path.exists():
        return None

    original_text = target_path.read_text(encoding="utf-8", errors="replace")
    comment = _build_comment(target_path, patch_family, action.action_id)
    new_text = _insert_comment(original_text, comment)
    if new_text == original_text:
        return None

    rel_path = target_path.relative_to(repo_root)
    diff = difflib.unified_diff(
        original_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{rel_path.as_posix()}",
        tofile=f"b/{rel_path.as_posix()}",
    )
    diff_text = "".join(diff)
    if not diff_text.strip():
        return None

    patch_path = run_dir / "patch_generated.diff"
    patch_path.write_text(diff_text, encoding="utf-8")
    return patch_path


def _build_comment(path: Path, family: str, action_id: str) -> str:
    suffix = path.suffix.lower()
    if suffix in {".f", ".f90", ".f95", ".for"}:
        return f"! HAP_PATCH {family} {action_id}"
    if suffix in {".py", ".sh"}:
        return f"# HAP_PATCH {family} {action_id}"
    return f"// HAP_PATCH {family} {action_id}"


def _insert_comment(text: str, comment: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return comment + "\n"
    if comment in text:
        return text
    insert_at = 1 if len(lines) > 1 else 0
    lines.insert(insert_at, comment + "\n")
    return "".join(lines)
