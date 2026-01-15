from __future__ import annotations

import re
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Dict, Optional


class GitPatchContext(AbstractContextManager):
    def __init__(
        self,
        repo_root: Path,
        exp_id: str,
        artifacts_dir: Path,
        input_script: Path,
        input_edit: Optional[Dict[str, object]],
        allowlist: list[str],
    ) -> None:
        self.repo_root = repo_root
        self.exp_id = exp_id
        self.artifacts_dir = artifacts_dir
        self.input_script = input_script
        self.input_edit = input_edit
        self.allowlist = allowlist
        self.worktree_dir = artifacts_dir / "worktrees" / exp_id
        self.branch_name = f"exp/{exp_id}"
        self.patch_path = artifacts_dir / "patch.diff"
        self.git_commit_before: Optional[str] = None
        self.git_commit_after: Optional[str] = None

    def __enter__(self) -> "GitPatchContext":
        self._ensure_git()
        self.worktree_dir.parent.mkdir(parents=True, exist_ok=True)
        self._git(["worktree", "add", "-b", self.branch_name, str(self.worktree_dir), "HEAD"])
        self._git(["-C", str(self.worktree_dir), "submodule", "update", "--init", "--recursive"])

        if self.input_edit:
            self._apply_input_edit()
            patch_repo = self._repo_root_for_path(self.map_to_worktree(self.input_script))
            self.git_commit_before = self._git(["-C", str(patch_repo), "rev-parse", "HEAD"]).strip()
            diff = self._git(["-C", str(patch_repo), "diff", "--binary"])
            self.patch_path.write_text(diff, encoding="utf-8")
        else:
            self.git_commit_before = self._git(["rev-parse", "HEAD"]).strip()
            self.patch_path.write_text("", encoding="utf-8")

        self.git_commit_after = self.git_commit_before
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._git(["worktree", "remove", "-f", str(self.worktree_dir)])
        finally:
            self._git(["branch", "-D", self.branch_name], check=False)

    def _ensure_git(self) -> None:
        try:
            self._git(["rev-parse", "--is-inside-work-tree"])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Git repo not initialized. Run git init and commit a baseline.") from exc

    def _git(self, args: list[str], check: bool = True) -> str:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(self.repo_root),
            check=check,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def _apply_input_edit(self) -> None:
        directive = str(self.input_edit.get("directive", ""))
        if directive not in self.allowlist:
            raise RuntimeError(f"Directive '{directive}' not allowed by policy.")

        mode = self.input_edit.get("mode")
        match = self.input_edit.get("match")
        replace = self.input_edit.get("replace")
        if mode != "replace_line" or not match or replace is None:
            raise RuntimeError("Unsupported input_edit configuration.")

        worktree_script = self.map_to_worktree(self.input_script)
        text = worktree_script.read_text(encoding="utf-8")
        new_text, count = re.subn(match, str(replace), text, count=1, flags=re.MULTILINE)
        if count == 0:
            raise RuntimeError("Input edit pattern not found.")
        worktree_script.write_text(new_text, encoding="utf-8")

    def map_to_worktree(self, path: Path) -> Path:
        rel = path.resolve().relative_to(self.repo_root.resolve())
        return (self.worktree_dir / rel).resolve()

    def _repo_root_for_path(self, path: Path) -> Path:
        result = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip())


def get_git_head(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None
