from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Dict, List, Optional


class WorktreeAddError(RuntimeError):
    def __init__(self, message: str, attempts: int, last_error: str) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def _compact_git_token(raw: str, max_len: int = 72) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip())
    slug = re.sub(r"_+", "_", slug).strip("._-") or "exp"
    digest = hashlib.sha1(str(raw or "").encode("utf-8")).hexdigest()[:10]
    keep = max_len - len(digest) - 1
    if keep < 8:
        keep = 8
    if len(slug) > keep:
        slug = slug[:keep].rstrip("._-")
    return f"{slug}-{digest}"


class GitPatchContext(AbstractContextManager):
    def __init__(
        self,
        repo_root: Path,
        exp_id: str,
        artifacts_dir: Path,
        input_script: Path,
        input_edit: Optional[Dict[str, object]],
        allowlist: list[str],
        patch_path: Optional[Path] = None,
        patch_paths: Optional[List[Path]] = None,
        patch_root: Optional[Path] = None,
        worktree_retries: int = 2,
    ) -> None:
        self.repo_root = repo_root
        self.exp_id = exp_id
        self.artifacts_dir = artifacts_dir
        self.input_script = input_script
        self.input_edit = input_edit
        self.allowlist = allowlist
        self.patch_source_paths: List[Path] = []
        for candidate in patch_paths or []:
            if isinstance(candidate, Path):
                self.patch_source_paths.append(candidate)
        if patch_path and patch_path not in self.patch_source_paths:
            self.patch_source_paths.append(patch_path)
        self.patch_root = patch_root
        self.worktree_token = _compact_git_token(exp_id)
        self.worktree_dir = artifacts_dir / "worktrees" / self.worktree_token
        self.branch_name = f"exp/{self.worktree_token}"
        self.patch_path = artifacts_dir / "patch.diff"
        self.git_commit_before: Optional[str] = None
        self.git_commit_after: Optional[str] = None
        self.submodule_worktrees: list[tuple[Path, Path]] = []
        self.worktree_retries = max(1, int(worktree_retries))

    def __enter__(self) -> "GitPatchContext":
        self._ensure_git()
        self.worktree_dir.parent.mkdir(parents=True, exist_ok=True)
        self._cleanup_worktree_state()
        self._add_worktree_with_retry()
        self._init_submodule_worktrees()
        self._sync_input_script()

        if self.input_edit and self.patch_source_paths:
            raise RuntimeError("input_edit and patch_path(s) cannot be combined in one action")

        patch_repo = self.repo_root
        if self.input_edit:
            self._apply_input_edit()
            patch_repo = self._repo_root_for_path(self.map_to_worktree(self.input_script))
        elif self.patch_source_paths:
            patch_repo = self._resolve_patch_root()
            self._apply_patch_files(patch_repo, self.patch_source_paths)

        if self.input_edit or self.patch_source_paths:
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
            self._cleanup_submodule_worktrees()
            self._git(["worktree", "remove", "-f", str(self.worktree_dir)])
        finally:
            self._git(["branch", "-D", self.branch_name], check=False)

    def _ensure_git(self) -> None:
        try:
            self._git(["rev-parse", "--is-inside-work-tree"])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Git repo not initialized. Run git init and commit a baseline.") from exc

    def _cleanup_worktree_state(self) -> None:
        self._git(["worktree", "prune", "--expire", "now"], check=False)
        self._git(["worktree", "remove", "-f", str(self.worktree_dir)], check=False)
        if self.worktree_dir.exists():
            shutil.rmtree(self.worktree_dir, ignore_errors=True)
        self._git(["branch", "-D", self.branch_name], check=False)
        worktree_gitdir = self.repo_root / ".git" / "worktrees" / self.worktree_token
        if worktree_gitdir.exists():
            shutil.rmtree(worktree_gitdir, ignore_errors=True)

    def _add_worktree_with_retry(self) -> None:
        last_error = ""
        for attempt in range(1, self.worktree_retries + 1):
            try:
                self._git(
                    [
                        "worktree",
                        "add",
                        "-b",
                        self.branch_name,
                        str(self.worktree_dir),
                        "HEAD",
                    ]
                )
                return
            except subprocess.CalledProcessError as exc:
                last_error = (exc.stderr or exc.stdout or "").strip()
                self._cleanup_worktree_state()
                if attempt >= self.worktree_retries:
                    detail = f": {last_error}" if last_error else ""
                    raise WorktreeAddError(
                        f"git worktree add failed after {attempt} attempts{detail}",
                        attempts=attempt,
                        last_error=last_error,
                    ) from exc

    def _git(self, args: list[str], check: bool = True) -> str:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(self.repo_root),
            check=check,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def _link_submodule_alternates(self) -> None:
        for sub_path in self._submodule_paths():
            main_gitdir = self.repo_root / ".git" / "modules" / sub_path
            worktree_gitdir = (
                self.repo_root
                / ".git"
                / "worktrees"
                / self.worktree_token
                / "modules"
                / sub_path
            )
            if not main_gitdir.exists() or not worktree_gitdir.exists():
                continue
            alternates = worktree_gitdir / "objects" / "info" / "alternates"
            alternates.parent.mkdir(parents=True, exist_ok=True)
            alternates.write_text(str(main_gitdir / "objects") + "\n", encoding="utf-8")

    def _submodule_paths(self) -> list[Path]:
        gitmodules = self.repo_root / ".gitmodules"
        if not gitmodules.exists():
            return []
        paths: list[Path] = []
        for line in gitmodules.read_text(encoding="utf-8").splitlines():
            if "path" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != "path":
                continue
            path = value.strip()
            if path:
                paths.append(Path(path))
        return paths

    def _submodule_commit(self, sub_path: Path) -> Optional[str]:
        result = subprocess.run(
            ["git", "-C", str(self.worktree_dir), "ls-tree", "HEAD", str(sub_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        parts = result.stdout.strip().split()
        if len(parts) >= 3:
            return parts[2]
        return None

    def _init_submodule_worktrees(self) -> None:
        for sub_path in self._submodule_paths():
            commit = self._submodule_commit(sub_path)
            if not commit:
                continue
            source_repo = (self.repo_root / sub_path).resolve()
            if not source_repo.exists():
                raise RuntimeError(f"Submodule not initialized: {sub_path}")
            target = (self.worktree_dir / sub_path).resolve()
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            subprocess.run(
                ["git", "-C", str(source_repo), "worktree", "prune", "--expire", "now"],
                check=False,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_repo),
                    "worktree",
                    "add",
                    "--detach",
                    str(target),
                    commit,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.submodule_worktrees.append((source_repo, target))

    def _cleanup_submodule_worktrees(self) -> None:
        for source_repo, target in self.submodule_worktrees:
            subprocess.run(
                ["git", "-C", str(source_repo), "worktree", "remove", "-f", str(target)],
                check=False,
                capture_output=True,
                text=True,
            )
        self.submodule_worktrees.clear()

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

    def _sync_input_script(self) -> None:
        try:
            source_path = self.input_script.resolve()
        except FileNotFoundError:
            return
        if not source_path.exists():
            return
        try:
            rel = source_path.relative_to(self.repo_root.resolve())
        except ValueError:
            return
        target = (self.worktree_dir / rel).resolve()
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)

    def map_to_worktree(self, path: Path) -> Path:
        resolved = path.resolve()
        repo_resolved = self.repo_root.resolve()
        try:
            rel = resolved.relative_to(repo_resolved)
        except ValueError:
            # Path is not under repo_root — likely inside a previous worktree.
            # Strip the worktree prefix to recover the repo-relative path.
            path_str = str(resolved)
            repo_str = str(repo_resolved)
            wt_marker = "/worktrees/"
            idx = path_str.find(wt_marker)
            if idx != -1:
                # Path looks like <prefix>/worktrees/<run_id>/<repo_relative>
                after_marker = path_str[idx + len(wt_marker):]
                # Skip the run_id component
                slash = after_marker.find("/")
                if slash != -1:
                    rel = Path(after_marker[slash + 1:])
                else:
                    raise ValueError(
                        f"Cannot resolve {resolved} relative to {repo_resolved}"
                    )
            else:
                raise ValueError(
                    f"Cannot resolve {resolved} relative to {repo_resolved}"
                )
        return (self.worktree_dir / rel).resolve()

    def _repo_root_for_path(self, path: Path) -> Path:
        result = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip())

    def _apply_patch_file(self, patch_repo: Path, patch_path: Path) -> None:
        if not patch_path.exists():
            raise RuntimeError(f"Patch file not found: {patch_path}")
        patch_to_apply = patch_path
        if self.patch_root:
            prefix = self.patch_root.as_posix().rstrip("/") + "/"
            patch_text = patch_path.read_text(encoding="utf-8")
            if _patch_has_prefix(patch_text, prefix):
                patch_text = _strip_patch_prefix(patch_text, prefix)
                patch_to_apply = patch_path.with_name("patch_root_adjusted.diff")
                patch_to_apply.write_text(patch_text, encoding="utf-8")
        result = subprocess.run(
            ["git", "-C", str(patch_repo), "apply", str(patch_to_apply)],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to apply patch: {patch_to_apply}")

    def _apply_patch_files(self, patch_repo: Path, patch_paths: List[Path]) -> None:
        for patch_path in patch_paths:
            self._apply_patch_file(patch_repo, patch_path)

    def _resolve_patch_root(self) -> Path:
        if not self.patch_root:
            return self.worktree_dir
        patch_root = (self.worktree_dir / self.patch_root).resolve()
        try:
            patch_root.relative_to(self.worktree_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Patch root escapes worktree: {patch_root}") from exc
        if not patch_root.exists():
            raise RuntimeError(f"Patch root not found: {patch_root}")
        return patch_root


def _strip_patch_prefix(patch_text: str, prefix: str) -> str:
    lines = []
    for line in patch_text.splitlines():
        if line.startswith(("--- ", "+++ ")):
            head = line[:4]
            path = line[4:]
            lead = ""
            rest = path
            if path.startswith(("a/", "b/")):
                lead = path[:2]
                rest = path[2:]
            if rest.startswith(prefix):
                rest = rest[len(prefix) :]
            line = f"{head}{lead}{rest}"
        lines.append(line)
    return "\n".join(lines) + ("\n" if patch_text.endswith("\n") else "")


def _patch_has_prefix(patch_text: str, prefix: str) -> bool:
    prefixes = (
        f"--- a/{prefix}",
        f"+++ b/{prefix}",
        f"--- {prefix}",
        f"+++ {prefix}",
    )
    for line in patch_text.splitlines():
        if line.startswith(prefixes):
            return True
    return False


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


def get_git_status(repo_root: Path) -> Dict[str, Optional[bool]]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = bool(result.stdout.strip())
        return {"dirty": dirty}
    except Exception:
        return {"dirty": None}
