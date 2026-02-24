from pathlib import Path

from orchestrator.graph import _repo_rel_from_worktree_path, _resolve_worktree_path
from skills.patch_git import GitPatchContext


def test_repo_rel_from_worktree_path_prefers_innermost_marker() -> None:
    nested = (
        "/repo/artifacts/sessions/s1/runs/iter15/worktrees/iter15-a"
        "/artifacts/sessions/s1/runs/iter14/worktrees/iter14-b/third_party/bwa/bwt.c"
    )
    assert _repo_rel_from_worktree_path(nested) == "third_party/bwa/bwt.c"


def test_resolve_worktree_path_collapses_nested_worktree_path(tmp_path: Path) -> None:
    target = tmp_path / "third_party" / "bwa" / "bwt.c"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("int x;\n", encoding="utf-8")

    nested = (
        f"{tmp_path}/artifacts/sessions/s1/runs/iter15/worktrees/iter15-a"
        f"/artifacts/sessions/s1/runs/iter14/worktrees/iter14-b/third_party/bwa/bwt.c"
    )
    resolved = _resolve_worktree_path(nested, tmp_path)
    assert resolved == str(target.resolve())


def test_git_patch_context_map_to_worktree_strips_stale_nested_prefix(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts" / "sessions" / "s1" / "runs" / "iter16"
    ctx = GitPatchContext(
        repo_root=tmp_path,
        exp_id="iter16-profile",
        artifacts_dir=artifacts_dir,
        input_script=tmp_path / "input.txt",
        input_edit=None,
        allowlist=[],
    )

    stale_nested = Path(
        f"{tmp_path}/artifacts/sessions/s1/runs/iter15/worktrees/iter15-a"
        f"/artifacts/sessions/s1/runs/iter14/worktrees/iter14-b/third_party/bwa/bwt.c"
    )
    mapped = ctx.map_to_worktree(stale_nested)
    expected = (ctx.worktree_dir / "third_party" / "bwa" / "bwt.c").resolve()
    assert mapped == expected
