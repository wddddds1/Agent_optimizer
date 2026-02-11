from pathlib import Path

from skills.patch_review import review_patch_diff


def test_patch_review_allows_whitelisted_file(tmp_path: Path) -> None:
    repo = tmp_path
    target = repo / "src" / "foo.c"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("int foo(){return 1;}\n", encoding="utf-8")
    patch = (
        "--- a/src/foo.c\n"
        "+++ b/src/foo.c\n"
        "@@ -1 +1 @@\n"
        "-int foo(){return 1;}\n"
        "+int foo(){return 2;}\n"
    )
    rules = {
        "allowed_globs": ["src/*.c"],
        "max_lines_changed": 4,
        "max_files_changed": 1,
        "forbidden_patterns": [],
    }
    ok, reasons, info = review_patch_diff(patch, repo, rules)
    assert ok
    assert not reasons
    assert info["files"] == ["src/foo.c"]


def test_patch_review_rejects_forbidden_pattern(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "src").mkdir()
    patch = (
        "--- a/src/foo.c\n"
        "+++ b/src/foo.c\n"
        "@@ -1 +1 @@\n"
        "-int foo(){return 1;}\n"
        "+int foo(){MPI_Abort(0,1);}\n"
    )
    rules = {
        "allowed_globs": ["src/*.c"],
        "max_lines_changed": 10,
        "max_files_changed": 1,
        "forbidden_patterns": ["MPI_Abort"],
    }
    ok, reasons, _ = review_patch_diff(patch, repo, rules)
    assert not ok
    assert reasons
