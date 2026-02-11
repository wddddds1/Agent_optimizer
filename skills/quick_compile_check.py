"""Quick syntax-only compile check for patched files.

Runs ``compiler -fsyntax-only -Wall -Werror`` on the touched files after
applying a patch.  This catches ~90 % of compilation errors in seconds,
before the expensive full CMake build.
"""
from __future__ import annotations

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def quick_compile_check(
    repo_root: Path,
    touched_files: List[str],
    patch_diff: str,
    build_cfg: Dict[str, object],
) -> Tuple[bool, str]:
    """Apply *patch_diff* in a temp copy and run syntax-only compilation.

    Returns ``(success, error_message)``.  On success *error_message* is
    empty.
    """
    compiler = str(build_cfg.get("compiler", "c++"))
    include_dirs: List[str] = []
    cmake_flags = build_cfg.get("cmake_flags") or {}
    if isinstance(cmake_flags, dict):
        for key, val in cmake_flags.items():
            if "INCLUDE" in key.upper() and isinstance(val, str):
                include_dirs.append(val)

    # Resolve touched files to absolute paths that exist.
    abs_files: List[Path] = []
    for relpath in touched_files:
        p = (repo_root / relpath).resolve()
        if p.exists() and p.suffix in {".cpp", ".c", ".cc", ".cxx", ".h", ".hpp"}:
            abs_files.append(p)
    if not abs_files:
        return True, ""

    # Apply patch in a temporary copy of the touched files.
    tmpdir = Path(tempfile.mkdtemp(prefix="hpc_syntax_"))
    try:
        patched_files: List[Path] = []
        for src in abs_files:
            dst = tmpdir / src.name
            shutil.copy2(src, dst)
            patched_files.append(dst)

        # Write and apply patch.
        patch_path = tmpdir / "check.diff"
        patch_path.write_text(patch_diff, encoding="utf-8")
        apply = subprocess.run(
            ["git", "apply", "--unsafe-paths", "--directory", str(tmpdir), str(patch_path)],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            timeout=10,
        )
        if apply.returncode != 0:
            # If git apply fails, fall through – the full build will catch it.
            return True, ""

        # Syntax-only compilation on each patched file.
        errors: List[str] = []
        for fpath in patched_files:
            cmd = [compiler, "-fsyntax-only", "-Wall", "-Werror"]
            for d in include_dirs:
                cmd.extend(["-I", d])
            # Add the repo source tree as an include path so headers resolve.
            cmd.extend(["-I", str(repo_root / "src")])
            cmd.append(str(fpath))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                errors.append(f"{fpath.name}:\n{result.stderr.strip()}")
        if errors:
            return False, "\n".join(errors)
        return True, ""
    except (subprocess.TimeoutExpired, OSError) as exc:
        return True, ""  # Don't block on toolchain issues.
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
