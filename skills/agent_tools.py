"""Tool implementations for the code optimization agent."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.agent_llm import ToolDefinition
from skills.shell_tool import ShellTool


# ---------------------------------------------------------------------------
# Module-level helpers for compiler optimization reports
# ---------------------------------------------------------------------------

def _compiler_opt_report_flags(compile_cmd: str) -> List[str]:
    """Return compiler flags that enable optimization remarks.

    Detects whether the compile command uses Clang or GCC (including
    platform-specific aliases like ``AppleClang``, ``cc``, ``c++``) and
    returns the appropriate flags.

    Returns an empty list if the compiler cannot be identified.
    """
    cmd_lower = compile_cmd.lower()
    # Clang / AppleClang (also matches "clang++", "/usr/bin/clang")
    if "clang" in cmd_lower:
        return [
            "-Rpass=.*",
            "-Rpass-missed=.*",
            "-Rpass-analysis=.*",
        ]
    # GCC / g++ (match "gcc", "g++", "/usr/bin/g++-13", etc.)
    if re.search(r'\bg(?:cc|\+\+)', cmd_lower):
        return [
            "-fopt-info-vec-all",
            "-fopt-info-loop-all",
            "-fopt-info-inline-all",
        ]
    # macOS default cc/c++ — typically AppleClang
    # Note: \b doesn't work after '+', so use lookahead for non-word or end
    if re.search(r'(?:^|/|\\)(?:cc(?=\s|$)|c\+\+)', cmd_lower):
        return [
            "-Rpass=.*",
            "-Rpass-missed=.*",
            "-Rpass-analysis=.*",
        ]
    return []


def _filter_opt_report_lines(report: str) -> List[str]:
    """Filter compiler stderr to only optimization-related remark lines.

    Handles both Clang ``-Rpass`` output (lines containing ``remark:``) and
    GCC ``-fopt-info`` output (lines containing ``note:`` or optimised/
    vectorized/unrolled keywords).
    """
    if not report:
        return []

    # Patterns that indicate an optimisation remark
    _REMARK_PATTERNS = (
        # Clang: "… remark: …"
        re.compile(r"remark:", re.IGNORECASE),
        # GCC -fopt-info: "… note: …" with optimisation keywords
        re.compile(
            r"note:.*(?:vectoriz|unroll|peel|inline|loop|hoist|"
            r"version|alias|prefetch|distribute|fuse|interleav)",
            re.IGNORECASE,
        ),
        # GCC -fopt-info standalone lines (no "note:" prefix)
        re.compile(
            r"(?:vectoriz|not vectoriz|loop unroll|loop peel|"
            r"loop distribut|loop fus|inlin|hoist)",
            re.IGNORECASE,
        ),
    )

    result: List[str] = []
    for line in report.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip pure error/warning diagnostics (keep only remarks)
        if re.match(r".*:\d+:\d+: (?:error|warning):", stripped):
            continue
        for pat in _REMARK_PATTERNS:
            if pat.search(stripped):
                result.append(stripped)
                break
    return result


class CodeOptimizationTools:
    """Tools for code exploration, understanding, and optimization."""

    def __init__(
        self,
        repo_root: Path,
        build_dir: Optional[Path] = None,
        experience_db: Optional[Any] = None,
    ) -> None:
        self.repo_root = repo_root
        self.build_dir = build_dir or repo_root / "build"
        self.experience_db = experience_db
        self._profile_data: Optional[Dict[str, Any]] = None
        self._current_patch: Optional[str] = None
        self._patch_created_this_session: bool = False
        self._last_build_errors: Optional[str] = None
        self._benchmark_input: Optional[str] = None  # Path to benchmark input script
        self._shell_tool = ShellTool(cwd=repo_root)

    def reset_session_state(self) -> None:
        """Reset per-action transient state before a new optimization attempt."""
        self._current_patch = None
        self._patch_created_this_session = False

    def set_profile_data(self, profile: Dict[str, Any]) -> None:
        """Set the current profile data for analysis."""
        self._profile_data = profile

    def set_last_build_errors(self, errors: str) -> None:
        """Set the last build errors for retrieval."""
        self._last_build_errors = errors

    def set_benchmark_input(self, input_path: str) -> None:
        """Set the benchmark input script path."""
        self._benchmark_input = input_path

    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return [
            # === Code Exploration ===
            self._tool_read_file(),
            self._tool_grep(),
            self._tool_find_files(),
            self._tool_get_file_outline(),

            # === Code Understanding ===
            self._tool_get_type_definition(),
            self._tool_get_type_layout(),
            self._tool_get_include_chain(),
            self._tool_get_function_signature(),
            self._tool_get_callers(),
            self._tool_get_macro_definition(),

            # === Build Context ===
            self._tool_get_compile_flags(),
            self._tool_get_build_target_files(),
            self._tool_resolve_backend(),

            # === Performance Analysis ===
            self._tool_get_profile(),
            self._tool_get_assembly(),

            # === Code Modification ===
            self._tool_create_patch(),
            self._tool_preview_patch(),
            self._tool_find_anchor_occurrences(),
            self._tool_apply_patch_dry_run(),

            # === Verification ===
            self._tool_compile(),
            self._tool_compile_single(),
            self._tool_get_last_build_errors(),
            self._tool_run_benchmark(),

            # === Platform Inspection ===
            self._shell_tool.get_tool_definition(),  # run_shell

            # === Compiler Analysis ===
            self._tool_get_compiler_opt_report(),

            # === Knowledge ===
            self._tool_get_reference_implementation(),
            self._tool_search_experience(),
        ]

    # =========================================================================
    # Code Exploration Tools
    # =========================================================================

    def _tool_read_file(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read the contents of a source code file. Use this to understand code structure and implementation details.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to repository root"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-indexed, optional)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (1-indexed, optional)"
                    }
                },
                "required": ["path"]
            },
            handler=self._handle_read_file,
        )

    def _handle_read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        file_path = self.repo_root / path
        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            if start_line is not None or end_line is not None:
                start = (start_line or 1) - 1
                end = end_line or len(lines)
                lines = lines[start:end]
                # Add line numbers
                numbered = [
                    f"{i + start + 1:4d} | {line}"
                    for i, line in enumerate(lines)
                ]
                return "\n".join(numbered)

            # For full file, add line numbers
            numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)

        except Exception as e:
            return f"Error reading file: {e}"

    def _tool_grep(self) -> ToolDefinition:
        return ToolDefinition(
            name="grep",
            description="Search for code patterns in the repository. Use regex patterns for flexible matching.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex supported)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: entire repo)"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "File name pattern, e.g., '*.cpp' or '*.h'"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines before and after match (default: 2)"
                    }
                },
                "required": ["pattern"]
            },
            handler=self._handle_grep,
        )

    def _handle_grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        file_pattern: Optional[str] = None,
        context_lines: int = 2,
    ) -> str:
        search_path = self.repo_root / path if path else self.repo_root

        cmd = ["grep", "-rn", f"-C{context_lines}", "-E", pattern]
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        cmd.append(str(search_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.repo_root,
            )
            output = result.stdout.strip()
            if not output:
                return f"No matches found for pattern: {pattern}"

            # Limit output size
            lines = output.split("\n")
            if len(lines) > 100:
                lines = lines[:100]
                lines.append(f"\n... (truncated, {len(lines)} more matches)")

            return "\n".join(lines)

        except subprocess.TimeoutExpired:
            return "Error: Search timed out"
        except Exception as e:
            return f"Error during search: {e}"

    def _tool_find_files(self) -> ToolDefinition:
        return ToolDefinition(
            name="find_files",
            description="Find files by name pattern in the repository.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File name pattern, e.g., 'pair_*.cpp' or '*_omp.cpp'"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: entire repo)"
                    }
                },
                "required": ["pattern"]
            },
            handler=self._handle_find_files,
        )

    def _handle_find_files(
        self,
        pattern: str,
        path: Optional[str] = None,
    ) -> str:
        search_path = self.repo_root / path if path else self.repo_root

        try:
            matches = list(search_path.rglob(pattern))
            if not matches:
                return f"No files found matching: {pattern}"

            # Convert to relative paths
            relative = [str(m.relative_to(self.repo_root)) for m in matches[:50]]
            result = "\n".join(relative)

            if len(matches) > 50:
                result += f"\n... ({len(matches) - 50} more files)"

            return result

        except Exception as e:
            return f"Error finding files: {e}"

    def _tool_get_file_outline(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_file_outline",
            description="Get the structure of a C++ file: classes, functions, and their line numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to repository root"
                    }
                },
                "required": ["path"]
            },
            handler=self._handle_get_file_outline,
        )

    def _handle_get_file_outline(self, path: str) -> str:
        file_path = self.repo_root / path
        if not file_path.exists():
            return f"Error: File not found: {path}"

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            outline = []

            # Simple pattern matching for C++ structures
            class_pattern = re.compile(r"^\s*(class|struct)\s+(\w+)")
            func_pattern = re.compile(
                r"^\s*(?:(?:virtual|static|inline|explicit|constexpr)\s+)*"
                r"(?:[\w:]+(?:<[^>]+>)?[\s*&]+)?"
                r"(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*(?:=\s*\w+)?\s*[{;]"
            )
            template_func = re.compile(r"^template\s*<")

            in_class = None
            for i, line in enumerate(lines, 1):
                # Check for class/struct
                class_match = class_pattern.match(line)
                if class_match:
                    in_class = class_match.group(2)
                    outline.append(f"{i:4d} | class {in_class}")
                    continue

                # Check for function
                func_match = func_pattern.match(line)
                if func_match and not template_func.match(line):
                    func_name = func_match.group(1)
                    if func_name not in ("if", "while", "for", "switch", "return"):
                        prefix = f"  {in_class}::" if in_class else ""
                        outline.append(f"{i:4d} |   {prefix}{func_name}()")

            if not outline:
                return "Could not extract outline (file may not be C++)"

            return "\n".join(outline)

        except Exception as e:
            return f"Error: {e}"

    # =========================================================================
    # Code Understanding Tools
    # =========================================================================

    def _tool_get_type_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_type_definition",
            description="Find the definition of a type, struct, or class. Essential for understanding data layouts.",
            parameters={
                "type": "object",
                "properties": {
                    "type_name": {
                        "type": "string",
                        "description": "Name of the type to find, e.g., 'dbl3_t', 'PairLJCut'"
                    }
                },
                "required": ["type_name"]
            },
            handler=self._handle_get_type_definition,
        )

    def _handle_get_type_definition(self, type_name: str) -> str:
        # Search for type definition
        patterns = [
            f"(struct|class|typedef|using)\\s+{type_name}\\b",
            f"#define\\s+{type_name}\\b",
        ]

        for pattern in patterns:
            result = self._handle_grep(pattern, file_pattern="*.h", context_lines=10)
            if "No matches found" not in result and "Error" not in result:
                return f"Definition of {type_name}:\n\n{result}"

        # Also check cpp files
        for pattern in patterns:
            result = self._handle_grep(pattern, file_pattern="*.cpp", context_lines=10)
            if "No matches found" not in result and "Error" not in result:
                return f"Definition of {type_name}:\n\n{result}"

        return f"Could not find definition of type: {type_name}"

    def _tool_get_type_layout(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_type_layout",
            description="Get the memory layout of a struct/class including field offsets and sizes. Critical for understanding data access patterns like dbl3_t.",
            parameters={
                "type": "object",
                "properties": {
                    "type_name": {
                        "type": "string",
                        "description": "Name of the type, e.g., 'dbl3_t', 'Atom'"
                    }
                },
                "required": ["type_name"]
            },
            handler=self._handle_get_type_layout,
        )

    def _handle_get_type_layout(self, type_name: str) -> str:
        """Get struct layout with field names and access patterns."""
        # First find the type definition
        type_def = self._handle_get_type_definition(type_name)
        if "Could not find" in type_def:
            return type_def

        # Extract struct/class body to show fields
        result = [f"Type layout for {type_name}:\n"]
        result.append(type_def)

        # Add usage hints for common types
        known_layouts = {
            "dbl3_t": """
ACCESS PATTERN for dbl3_t (OpenMP coordinate type):
  - This is a struct with fields: x, y, z (NOT an array!)
  - CORRECT:   x[j].x, x[j].y, x[j].z
  - WRONG:     x[j][0], x[j][1], x[j][2]
  - WRONG:     x[3*j], x[3*j+1], x[3*j+2]
""",
            "flt_t": """
ACCESS PATTERN for flt_t:
  - Alias for float in LAMMPS single-precision builds
""",
            "tagint": """
ACCESS PATTERN for tagint:
  - 32 or 64-bit integer depending on LAMMPS_BIGBIG compile flag
""",
        }

        if type_name in known_layouts:
            result.append(known_layouts[type_name])

        return "\n".join(result)

    def _tool_get_include_chain(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_include_chain",
            description="Find where a type or symbol is defined by tracing include chains.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name to trace"
                    },
                    "start_file": {
                        "type": "string",
                        "description": "File to start searching from (optional)"
                    }
                },
                "required": ["symbol"]
            },
            handler=self._handle_get_include_chain,
        )

    def _handle_get_include_chain(
        self,
        symbol: str,
        start_file: Optional[str] = None,
    ) -> str:
        """Trace include chain to find where a symbol is defined."""
        # First find where the symbol is defined
        definition = self._handle_get_type_definition(symbol)

        # Find all files that include this
        if "Error" in definition or "Could not find" in definition:
            return f"Could not find symbol: {symbol}"

        # Extract the file path from grep output
        lines = definition.split("\n")
        defining_file = None
        for line in lines:
            if ":" in line and ".h" in line:
                defining_file = line.split(":")[0].strip()
                break

        if not defining_file:
            return definition

        result = [f"Symbol '{symbol}' defined in: {defining_file}\n"]

        # Find what includes this file
        header_name = Path(defining_file).name
        includers = self._handle_grep(f'#include.*{header_name}', file_pattern="*.h")
        if "No matches" not in includers:
            result.append(f"\nIncluded by:\n{includers}")

        return "\n".join(result)

    def _tool_get_function_signature(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_function_signature",
            description="Get the signature of a function including parameters and return type.",
            parameters={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function"
                    },
                    "class_name": {
                        "type": "string",
                        "description": "Class name if it's a member function (optional)"
                    }
                },
                "required": ["function_name"]
            },
            handler=self._handle_get_function_signature,
        )

    def _handle_get_function_signature(
        self,
        function_name: str,
        class_name: Optional[str] = None,
    ) -> str:
        if class_name:
            pattern = f"{class_name}::{function_name}\\s*\\("
        else:
            pattern = f"\\b{function_name}\\s*\\([^)]*\\)"

        result = self._handle_grep(pattern, file_pattern="*.cpp", context_lines=3)
        if "No matches found" not in result:
            return result

        result = self._handle_grep(pattern, file_pattern="*.h", context_lines=3)
        return result

    def _tool_get_callers(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_callers",
            description="Find all places where a function is called.",
            parameters={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function to find callers for"
                    }
                },
                "required": ["function_name"]
            },
            handler=self._handle_get_callers,
        )

    def _handle_get_callers(self, function_name: str) -> str:
        # Search for function calls (not definitions)
        pattern = f"[^a-zA-Z_]{function_name}\\s*\\("
        return self._handle_grep(pattern, context_lines=1)

    def _tool_get_macro_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_macro_definition",
            description="Find the definition of a preprocessor macro.",
            parameters={
                "type": "object",
                "properties": {
                    "macro_name": {
                        "type": "string",
                        "description": "Name of the macro, e.g., 'NEIGHMASK', 'sbmask'"
                    }
                },
                "required": ["macro_name"]
            },
            handler=self._handle_get_macro_definition,
        )

    def _handle_get_macro_definition(self, macro_name: str) -> str:
        pattern = f"#define\\s+{macro_name}\\b"
        result = self._handle_grep(pattern, context_lines=2)

        if "No matches found" in result:
            # Try as inline function
            pattern = f"\\b{macro_name}\\s*\\("
            result = self._handle_grep(pattern, file_pattern="*.h", context_lines=5)

        return result

    # =========================================================================
    # Build Context Tools
    # =========================================================================

    def _tool_get_compile_flags(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_compile_flags",
            description="Get actual compile flags for a file from compile_commands.json or CMakeCache. Shows macros, optimization level, includes.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Source file to get flags for (optional - shows general flags if omitted)"
                    }
                }
            },
            handler=self._handle_get_compile_flags,
        )

    def _handle_get_compile_flags(self, file_path: Optional[str] = None) -> str:
        result = []

        # Try compile_commands.json first
        compile_cmds = self.build_dir / "compile_commands.json"
        if compile_cmds.exists():
            try:
                cmds = json.loads(compile_cmds.read_text())
                if file_path:
                    # Find specific file
                    for entry in cmds:
                        if file_path in entry.get("file", ""):
                            result.append(f"Compile command for {file_path}:")
                            result.append(entry.get("command", ""))
                            break
                else:
                    # Show first entry as example
                    if cmds:
                        result.append("Example compile command:")
                        result.append(cmds[0].get("command", ""))
            except Exception as e:
                result.append(f"Error reading compile_commands.json: {e}")

        # Also check CMakeCache
        cmake_cache = self.build_dir / "CMakeCache.txt"
        if cmake_cache.exists():
            try:
                cache = cmake_cache.read_text()
                # Extract key flags
                patterns = [
                    r"CMAKE_CXX_FLAGS:STRING=(.*)",
                    r"CMAKE_CXX_FLAGS_RELEASE:STRING=(.*)",
                    r"CMAKE_C_COMPILER:FILEPATH=(.*)",
                    r"CMAKE_CXX_COMPILER:FILEPATH=(.*)",
                    r"OpenMP_CXX_FLAGS:STRING=(.*)",
                    r"PKG_OPENMP:BOOL=(.*)",
                    r"PKG_OPT:BOOL=(.*)",
                    r"PKG_INTEL:BOOL=(.*)",
                ]
                result.append("\nCMake configuration:")
                for pattern in patterns:
                    match = re.search(pattern, cache)
                    if match:
                        result.append(f"  {pattern.split(':')[0]}: {match.group(1)}")
            except Exception as e:
                result.append(f"Error reading CMakeCache: {e}")

        if not result:
            return "No build configuration found. Build directory may not exist."

        return "\n".join(result)

    def _tool_get_build_target_files(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_build_target_files",
            description="Get the source files that are compiled into the current build target. Helps identify which backend files are used (OMP/OPT/INTEL).",
            parameters={
                "type": "object",
                "properties": {
                    "filter_pattern": {
                        "type": "string",
                        "description": "Filter files by pattern, e.g., 'pair_' to show only pair style files"
                    }
                }
            },
            handler=self._handle_get_build_target_files,
        )

    def _handle_get_build_target_files(self, filter_pattern: Optional[str] = None) -> str:
        result = []

        # Check compile_commands.json for list of compiled files
        compile_cmds = self.build_dir / "compile_commands.json"
        if compile_cmds.exists():
            try:
                cmds = json.loads(compile_cmds.read_text())
                files = [entry.get("file", "") for entry in cmds]

                # Filter if requested
                if filter_pattern:
                    files = [f for f in files if filter_pattern in f]

                # Group by directory
                by_dir: Dict[str, List[str]] = {}
                for f in files:
                    dir_name = str(Path(f).parent.name)
                    if dir_name not in by_dir:
                        by_dir[dir_name] = []
                    by_dir[dir_name].append(Path(f).name)

                result.append(f"Compiled source files ({len(files)} total):\n")
                for dir_name, dir_files in sorted(by_dir.items()):
                    if filter_pattern:
                        result.append(f"\n{dir_name}/ ({len(dir_files)} files):")
                        for f in sorted(dir_files)[:20]:
                            result.append(f"  {f}")
                        if len(dir_files) > 20:
                            result.append(f"  ... and {len(dir_files) - 20} more")
                    else:
                        result.append(f"  {dir_name}/: {len(dir_files)} files")

            except Exception as e:
                result.append(f"Error: {e}")
        else:
            result.append("compile_commands.json not found. Run cmake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON")

        return "\n".join(result)

    def _tool_resolve_backend(self) -> ToolDefinition:
        return ToolDefinition(
            name="resolve_backend",
            description="Determine which implementation file is actually used for a pair style given the build configuration.",
            parameters={
                "type": "object",
                "properties": {
                    "pair_style": {
                        "type": "string",
                        "description": "Pair style name, e.g., 'lj/cut', 'lj/cut/omp'"
                    }
                },
                "required": ["pair_style"]
            },
            handler=self._handle_resolve_backend,
        )

    def _handle_resolve_backend(self, pair_style: str) -> str:
        # Convert pair style name to file name pattern
        # lj/cut -> pair_lj_cut, lj/cut/omp -> pair_lj_cut_omp
        base_name = "pair_" + pair_style.replace("/", "_")

        result = [f"Resolving backend for pair style: {pair_style}\n"]

        # Check what packages are enabled
        cmake_cache = self.build_dir / "CMakeCache.txt"
        enabled_packages = []
        if cmake_cache.exists():
            cache = cmake_cache.read_text()
            for pkg in ["OPENMP", "OPT", "INTEL", "GPU", "KOKKOS"]:
                if re.search(f"PKG_{pkg}:BOOL=ON", cache):
                    enabled_packages.append(pkg)
            result.append(f"Enabled packages: {', '.join(enabled_packages) or 'none'}")

        # Find all matching source files
        candidates = []
        for pattern in [f"{base_name}.cpp", f"{base_name}_*.cpp"]:
            found = list(self.repo_root.rglob(f"src/**/{pattern}"))
            candidates.extend(found)

        result.append(f"\nCandidate files found:")
        for f in candidates:
            rel = f.relative_to(self.repo_root)
            # Determine which package it belongs to
            pkg = "base"
            for p in ["OPENMP", "OPT", "INTEL", "GPU", "KOKKOS"]:
                if p in str(rel):
                    pkg = p
                    break

            status = "✓ ACTIVE" if pkg == "base" or pkg in enabled_packages else "✗ disabled"
            result.append(f"  {rel} [{pkg}] {status}")

        # Check compile_commands.json to see what's actually compiled
        compile_cmds = self.build_dir / "compile_commands.json"
        if compile_cmds.exists():
            cmds = json.loads(compile_cmds.read_text())
            compiled = [e["file"] for e in cmds if base_name in e.get("file", "")]
            if compiled:
                result.append(f"\nActually compiled:")
                for f in compiled:
                    result.append(f"  {f}")

        return "\n".join(result)

    # =========================================================================
    # Performance Analysis Tools
    # =========================================================================

    def _tool_get_profile(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_profile",
            description="Get the current performance profile data including timing breakdown, IPC, and hotspots.",
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._handle_get_profile,
        )

    def _handle_get_profile(self) -> str:
        if not self._profile_data:
            return "No profile data available. Run a benchmark first."

        return json.dumps(self._profile_data, indent=2, ensure_ascii=False)

    def _tool_get_assembly(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_assembly",
            description="Get the compiled assembly code for a function. Useful for understanding compiler optimizations.",
            parameters={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function"
                    },
                    "object_file": {
                        "type": "string",
                        "description": "Path to the object file or executable (optional)"
                    }
                },
                "required": ["function_name"]
            },
            handler=self._handle_get_assembly,
        )

    def _handle_get_assembly(
        self,
        function_name: str,
        object_file: Optional[str] = None,
    ) -> str:
        # Find the executable
        if object_file:
            exe_path = self.repo_root / object_file
        else:
            # Try to find lmp executable
            candidates = list(self.build_dir.rglob("lmp"))
            if not candidates:
                return "Error: Could not find executable. Specify object_file."
            exe_path = candidates[0]

        if not exe_path.exists():
            return f"Error: File not found: {exe_path}"

        try:
            # Use objdump to get assembly
            result = subprocess.run(
                ["objdump", "-d", "-C", str(exe_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            output = result.stdout

            # Find the function
            pattern = f"<.*{function_name}.*>:"
            matches = list(re.finditer(pattern, output))

            if not matches:
                return f"Function {function_name} not found in assembly"

            # Extract function assembly
            start = matches[0].start()
            # Find next function or end
            next_func = re.search(r"\n\n[0-9a-f]+ <", output[start + 1:])
            if next_func:
                end = start + 1 + next_func.start()
            else:
                end = min(start + 5000, len(output))  # Limit size

            asm = output[start:end]

            # Limit lines
            lines = asm.split("\n")
            if len(lines) > 100:
                lines = lines[:100]
                lines.append("... (truncated)")

            return "\n".join(lines)

        except subprocess.TimeoutExpired:
            return "Error: Disassembly timed out"
        except Exception as e:
            return f"Error getting assembly: {e}"

    # =========================================================================
    # Code Modification Tools
    # =========================================================================

    def _tool_create_patch(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_patch",
            description="Create a code modification patch. The patch should be in unified diff format.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File to modify"
                    },
                    "changes": {
                        "type": "array",
                        "description": "List of changes to make",
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["replace", "insert_before", "insert_after", "delete"],
                                    "description": "Type of change"
                                },
                                "anchor": {
                                    "type": "string",
                                    "description": "Code to locate the change position (must be unique in file)"
                                },
                                "old_code": {
                                    "type": "string",
                                    "description": "Code to replace (for replace operation)"
                                },
                                "new_code": {
                                    "type": "string",
                                    "description": "New code to insert"
                                }
                            },
                            "required": ["operation", "anchor"]
                        }
                    }
                },
                "required": ["file_path", "changes"]
            },
            handler=self._handle_create_patch,
        )

    def _handle_create_patch(
        self,
        file_path: str,
        changes: List[Dict[str, Any]],
    ) -> str:
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        try:
            original = full_path.read_text(encoding="utf-8")
            modified = original

            for change in changes:
                op = change.get("operation")
                anchor = change.get("anchor", "")
                old_code = change.get("old_code", "")
                new_code = change.get("new_code", "")

                if anchor not in modified:
                    return f"Error: Anchor not found in file:\n{anchor[:100]}..."

                if modified.count(anchor) > 1:
                    return f"Error: Anchor is not unique (found {modified.count(anchor)} times)"

                if op == "replace":
                    if old_code and old_code in modified:
                        modified = modified.replace(old_code, new_code, 1)
                    else:
                        return f"Error: old_code not found for replace operation"

                elif op == "insert_before":
                    idx = modified.find(anchor)
                    modified = modified[:idx] + new_code + modified[idx:]

                elif op == "insert_after":
                    idx = modified.find(anchor) + len(anchor)
                    modified = modified[:idx] + new_code + modified[idx:]

                elif op == "delete":
                    if old_code:
                        modified = modified.replace(old_code, "", 1)

            # Generate unified diff
            import difflib
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
            diff_text = "".join(diff)

            if not diff_text:
                return "No changes made"

            self._current_patch = diff_text
            self._patch_created_this_session = True
            return f"Patch created:\n\n{diff_text}"

        except Exception as e:
            return f"Error creating patch: {e}"

    def _tool_preview_patch(self) -> ToolDefinition:
        return ToolDefinition(
            name="preview_patch",
            description="Preview the result of applying the current patch.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File to preview (optional, shows all changed files if omitted)"
                    }
                }
            },
            handler=self._handle_preview_patch,
        )

    def _handle_preview_patch(self, file_path: Optional[str] = None) -> str:
        if not self._current_patch:
            return "No patch created yet. Use create_patch first."

        return f"Current patch:\n\n{self._current_patch}"

    def _tool_find_anchor_occurrences(self) -> ToolDefinition:
        return ToolDefinition(
            name="find_anchor_occurrences",
            description="Find all occurrences of an anchor string in a file. Use this BEFORE create_patch to ensure anchor is unique.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File to search in"
                    },
                    "anchor": {
                        "type": "string",
                        "description": "The anchor string to search for"
                    }
                },
                "required": ["file_path", "anchor"]
            },
            handler=self._handle_find_anchor_occurrences,
        )

    def _handle_find_anchor_occurrences(self, file_path: str, anchor: str) -> str:
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        try:
            content = full_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            occurrences = []
            for i, line in enumerate(lines, 1):
                if anchor in line:
                    # Show context
                    start = max(0, i - 2)
                    end = min(len(lines), i + 2)
                    context = []
                    for j in range(start, end):
                        marker = ">>>" if j + 1 == i else "   "
                        context.append(f"{marker} {j + 1:4d} | {lines[j]}")
                    occurrences.append((i, "\n".join(context)))

            count = len(occurrences)
            if count == 0:
                return f"Anchor NOT FOUND in {file_path}:\n  '{anchor[:80]}...'" if len(anchor) > 80 else f"Anchor NOT FOUND in {file_path}:\n  '{anchor}'"

            result = [f"Found {count} occurrence(s) of anchor in {file_path}:"]
            if count == 1:
                result.append("✓ Anchor is UNIQUE - safe to use for patching\n")
            else:
                result.append("⚠ Anchor is NOT UNIQUE - patch may fail or apply to wrong location\n")

            for i, (line_num, context) in enumerate(occurrences[:5], 1):
                result.append(f"Occurrence {i} at line {line_num}:")
                result.append(context)
                result.append("")

            if count > 5:
                result.append(f"... and {count - 5} more occurrences")

            return "\n".join(result)

        except Exception as e:
            return f"Error: {e}"

    def _tool_apply_patch_dry_run(self) -> ToolDefinition:
        return ToolDefinition(
            name="apply_patch_dry_run",
            description="Test if the current patch can be applied cleanly without actually modifying files.",
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._handle_apply_patch_dry_run,
        )

    def _handle_apply_patch_dry_run(self) -> str:
        if not self._current_patch:
            return "No patch created yet. Use create_patch first."

        # Write patch to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(self._current_patch)
            patch_file = f.name

        try:
            # Try to apply with --dry-run
            result = subprocess.run(
                ["patch", "-p1", "--dry-run", "-i", patch_file],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
                timeout=30,
            )

            if result.returncode == 0:
                return f"✓ Patch can be applied cleanly:\n{result.stdout}"
            else:
                return f"✗ Patch cannot be applied:\n{result.stdout}\n{result.stderr}"

        except subprocess.TimeoutExpired:
            return "Error: Dry run timed out"
        except Exception as e:
            return f"Error during dry run: {e}"
        finally:
            Path(patch_file).unlink(missing_ok=True)

    # =========================================================================
    # Verification Tools
    # =========================================================================

    def _tool_compile(self) -> ToolDefinition:
        return ToolDefinition(
            name="compile",
            description="Compile the code to check for errors. Optionally apply a patch first.",
            parameters={
                "type": "object",
                "properties": {
                    "apply_patch": {
                        "type": "boolean",
                        "description": "Whether to apply the current patch before compiling"
                    }
                }
            },
            handler=self._handle_compile,
        )

    def _handle_compile(self, apply_patch: bool = False) -> str:
        # This is a simplified version - actual implementation would integrate
        # with the build system
        try:
            build_cmd = ["cmake", "--build", str(self.build_dir), "--", "-j4"]

            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=self.repo_root,
            )

            if result.returncode == 0:
                return "Compilation successful!"

            # Extract error messages
            output = result.stderr or result.stdout
            lines = output.split("\n")

            # Find error lines
            errors = [l for l in lines if "error:" in l.lower()]
            if errors:
                return "Compilation failed:\n\n" + "\n".join(errors[:20])

            return f"Compilation failed:\n\n{output[:2000]}"

        except subprocess.TimeoutExpired:
            return "Error: Compilation timed out"
        except Exception as e:
            return f"Error during compilation: {e}"

    def _tool_compile_single(self) -> ToolDefinition:
        return ToolDefinition(
            name="compile_single",
            description=(
                "Compile a single source file. By default checks for errors only "
                "(fast, syntax-only). Set with_opt_report=true to get the compiler's "
                "optimization report (vectorization, inlining, loop unrolling decisions)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Source file to compile (relative to repo root)",
                    },
                    "with_opt_report": {
                        "type": "boolean",
                        "description": (
                            "If true, compile with full codegen and output the "
                            "compiler's optimization report (vectorization, loop opts, "
                            "inlining). Slower but shows what the compiler already "
                            "optimizes. Default: false."
                        ),
                    },
                },
                "required": ["file_path"],
            },
            handler=self._handle_compile_single,
        )

    def _handle_compile_single(
        self, file_path: str, with_opt_report: bool = False
    ) -> str:
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        # Get compile command from compile_commands.json
        compile_cmds = self.build_dir / "compile_commands.json"
        if not compile_cmds.exists():
            return self._compile_single_without_compile_commands(
                file_path=file_path,
                full_path=full_path,
                with_opt_report=with_opt_report,
            )

        try:
            cmds = json.loads(compile_cmds.read_text())

            # Find the compile command for this file
            cmd_entry = None
            for entry in cmds:
                if file_path in entry.get("file", "") or full_path.name in entry.get("file", ""):
                    cmd_entry = entry
                    break

            if not cmd_entry:
                # Try to construct a command from similar files
                for entry in cmds:
                    if entry.get("file", "").endswith(".cpp"):
                        # Use this as template, replace the file
                        cmd_entry = entry.copy()
                        cmd_entry["file"] = str(full_path)
                        break

            if not cmd_entry:
                return "Error: Could not find compile command for this file"

            # Modify command to only compile (not link) and output to /dev/null
            cmd = cmd_entry.get("command", "")
            # Replace output file with /dev/null
            cmd = re.sub(r'-o\s+\S+', '-o /dev/null', cmd)

            if with_opt_report:
                # Full codegen with optimization reports (no -fsyntax-only)
                opt_flags = _compiler_opt_report_flags(cmd)
                if opt_flags:
                    cmd += " " + " ".join(opt_flags)
            else:
                # Fast syntax-only check
                cmd = cmd.replace(' -c ', ' -c -fsyntax-only ')

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120 if with_opt_report else 60,
                cwd=cmd_entry.get("directory", str(self.repo_root)),
            )

            if with_opt_report:
                # Optimization reports go to stderr for both GCC and Clang
                report = result.stderr or ""
                output = result.stdout or ""
                if result.returncode != 0:
                    self._last_build_errors = report
                    return f"✗ Compilation errors in {file_path}:\n\n{report[:5000]}"
                # Filter to optimization-related lines
                opt_lines = _filter_opt_report_lines(report)
                if opt_lines:
                    return (
                        f"✓ {file_path} compiled. Compiler optimization report:\n\n"
                        + "\n".join(opt_lines[:200])
                    )
                return (
                    f"✓ {file_path} compiled. No optimization remarks found "
                    "(compiler may not support -Rpass or -fopt-info)."
                )

            if result.returncode == 0:
                return f"✓ {file_path} compiles successfully"
            else:
                errors = result.stderr or result.stdout
                self._last_build_errors = errors
                return f"✗ Compilation errors in {file_path}:\n\n{errors[:3000]}"

        except subprocess.TimeoutExpired:
            return "Error: Compilation timed out"
        except Exception as e:
            return f"Error: {e}"

    def _compile_single_without_compile_commands(
        self,
        file_path: str,
        full_path: Path,
        with_opt_report: bool,
    ) -> str:
        """Compile a single file when compile_commands.json is unavailable.

        This supports make-based projects where no compilation database exists.
        """
        compiler = os.environ.get("CC", "cc")
        cmd: List[str] = [compiler, "-c", str(full_path), "-o", "/dev/null"]
        if not with_opt_report:
            cmd.append("-fsyntax-only")

        include_dirs = [full_path.parent, self.repo_root, self.repo_root / "include"]
        seen: set[str] = set()
        for inc in include_dirs:
            inc_str = str(inc)
            if not Path(inc_str).exists() or inc_str in seen:
                continue
            cmd.extend(["-I", inc_str])
            seen.add(inc_str)

        if with_opt_report:
            opt_flags = _compiler_opt_report_flags(" ".join(cmd))
            if opt_flags:
                cmd.extend(opt_flags)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120 if with_opt_report else 60,
                cwd=str(self.repo_root),
            )
        except subprocess.TimeoutExpired:
            return "Error: Compilation timed out"
        except Exception as e:
            return f"Error: {e}"

        stderr_text = result.stderr or ""
        stdout_text = result.stdout or ""
        if result.returncode != 0:
            self._last_build_errors = stderr_text or stdout_text
            return (
                f"✗ Compilation errors in {file_path} "
                "(fallback mode: no compile_commands.json):\n\n"
                f"{(stderr_text or stdout_text)[:5000]}"
            )

        if with_opt_report:
            opt_lines = _filter_opt_report_lines(stderr_text)
            if opt_lines:
                return (
                    f"✓ {file_path} compiled (fallback mode: no compile_commands.json). "
                    "Compiler optimization report:\n\n"
                    + "\n".join(opt_lines[:200])
                )
            return (
                f"✓ {file_path} compiled (fallback mode: no compile_commands.json). "
                "No optimization remarks found."
            )

        warnings = stderr_text.strip()
        if warnings:
            return (
                f"✓ {file_path} compiled (fallback mode: no compile_commands.json). "
                "Compiler warnings:\n\n"
                + warnings[:3000]
            )
        return f"✓ {file_path} compiles successfully (fallback mode: no compile_commands.json)"

    # -----------------------------------------------------------------
    # Compiler optimization report
    # -----------------------------------------------------------------

    def _tool_get_compiler_opt_report(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_compiler_opt_report",
            description=(
                "Compile a source file and return the compiler's optimization "
                "report showing which loops were vectorized, unrolled, or inlined, "
                "and which loops the compiler FAILED to optimize (and why). "
                "Use this BEFORE proposing source patches to avoid duplicating "
                "what the compiler already does at -O3."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Source file to analyze (relative to repo root)",
                    },
                    "function_filter": {
                        "type": "string",
                        "description": (
                            "Optional: only show remarks mentioning this function name. "
                            "Leave empty for all remarks."
                        ),
                    },
                },
                "required": ["file_path"],
            },
            handler=self._handle_compiler_opt_report,
        )

    def _fallback_compiler_opt_report(
        self, full_path: Path, function_filter: str = ""
    ) -> Optional[str]:
        """Try to generate optimization report without compile_commands.json.

        Uses a basic clang command with typical LAMMPS flags. Returns None if
        this approach fails too.
        """
        # Try a basic clang compile with common LAMMPS-like flags
        try:
            # Detect if we have clang available
            result = subprocess.run(
                ["clang++", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None

            # Construct a minimal compile command
            include_dirs = [
                str(self.repo_root / "src"),
                str(self.repo_root / "src" / "OPENMP"),
                str(self.repo_root / "src" / "KSPACE"),
            ]
            include_flags = " ".join(f"-I{d}" for d in include_dirs if Path(d).exists())

            cmd = (
                f"clang++ -std=c++11 -O3 -march=native -fopenmp "
                f"{include_flags} "
                f"-DLAMMPS_OMP -DLAMMPS_MEMALIGN=64 "
                f"-Rpass=.* -Rpass-missed=.* -Rpass-analysis=.* "
                f"-fsyntax-only {full_path}"
            )

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.repo_root),
            )

            # Even if compile fails, we might get useful remarks
            report = result.stderr or ""
            opt_lines = _filter_opt_report_lines(report)

            if function_filter:
                opt_lines = [
                    line for line in opt_lines
                    if function_filter.lower() in line.lower()
                ]

            if opt_lines:
                vectorized = [l for l in opt_lines if "vectoriz" in l.lower()]
                missed = [l for l in opt_lines if "missed" in l.lower() or "not vectorized" in l.lower()]

                parts = [f"Compiler optimization report (fallback mode) for {full_path.name}:"]
                if vectorized:
                    parts.append(f"\n## Successfully vectorized ({len(vectorized)} loops):")
                    parts.extend(vectorized[:20])
                if missed:
                    parts.append(f"\n## MISSED optimizations ({len(missed)} items):")
                    parts.extend(missed[:20])
                if len(opt_lines) > len(vectorized) + len(missed):
                    parts.append(f"\n## Other remarks:")
                    other = [l for l in opt_lines if l not in vectorized + missed]
                    parts.extend(other[:15])
                return "\n".join(parts)

            return None  # No useful remarks, fall through to guidance message

        except Exception:
            return None

    def _handle_compiler_opt_report(
        self, file_path: str, function_filter: str = ""
    ) -> str:
        full_path = self.repo_root / file_path
        if not full_path.exists():
            return f"Error: File not found: {file_path}"

        compile_cmds = self.build_dir / "compile_commands.json"
        if not compile_cmds.exists():
            # Fallback: try to find compile_commands.json in artifacts or construct basic command
            fallback_result = self._fallback_compiler_opt_report(full_path, function_filter)
            if fallback_result:
                return fallback_result
            return (
                "Note: compile_commands.json not found. Cannot generate detailed "
                "optimization reports. Proceed with patch design based on code "
                "structure and patterns. The compiler at -O3 -march=native typically:\n"
                "- Vectorizes simple loops with no data dependencies\n"
                "- Unrolls small loops (trip count < 8)\n"
                "- Inlines small functions\n"
                "- Hoists invariants out of loops\n\n"
                "Focus on optimizations the compiler CANNOT do:\n"
                "- Data layout changes (AoS → SoA)\n"
                "- Algorithm-level improvements\n"
                "- Aliasing disambiguation (__restrict__)\n"
                "- Branch elimination"
            )

        try:
            cmds = json.loads(compile_cmds.read_text())
            cmd_entry = None
            for entry in cmds:
                if file_path in entry.get("file", "") or full_path.name in entry.get("file", ""):
                    cmd_entry = entry
                    break
            if not cmd_entry:
                for entry in cmds:
                    if entry.get("file", "").endswith(".cpp"):
                        cmd_entry = entry.copy()
                        cmd_entry["file"] = str(full_path)
                        break
            if not cmd_entry:
                return "Error: Could not find compile command for this file"

            cmd = cmd_entry.get("command", "")
            cmd = re.sub(r'-o\s+\S+', '-o /dev/null', cmd)

            # Remove -fsyntax-only if present (need full codegen for opt reports)
            cmd = cmd.replace('-fsyntax-only', '')

            opt_flags = _compiler_opt_report_flags(cmd)
            if opt_flags:
                cmd += " " + " ".join(opt_flags)
            else:
                return (
                    "Error: Could not detect compiler type (Clang or GCC) "
                    "from the compile command."
                )

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=cmd_entry.get("directory", str(self.repo_root)),
            )

            if result.returncode != 0:
                errors = result.stderr or result.stdout
                return f"Compilation failed:\n{errors[:3000]}"

            report = result.stderr or ""
            opt_lines = _filter_opt_report_lines(report)

            if function_filter:
                opt_lines = [
                    line for line in opt_lines
                    if function_filter.lower() in line.lower()
                ]

            if not opt_lines:
                return (
                    f"No optimization remarks found for {file_path}"
                    + (f" (filter: {function_filter})" if function_filter else "")
                    + ". The compiler may have fully optimized all loops, or "
                    "optimization remarks are not supported."
                )

            # Categorize
            vectorized = [l for l in opt_lines if "vectoriz" in l.lower()]
            missed = [l for l in opt_lines if "missed" in l.lower() or "not vectorized" in l.lower() or "failed" in l.lower()]
            inlined = [l for l in opt_lines if "inline" in l.lower()]
            unrolled = [l for l in opt_lines if "unroll" in l.lower()]
            other = [l for l in opt_lines if l not in vectorized + missed + inlined + unrolled]

            parts = [f"Compiler optimization report for {file_path}:"]
            if vectorized:
                parts.append(f"\n## Successfully vectorized ({len(vectorized)} loops):")
                parts.extend(vectorized[:30])
            if missed:
                parts.append(f"\n## MISSED optimizations ({len(missed)} items):")
                parts.extend(missed[:30])
            if inlined:
                parts.append(f"\n## Inlining decisions ({len(inlined)} items):")
                parts.extend(inlined[:20])
            if unrolled:
                parts.append(f"\n## Loop unrolling ({len(unrolled)} items):")
                parts.extend(unrolled[:20])
            if other:
                parts.append(f"\n## Other remarks ({len(other)} items):")
                parts.extend(other[:20])

            return "\n".join(parts)

        except subprocess.TimeoutExpired:
            return "Error: Compilation timed out"
        except Exception as e:
            return f"Error: {e}"

    def _tool_get_last_build_errors(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_last_build_errors",
            description="Get the detailed error messages from the last failed compilation.",
            parameters={
                "type": "object",
                "properties": {}
            },
            handler=self._handle_get_last_build_errors,
        )

    def _handle_get_last_build_errors(self) -> str:
        if not self._last_build_errors:
            return "No build errors recorded. Run compile or compile_single first."

        # Parse and structure the errors
        errors = self._last_build_errors
        lines = errors.split("\n")

        structured = []
        current_error = []

        for line in lines:
            if re.match(r".*:\d+:\d+: (error|warning):", line):
                if current_error:
                    structured.append("\n".join(current_error))
                current_error = [line]
            elif current_error:
                current_error.append(line)

        if current_error:
            structured.append("\n".join(current_error))

        if structured:
            result = [f"Found {len(structured)} error(s)/warning(s):\n"]
            for i, err in enumerate(structured[:10], 1):
                result.append(f"--- Error {i} ---")
                result.append(err)
                result.append("")
            if len(structured) > 10:
                result.append(f"... and {len(structured) - 10} more")
            return "\n".join(result)

        return f"Raw build output:\n{errors[:5000]}"

    def _tool_run_benchmark(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_benchmark",
            description="Run a quick performance benchmark (50 timesteps) to validate optimization. Returns timing metrics.",
            parameters={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "description": "Number of timesteps to run (default: 50, max: 200)"
                    },
                    "threads": {
                        "type": "integer",
                        "description": "Number of OpenMP threads (default: 4)"
                    }
                }
            },
            handler=self._handle_run_benchmark,
        )

    def _handle_run_benchmark(self, steps: int = 50, threads: int = 4) -> str:
        # Limit steps for quick validation
        steps = min(steps, 200)

        # Find lmp executable
        exe_candidates = list(self.build_dir.rglob("lmp"))
        if not exe_candidates:
            return "Error: lmp executable not found in build directory"
        lmp_exe = exe_candidates[0]

        # Find input script
        input_script = self._benchmark_input
        if not input_script:
            # Try to find a default input in examples
            example_inputs = list(self.repo_root.glob("examples/**/in.*"))
            if example_inputs:
                input_script = str(example_inputs[0])
            else:
                return "Error: No benchmark input script configured. Set _benchmark_input."

        if not Path(input_script).exists():
            return f"Error: Input script not found: {input_script}"

        try:
            import tempfile
            import time

            # Create a modified input that runs fewer steps
            with tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False) as f:
                original = Path(input_script).read_text()
                # Replace run command with shorter run
                modified = re.sub(r'run\s+\d+', f'run {steps}', original)
                f.write(modified)
                temp_input = f.name

            env = {
                **subprocess.os.environ,
                "OMP_NUM_THREADS": str(threads),
            }

            # Run with timing
            start_time = time.time()
            result = subprocess.run(
                [str(lmp_exe), "-in", temp_input],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=Path(input_script).parent,
                env=env,
            )
            elapsed = time.time() - start_time

            # Clean up
            Path(temp_input).unlink(missing_ok=True)

            if result.returncode != 0:
                return f"✗ Benchmark failed:\n{result.stderr[:1000]}"

            # Parse LAMMPS output for timing
            output = result.stdout
            timing_match = re.search(r"Loop time of ([\d.]+)", output)
            loop_time = float(timing_match.group(1)) if timing_match else elapsed

            # Extract performance metrics
            perf_match = re.search(r"([\d.]+) timesteps/s", output)
            timesteps_per_sec = float(perf_match.group(1)) if perf_match else steps / loop_time

            # Extract timing breakdown
            breakdown = {}
            for line in output.split("\n"):
                if "|" in line and "%" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        name = parts[0].strip()
                        try:
                            pct = float(parts[2].strip().replace("%", ""))
                            breakdown[name] = pct
                        except ValueError:
                            pass

            return f"""✓ Benchmark completed ({steps} steps, {threads} threads):

Wall time:     {elapsed:.3f} s
Loop time:     {loop_time:.3f} s
Performance:   {timesteps_per_sec:.2f} timesteps/s

Timing breakdown:
{json.dumps(breakdown, indent=2) if breakdown else '(not available)'}
"""

        except subprocess.TimeoutExpired:
            return "Error: Benchmark timed out (>120s)"
        except Exception as e:
            return f"Error running benchmark: {e}"

    # =========================================================================
    # Knowledge Tools
    # =========================================================================

    def _tool_get_reference_implementation(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_reference_implementation",
            description="Find optimized reference implementations of similar functionality. For example, find the OPT version of a pair style.",
            parameters={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function or class to find references for"
                    }
                },
                "required": ["function_name"]
            },
            handler=self._handle_get_reference_implementation,
        )

    def _handle_get_reference_implementation(self, function_name: str) -> str:
        # Look for OPT versions
        base_name = function_name.replace("_omp", "").replace("OMP", "")

        # Search in OPT directory
        opt_files = self._handle_find_files(f"*{base_name}*", "src/OPT")
        if "No files found" not in opt_files:
            result = f"Found OPT implementations:\n{opt_files}\n\n"
            # Read first file
            first_file = opt_files.split("\n")[0]
            if first_file:
                content = self._handle_read_file(first_file)
                result += f"Content of {first_file}:\n{content}"
            return result

        # Try INTEL versions
        intel_files = self._handle_find_files(f"*{base_name}*", "src/INTEL")
        if "No files found" not in intel_files:
            return f"Found INTEL implementations:\n{intel_files}"

        return f"No optimized reference found for: {function_name}"

    def _tool_search_experience(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_experience",
            description=(
                "Search historical optimization experiences. Returns past results "
                "including outcome, improvement, diagnosis, and compiler_gap when available. "
                "Use this to avoid repeating failed approaches and to build on successes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "patch_family": {
                        "type": "string",
                        "description": "Filter by patch family, e.g., 'loop_fusion', 'param_table_pack'"
                    },
                    "target_file": {
                        "type": "string",
                        "description": "Filter by target file path (substring match)"
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["PASS", "FAIL"],
                        "description": "Filter by outcome"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by optimization category, e.g., 'data_layout', 'algorithmic'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10)"
                    }
                }
            },
            handler=self._handle_search_experience,
        )

    def _handle_search_experience(
        self,
        patch_family: Optional[str] = None,
        target_file: Optional[str] = None,
        outcome: Optional[str] = None,
        category: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> str:
        if not self.experience_db:
            return "No experience database available."

        records = self.experience_db.records
        if not records:
            return "No historical records found."

        max_results = min(max_results or 10, 20)
        matches = []
        for rec in records:
            if patch_family and rec.patch_family != patch_family:
                continue
            if target_file and (not rec.target_file or target_file not in rec.target_file):
                continue
            if outcome and rec.outcome != outcome:
                continue
            if category and getattr(rec, "category", None) != category:
                continue
            matches.append(rec)

        if not matches:
            return "No matching records found."

        # Sort: successes first, then by improvement descending
        matches.sort(key=lambda r: (-r.improvement_pct if r.outcome == "PASS" else 1000))
        matches = matches[:max_results]

        lines = [f"Found {len(matches)} matching record(s):\n"]
        for i, rec in enumerate(matches, 1):
            entry = [f"### Record {i}"]
            entry.append(f"- **action**: {rec.action_id}")
            entry.append(f"- **family**: {rec.family}, **patch_family**: {rec.patch_family or 'N/A'}")
            entry.append(f"- **outcome**: {rec.outcome}, **improvement**: {rec.improvement_pct:+.2f}%")
            entry.append(f"- **target_file**: {rec.target_file or 'N/A'}")
            entry.append(f"- **strength**: {rec.strength}, **app**: {rec.app}, **case**: {rec.case_id}")
            # Include deep analysis context when available
            if getattr(rec, "origin", None):
                entry.append(f"- **origin**: {rec.origin}")
            if getattr(rec, "category", None):
                entry.append(f"- **category**: {rec.category}")
            if getattr(rec, "diagnosis", None):
                entry.append(f"- **diagnosis**: {rec.diagnosis}")
            if getattr(rec, "mechanism", None):
                entry.append(f"- **mechanism**: {rec.mechanism}")
            if getattr(rec, "compiler_gap", None):
                entry.append(f"- **compiler_gap**: {rec.compiler_gap}")
            if getattr(rec, "target_functions", None):
                entry.append(f"- **functions**: {', '.join(rec.target_functions)}")
            lines.append("\n".join(entry))
        return "\n\n".join(lines)
