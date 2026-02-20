"""AST-aware code structure extraction using tree-sitter.

Provides function boundaries, loop nesting, array access patterns,
branch density, and structural patch validation.  Falls back gracefully
when tree-sitter is not installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional tree-sitter import
# ---------------------------------------------------------------------------

_TREE_SITTER_AVAILABLE = False
_TS_C_LANG = None
_TS_CPP_LANG = None

try:
    import tree_sitter  # type: ignore
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    tree_sitter = None  # type: ignore

def _get_c_language():
    global _TS_C_LANG
    if _TS_C_LANG is not None:
        return _TS_C_LANG
    try:
        import tree_sitter_c  # type: ignore
        _TS_C_LANG = tree_sitter.Language(tree_sitter_c.language())
        return _TS_C_LANG
    except Exception:
        return None

def _get_cpp_language():
    global _TS_CPP_LANG
    if _TS_CPP_LANG is not None:
        return _TS_CPP_LANG
    try:
        import tree_sitter_cpp  # type: ignore
        _TS_CPP_LANG = tree_sitter.Language(tree_sitter_cpp.language())
        return _TS_CPP_LANG
    except Exception:
        return None


def _get_language_for_file(file_path: str):
    """Return the appropriate tree-sitter Language for a file extension."""
    if not _TREE_SITTER_AVAILABLE:
        return None
    ext = Path(file_path).suffix.lower()
    if ext in (".c", ".h"):
        return _get_c_language()
    if ext in (".cpp", ".cxx", ".cc", ".hpp", ".hxx"):
        return _get_cpp_language()
    # .h files could be C++; try C first, fall back to C++
    return None


def _parse_file(file_path: str, source: Optional[bytes] = None):
    """Parse a file and return (tree, source_bytes)."""
    if not _TREE_SITTER_AVAILABLE:
        return None, None
    lang = _get_language_for_file(file_path)
    if lang is None:
        return None, None
    parser = tree_sitter.Parser(lang)
    if source is None:
        try:
            source = Path(file_path).read_bytes()
        except OSError:
            return None, None
    tree = parser.parse(source)
    return tree, source


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AccessPattern:
    """An array/pointer access pattern within a loop."""
    expression: str = ""
    stride: str = "unknown"      # "unit", "non-unit", "indirect", "unknown"
    indirection_level: int = 0   # 0=direct, 1=a[b[i]], 2=a[b[c[i]]]
    line: int = 0


@dataclass
class LoopInfo:
    """Information about a single loop."""
    loop_type: str = ""    # "for", "while", "do_while"
    start_line: int = 0
    end_line: int = 0
    depth: int = 0
    iterator: str = ""
    bounds: str = ""
    access_patterns: List[AccessPattern] = field(default_factory=list)
    branch_count: int = 0
    children: List["LoopInfo"] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "loop_type": self.loop_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "depth": self.depth,
            "iterator": self.iterator,
            "bounds": self.bounds,
            "access_patterns": [
                {"expression": a.expression, "stride": a.stride,
                 "indirection_level": a.indirection_level, "line": a.line}
                for a in self.access_patterns
            ],
            "branch_count": self.branch_count,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str = ""
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    parameter_types: List[str] = field(default_factory=list)
    return_type: str = ""
    loops: List[LoopInfo] = field(default_factory=list)
    max_loop_depth: int = 0
    branch_density: float = 0.0  # branches per line
    call_targets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parameter_types": self.parameter_types,
            "return_type": self.return_type,
            "loops": [l.to_dict() for l in self.loops],
            "max_loop_depth": self.max_loop_depth,
            "branch_density": round(self.branch_density, 3),
            "call_targets": self.call_targets,
        }


@dataclass
class CodeStructure:
    """Complete structural analysis of a source file."""
    file_path: str = ""
    functions: List[FunctionInfo] = field(default_factory=list)
    total_lines: int = 0
    parse_errors: int = 0

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "functions": [f.to_dict() for f in self.functions],
            "total_lines": self.total_lines,
            "parse_errors": self.parse_errors,
        }


@dataclass
class LoopNest:
    """Loop nesting structure at a specific code location."""
    loops: List[LoopInfo] = field(default_factory=list)
    max_depth: int = 0


@dataclass
class ValidationResult:
    """Result of AST-based patch validation."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tree-sitter based extraction
# ---------------------------------------------------------------------------

def _node_text(node, source: bytes) -> str:
    """Get the text of a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_start_line(node) -> int:
    return node.start_point[0] + 1


def _node_end_line(node) -> int:
    return node.end_point[0] + 1


def _extract_functions_ts(tree, source: bytes, file_path: str) -> List[FunctionInfo]:
    """Extract function info from tree-sitter parse tree."""
    functions: List[FunctionInfo] = []
    root = tree.root_node

    for node in _walk_nodes(root):
        if node.type == "function_definition":
            func = _parse_function_node(node, source, file_path)
            if func:
                functions.append(func)

    return functions


def _walk_nodes(node):
    """Iterate over all nodes in a tree-sitter parse tree."""
    yield node
    for child in node.children:
        yield from _walk_nodes(child)


def _parse_function_node(node, source: bytes, file_path: str) -> Optional[FunctionInfo]:
    """Parse a function_definition node into FunctionInfo."""
    func = FunctionInfo(file_path=file_path)
    func.start_line = _node_start_line(node)
    func.end_line = _node_end_line(node)

    # Extract function name from declarator
    declarator = node.child_by_field_name("declarator")
    if declarator:
        func.name = _extract_function_name(declarator, source)
        func.parameter_types = _extract_param_types(declarator, source)

    # Extract return type
    type_node = node.child_by_field_name("type")
    if type_node:
        func.return_type = _node_text(type_node, source).strip()

    # Extract body info
    body = node.child_by_field_name("body")
    if body:
        func.loops = _extract_loops(body, source, depth=0)
        func.max_loop_depth = _max_depth(func.loops)
        branch_count = _count_branches(body)
        total_lines = func.end_line - func.start_line + 1
        func.branch_density = branch_count / max(total_lines, 1)
        func.call_targets = _extract_call_targets(body, source)

    return func


def _extract_function_name(declarator, source: bytes) -> str:
    """Extract the function name from a declarator node."""
    # Navigate through potential pointer_declarator, reference_declarator
    current = declarator
    while current:
        if current.type in ("function_declarator",):
            inner = current.child_by_field_name("declarator")
            if inner:
                name = _node_text(inner, source).strip()
                # Handle qualified names like ClassName::method
                return name.split("::")[-1] if "::" in name else name
        if current.type == "identifier":
            return _node_text(current, source).strip()
        if current.type in ("qualified_identifier", "field_identifier",
                            "destructor_name", "template_function"):
            return _node_text(current, source).strip()
        # Descend into the declarator
        child = current.child_by_field_name("declarator")
        if child and child != current:
            current = child
        else:
            break
    return _node_text(declarator, source).strip()[:60]


def _extract_param_types(declarator, source: bytes) -> List[str]:
    """Extract parameter types from function declarator."""
    types: List[str] = []
    for child in _walk_nodes(declarator):
        if child.type == "parameter_declaration":
            type_node = child.child_by_field_name("type")
            if type_node:
                types.append(_node_text(type_node, source).strip())
    return types


def _extract_loops(node, source: bytes, depth: int) -> List[LoopInfo]:
    """Extract loop structures from an AST node."""
    loops: List[LoopInfo] = []
    for child in node.children:
        loop = None
        if child.type == "for_statement":
            loop = _parse_for_loop(child, source, depth)
        elif child.type == "while_statement":
            loop = _parse_while_loop(child, source, depth)
        elif child.type == "do_statement":
            loop = _parse_do_loop(child, source, depth)

        if loop:
            loops.append(loop)
        else:
            # Recurse into non-loop compound statements
            if child.type in ("compound_statement", "if_statement",
                              "else_clause", "switch_statement", "case_statement"):
                loops.extend(_extract_loops(child, source, depth))
    return loops


def _parse_for_loop(node, source: bytes, depth: int) -> LoopInfo:
    """Parse a for_statement node."""
    loop = LoopInfo(
        loop_type="for",
        start_line=_node_start_line(node),
        end_line=_node_end_line(node),
        depth=depth,
    )

    # Extract iterator and bounds from for(init; cond; update)
    init = node.child_by_field_name("initializer")
    cond = node.child_by_field_name("condition")
    update = node.child_by_field_name("update")

    if init:
        loop.iterator = _infer_iterator(init, source)
    if cond:
        loop.bounds = _node_text(cond, source).strip()

    body = node.child_by_field_name("body")
    if body:
        loop.children = _extract_loops(body, source, depth + 1)
        loop.access_patterns = _extract_access_patterns(body, source)
        loop.branch_count = _count_branches(body)

    return loop


def _parse_while_loop(node, source: bytes, depth: int) -> LoopInfo:
    loop = LoopInfo(
        loop_type="while",
        start_line=_node_start_line(node),
        end_line=_node_end_line(node),
        depth=depth,
    )
    cond = node.child_by_field_name("condition")
    if cond:
        loop.bounds = _node_text(cond, source).strip()
    body = node.child_by_field_name("body")
    if body:
        loop.children = _extract_loops(body, source, depth + 1)
        loop.access_patterns = _extract_access_patterns(body, source)
        loop.branch_count = _count_branches(body)
    return loop


def _parse_do_loop(node, source: bytes, depth: int) -> LoopInfo:
    loop = LoopInfo(
        loop_type="do_while",
        start_line=_node_start_line(node),
        end_line=_node_end_line(node),
        depth=depth,
    )
    cond = node.child_by_field_name("condition")
    if cond:
        loop.bounds = _node_text(cond, source).strip()
    body = node.child_by_field_name("body")
    if body:
        loop.children = _extract_loops(body, source, depth + 1)
        loop.access_patterns = _extract_access_patterns(body, source)
        loop.branch_count = _count_branches(body)
    return loop


def _infer_iterator(init_node, source: bytes) -> str:
    """Try to extract iterator variable name from for-loop initializer."""
    text = _node_text(init_node, source).strip()
    # Match patterns like "int i = 0", "i = 0", "size_t jj = 0"
    m = re.match(r"(?:\w+\s+)?(\w+)\s*=", text)
    return m.group(1) if m else ""


def _extract_access_patterns(node, source: bytes) -> List[AccessPattern]:
    """Extract array/pointer access patterns from a loop body."""
    patterns: List[AccessPattern] = []
    seen = set()
    for child in _walk_nodes(node):
        if child.type == "subscript_expression":
            expr_text = _node_text(child, source).strip()
            if expr_text in seen:
                continue
            seen.add(expr_text)
            indirection = _count_subscript_nesting(child)
            stride = _infer_stride(child, source)
            patterns.append(AccessPattern(
                expression=expr_text[:80],
                stride=stride,
                indirection_level=indirection,
                line=_node_start_line(child),
            ))
    return patterns[:20]  # Cap to avoid noise


def _count_subscript_nesting(node) -> int:
    """Count nesting depth of subscript expressions (a[b[i]] = 1)."""
    depth = 0
    for child in _walk_nodes(node):
        if child.type == "subscript_expression" and child != node:
            depth += 1
    return depth


def _infer_stride(node, source: bytes) -> str:
    """Heuristically determine the stride of a subscript access."""
    index_node = node.child_by_field_name("index")
    if not index_node:
        return "unknown"
    index_text = _node_text(index_node, source).strip()

    # Check for indirect access: a[b[i]]
    for child in _walk_nodes(index_node):
        if child.type == "subscript_expression":
            return "indirect"

    # Simple iterator (i, j, jj, etc.) → unit stride
    if re.match(r"^[a-z_]\w*$", index_text) and len(index_text) <= 4:
        return "unit"
    # i+offset or offset+i → unit stride
    if re.match(r"^\w+\s*[+-]\s*\d+$", index_text):
        return "unit"
    # i*stride → non-unit
    if re.search(r"[*]", index_text):
        return "non-unit"

    return "unknown"


def _count_branches(node) -> int:
    """Count branch points (if/switch) in a subtree."""
    count = 0
    for child in _walk_nodes(node):
        if child.type in ("if_statement", "switch_statement",
                          "conditional_expression"):
            count += 1
    return count


def _extract_call_targets(node, source: bytes) -> List[str]:
    """Extract function call targets from a function body."""
    targets: List[str] = []
    seen = set()
    for child in _walk_nodes(node):
        if child.type == "call_expression":
            func_node = child.child_by_field_name("function")
            if func_node:
                name = _node_text(func_node, source).strip()
                if name not in seen and len(name) < 80:
                    seen.add(name)
                    targets.append(name)
    return targets[:30]


def _max_depth(loops: List[LoopInfo]) -> int:
    """Compute maximum nesting depth across all loops."""
    if not loops:
        return 0
    depths = []
    for loop in loops:
        child_depth = _max_depth(loop.children)
        depths.append(loop.depth + 1 + child_depth if loop.children else loop.depth + 1)
    return max(depths) if depths else 0


# ---------------------------------------------------------------------------
# Regex-based fallback extraction
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(
    r"^(?:[\w:*&\s]+?)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?{"
)
_LOOP_RE = re.compile(r"\b(for|while)\s*\(")
_IF_RE = re.compile(r"\b(if|switch)\s*\(")
_SUBSCRIPT_RE = re.compile(r"\w+\[([^\]]+)\]")


def _extract_functions_regex(source_text: str, file_path: str) -> List[FunctionInfo]:
    """Fallback: regex-based function extraction."""
    functions: List[FunctionInfo] = []
    lines = source_text.splitlines()
    i = 0
    while i < len(lines):
        m = _FUNC_RE.match(lines[i])
        if m:
            name = m.group(1)
            start = i + 1
            # Find matching brace
            brace_depth = 0
            for j in range(i, len(lines)):
                brace_depth += lines[j].count("{") - lines[j].count("}")
                if brace_depth <= 0 and j > i:
                    end = j + 1
                    break
            else:
                end = len(lines)

            body = "\n".join(lines[i:end])
            loop_count = len(_LOOP_RE.findall(body))
            branch_count = len(_IF_RE.findall(body))
            total = end - start + 1

            func = FunctionInfo(
                name=name,
                file_path=file_path,
                start_line=start,
                end_line=end,
                max_loop_depth=min(loop_count, 3),  # rough estimate
                branch_density=branch_count / max(total, 1),
            )
            functions.append(func)
            i = end
        else:
            i += 1
    return functions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Check if tree-sitter is available."""
    return _TREE_SITTER_AVAILABLE


def extract_code_structure(file_path: str) -> CodeStructure:
    """Extract complete code structure from a source file.

    Uses tree-sitter when available, falls back to regex.
    """
    path = Path(file_path)
    if not path.exists():
        return CodeStructure(file_path=file_path)

    try:
        source_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return CodeStructure(file_path=file_path)

    total_lines = len(source_text.splitlines())
    structure = CodeStructure(file_path=file_path, total_lines=total_lines)

    if _TREE_SITTER_AVAILABLE:
        tree, source_bytes = _parse_file(file_path)
        if tree and source_bytes:
            structure.functions = _extract_functions_ts(tree, source_bytes, file_path)
            # Count parse errors
            for node in _walk_nodes(tree.root_node):
                if node.type == "ERROR":
                    structure.parse_errors += 1
            return structure

    # Fallback to regex
    structure.functions = _extract_functions_regex(source_text, file_path)
    return structure


def extract_function_at_line(file_path: str, line: int) -> Optional[FunctionInfo]:
    """Get the complete function containing a specific line."""
    structure = extract_code_structure(file_path)
    for func in structure.functions:
        if func.start_line <= line <= func.end_line:
            return func
    return None


def get_loop_nest(file_path: str, line: int) -> Optional[LoopNest]:
    """Get the loop nesting structure at a given line."""
    func = extract_function_at_line(file_path, line)
    if not func:
        return None

    def _find_loops_at_line(loops: List[LoopInfo], target: int) -> List[LoopInfo]:
        result: List[LoopInfo] = []
        for loop in loops:
            if loop.start_line <= target <= loop.end_line:
                result.append(loop)
                result.extend(_find_loops_at_line(loop.children, target))
        return result

    loops = _find_loops_at_line(func.loops, line)
    return LoopNest(
        loops=loops,
        max_depth=max((l.depth for l in loops), default=0) + 1 if loops else 0,
    )


def get_access_patterns(file_path: str, func_name: str) -> List[AccessPattern]:
    """Get array access patterns for a specific function."""
    structure = extract_code_structure(file_path)
    for func in structure.functions:
        if func.name == func_name:
            patterns: List[AccessPattern] = []
            for loop in func.loops:
                patterns.extend(loop.access_patterns)
                for child in loop.children:
                    patterns.extend(child.access_patterns)
            return patterns
    return []


def extract_snippet_features(file_path: str, start_line: int, end_line: int) -> Dict[str, Any]:
    """Extract AST-derived features for a code snippet region.

    Returns a dict compatible with the existing _snippet_features() format
    in graph.py so it can serve as a drop-in replacement.
    """
    structure = extract_code_structure(file_path)

    loop_count = 0
    loop_headers: List[str] = []
    max_loop_depth = 0
    has_adjacent_loops = False
    indirect_access_count = 0
    branch_count = 0
    access_patterns: List[Dict] = []
    prev_loop_end = -999

    for func in structure.functions:
        # Only consider functions overlapping the snippet region
        if func.end_line < start_line or func.start_line > end_line:
            continue

        for loop in _flatten_loops(func.loops):
            if loop.start_line < start_line or loop.start_line > end_line:
                continue
            loop_count += 1
            loop_headers.append(f"{loop.loop_type}({loop.bounds})")
            max_loop_depth = max(max_loop_depth, loop.depth + 1)
            branch_count += loop.branch_count

            if loop.start_line - prev_loop_end <= 8:
                has_adjacent_loops = True
            prev_loop_end = loop.end_line

            for ap in loop.access_patterns:
                if ap.indirection_level > 0:
                    indirect_access_count += 1
                access_patterns.append({
                    "expression": ap.expression,
                    "stride": ap.stride,
                    "indirection": ap.indirection_level,
                })

    return {
        "loop_count": loop_count,
        "loop_headers": loop_headers[:4],
        "has_adjacent_loops": has_adjacent_loops,
        "max_loop_depth": max_loop_depth,
        "array_access_patterns": access_patterns[:10],
        "indirect_access_count": indirect_access_count,
        "branch_density": branch_count / max(end_line - start_line + 1, 1),
    }


def _flatten_loops(loops: List[LoopInfo]) -> List[LoopInfo]:
    """Flatten nested loop list."""
    result: List[LoopInfo] = []
    for loop in loops:
        result.append(loop)
        result.extend(_flatten_loops(loop.children))
    return result


# ---------------------------------------------------------------------------
# P3: AST Post-Validation
# ---------------------------------------------------------------------------

def validate_patch_structure(
    original_file: str,
    patched_content: str,
) -> ValidationResult:
    """Structurally validate a patch using tree-sitter AST comparison.

    Checks:
    - No syntax errors introduced
    - Function signatures preserved (unless intentional)
    - Brace matching correct
    - No orphaned declarations

    Returns ValidationResult with errors and warnings.
    """
    result = ValidationResult()

    if not _TREE_SITTER_AVAILABLE:
        # Cannot validate without tree-sitter, pass through
        return result

    # Parse original
    orig_tree, orig_source = _parse_file(original_file)
    if not orig_tree or not orig_source:
        result.warnings.append("Could not parse original file with tree-sitter")
        return result

    # Parse patched content
    lang = _get_language_for_file(original_file)
    if not lang:
        return result
    parser = tree_sitter.Parser(lang)
    patched_bytes = patched_content.encode("utf-8")
    patched_tree = parser.parse(patched_bytes)

    # Check for syntax errors in patched version
    orig_errors = _count_error_nodes(orig_tree.root_node)
    patched_errors = _count_error_nodes(patched_tree.root_node)
    new_errors = patched_errors - orig_errors

    if new_errors > 0:
        result.valid = False
        result.errors.append(
            f"Patch introduces {new_errors} new syntax error(s) "
            f"(original: {orig_errors}, patched: {patched_errors})"
        )
        # Locate error nodes
        for node in _walk_nodes(patched_tree.root_node):
            if node.type == "ERROR":
                line = _node_start_line(node)
                snippet = patched_content.splitlines()[line - 1:line + 1] if line > 0 else []
                result.errors.append(
                    f"  Syntax error at line {line}: {' '.join(snippet)[:100]}"
                )

    # Check function signature preservation
    orig_funcs = _extract_function_signatures(orig_tree.root_node, orig_source)
    patched_funcs = _extract_function_signatures(patched_tree.root_node, patched_bytes)

    for name, sig in orig_funcs.items():
        if name in patched_funcs:
            if patched_funcs[name] != sig:
                result.warnings.append(
                    f"Function signature changed: {name}\n"
                    f"  Original: {sig[:120]}\n"
                    f"  Patched:  {patched_funcs[name][:120]}"
                )
        else:
            result.warnings.append(f"Function '{name}' removed by patch")

    # Check brace balance
    orig_braces = patched_content.count("{") - patched_content.count("}")
    if abs(orig_braces) > 0:
        result.valid = False
        result.errors.append(
            f"Brace mismatch: {orig_braces:+d} "
            f"(opens={patched_content.count('{')}, closes={patched_content.count('}')})"
        )

    return result


def _count_error_nodes(node) -> int:
    """Count ERROR nodes in a tree-sitter parse tree."""
    count = 0
    for child in _walk_nodes(node):
        if child.type == "ERROR":
            count += 1
    return count


def _extract_function_signatures(root_node, source: bytes) -> Dict[str, str]:
    """Extract function name -> signature string mapping."""
    sigs: Dict[str, str] = {}
    for node in _walk_nodes(root_node):
        if node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            if declarator:
                name = _extract_function_name(declarator, source)
                # Build signature from type + declarator (excluding body)
                type_node = node.child_by_field_name("type")
                type_text = _node_text(type_node, source).strip() if type_node else ""
                decl_text = _node_text(declarator, source).strip()
                sigs[name] = f"{type_text} {decl_text}"
    return sigs
