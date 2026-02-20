"""Structured parser for compiler optimization remarks.

Parses Clang ``-Rpass``/``-Rpass-missed``/``-Rpass-analysis`` and
GCC ``-fopt-info-all`` output into categorised, per-function summaries
that the LLM can reason about effectively.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CompilerRemark:
    """A single parsed compiler optimization remark."""
    file: str = ""
    line: int = 0
    col: int = 0
    category: str = "other"          # vectorization, unrolling, inlining, licm, fusion, other
    pass_name: str = ""
    status: str = "analysis"         # applied, missed, analysis
    message: str = ""
    loop_id: str = ""                # file:line identifier for grouping


@dataclass
class FunctionRemarkSummary:
    """Aggregated compiler remark summary for a single function."""
    function: str = ""
    file: str = ""
    start_line: int = 0
    end_line: int = 0
    vectorized_loops: List[str] = field(default_factory=list)
    missed_vectorizations: List[str] = field(default_factory=list)
    reasons_missed: List[str] = field(default_factory=list)
    inlining_decisions: List[str] = field(default_factory=list)
    unrolling_decisions: List[str] = field(default_factory=list)
    licm_decisions: List[str] = field(default_factory=list)
    other_remarks: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "function": self.function,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "vectorized_loops": self.vectorized_loops,
            "missed_vectorizations": self.missed_vectorizations,
            "reasons_missed": self.reasons_missed,
            "inlining_decisions": self.inlining_decisions,
            "unrolling_decisions": self.unrolling_decisions,
            "licm_decisions": self.licm_decisions,
            "other_remarks": self.other_remarks,
            "optimization_suggestions": self.optimization_suggestions,
        }


@dataclass
class StructuredCompilerRemarks:
    """Complete structured output from compiler remark parsing."""
    remarks: List[CompilerRemark] = field(default_factory=list)
    function_summaries: List[FunctionRemarkSummary] = field(default_factory=list)
    category_counts: Dict[str, int] = field(default_factory=dict)
    compiler_gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "function_summaries": [s.to_dict() for s in self.function_summaries],
            "category_counts": self.category_counts,
            "compiler_gaps": self.compiler_gaps,
            "total_remarks": len(self.remarks),
        }


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Clang: file:line:col: remark: <message> [-Rpass=<pass>]
_CLANG_REMARK_RE = re.compile(
    r"^(?P<file>[^:\s]+):(?P<line>\d+):(?P<col>\d+):\s*remark:\s*"
    r"(?P<message>.+?)(?:\s*\[-R(?P<status>pass|pass-missed|pass-analysis)=(?P<pass_name>[^\]]+)\])?\s*$"
)

# GCC: file:line:col: note: <message>
_GCC_NOTE_RE = re.compile(
    r"^(?P<file>[^:\s]+):(?P<line>\d+):(?P<col>\d+):\s*note:\s*(?P<message>.+)$"
)

# GCC standalone opt-info lines (no "note:" prefix)
_GCC_OPT_RE = re.compile(
    r"^(?P<file>[^:\s]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<message>.+)$"
)

# Category detection patterns
_CATEGORY_PATTERNS = {
    "vectorization": re.compile(
        r"(?:vectoriz|not vectoriz|vec_|loop vectoriz|SLP)", re.IGNORECASE
    ),
    "unrolling": re.compile(r"(?:unroll|peel)", re.IGNORECASE),
    "inlining": re.compile(r"(?:inline|inlin)", re.IGNORECASE),
    "licm": re.compile(r"(?:hoist|licm|invariant|loop.?invariant)", re.IGNORECASE),
    "fusion": re.compile(r"(?:fus(?:e|ion|ing)|distribut)", re.IGNORECASE),
}

# Missed-vectorization reason patterns
_MISSED_REASON_PATTERNS = {
    "data_dependency": re.compile(
        r"(?:data depend|loop.*depend|carried depend|reduction)", re.IGNORECASE
    ),
    "non_unit_stride": re.compile(
        r"(?:non.?unit stride|non.?contiguous|gather|scatter)", re.IGNORECASE
    ),
    "aliasing": re.compile(r"(?:alias|may alias|pointer alias)", re.IGNORECASE),
    "call_in_loop": re.compile(r"(?:call.*loop|function call)", re.IGNORECASE),
    "too_complex": re.compile(
        r"(?:too complex|cannot prove|cannot determine|unsupported)", re.IGNORECASE
    ),
    "mixed_types": re.compile(r"(?:mixed type|type conversion)", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _categorize_remark(message: str, pass_name: str) -> str:
    """Determine the category of a compiler remark."""
    combined = f"{message} {pass_name}"
    for cat, pattern in _CATEGORY_PATTERNS.items():
        if pattern.search(combined):
            return cat
    return "other"


def _clang_status(raw: str) -> str:
    """Map Clang -Rpass variant to status."""
    if raw == "pass":
        return "applied"
    if raw == "pass-missed":
        return "missed"
    return "analysis"


def _identify_missed_reasons(message: str) -> List[str]:
    """Identify specific reasons a vectorization was missed."""
    reasons = []
    for reason_name, pattern in _MISSED_REASON_PATTERNS.items():
        if pattern.search(message):
            reasons.append(reason_name)
    return reasons


def parse_remarks(report_text: str) -> List[CompilerRemark]:
    """Parse raw compiler remark output into structured CompilerRemark list."""
    remarks: List[CompilerRemark] = []
    if not report_text:
        return remarks

    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip pure error/warning lines
        if re.match(r".*:\d+:\d+:\s*(?:error|warning):", stripped):
            continue

        remark: Optional[CompilerRemark] = None

        # Try Clang format first
        m = _CLANG_REMARK_RE.match(stripped)
        if m:
            status_raw = m.group("status") or "analysis"
            remark = CompilerRemark(
                file=m.group("file"),
                line=int(m.group("line")),
                col=int(m.group("col")),
                pass_name=m.group("pass_name") or "",
                status=_clang_status(status_raw),
                message=m.group("message").strip(),
            )
        else:
            # Try GCC note format
            m = _GCC_NOTE_RE.match(stripped)
            if m:
                msg = m.group("message").strip()
                # GCC notes are usually analysis unless they say "optimized" or "vectorized"
                status = "analysis"
                if re.search(r"\b(?:optimized|vectorized|unrolled|inlined)\b", msg, re.IGNORECASE):
                    status = "applied"
                elif re.search(r"\b(?:not vectorized|not inlined|missed)\b", msg, re.IGNORECASE):
                    status = "missed"
                remark = CompilerRemark(
                    file=m.group("file"),
                    line=int(m.group("line")),
                    col=int(m.group("col")),
                    status=status,
                    message=msg,
                )
            else:
                # Try GCC standalone opt-info
                m = _GCC_OPT_RE.match(stripped)
                if m:
                    msg = m.group("message").strip()
                    # Only include if it looks like an optimization remark
                    opt_kw = re.search(
                        r"(?:vectoriz|unroll|peel|inline|hoist|fus|distribut|interleav|prefetch)",
                        msg,
                        re.IGNORECASE,
                    )
                    if opt_kw:
                        status = "analysis"
                        if re.search(r"\b(?:vectorized|optimized|unrolled|inlined)\b", msg, re.IGNORECASE):
                            status = "applied"
                        elif re.search(r"\b(?:not vectorized|not inlined)\b", msg, re.IGNORECASE):
                            status = "missed"
                        remark = CompilerRemark(
                            file=m.group("file"),
                            line=int(m.group("line")),
                            col=int(m.group("col")),
                            status=status,
                            message=msg,
                        )

        if remark is not None:
            remark.category = _categorize_remark(remark.message, remark.pass_name)
            remark.loop_id = f"{remark.file}:{remark.line}"
            remarks.append(remark)

    return remarks


def _infer_function_from_line(
    file_path: str, line: int, function_ranges: Dict[str, tuple]
) -> str:
    """Try to map a remark line to a function name using known ranges."""
    for func, (start, end) in function_ranges.items():
        if start <= line <= end:
            return func
    return f"{file_path}:{line}"


def aggregate_by_function(
    remarks: List[CompilerRemark],
    function_ranges: Optional[Dict[str, Dict[str, tuple]]] = None,
) -> List[FunctionRemarkSummary]:
    """Group remarks into per-function summaries.

    ``function_ranges`` is an optional mapping of
    ``{file: {func_name: (start_line, end_line)}}``.
    When not provided, remarks are grouped by file only.
    """
    # Group by (file, function)
    groups: Dict[str, FunctionRemarkSummary] = {}
    for r in remarks:
        func = ""
        if function_ranges and r.file in function_ranges:
            func = _infer_function_from_line(r.file, r.line, function_ranges[r.file])
        key = f"{r.file}::{func}" if func else r.file
        if key not in groups:
            groups[key] = FunctionRemarkSummary(
                function=func, file=r.file, start_line=r.line, end_line=r.line
            )
        summary = groups[key]
        summary.start_line = min(summary.start_line, r.line)
        summary.end_line = max(summary.end_line, r.line)

        label = f"L{r.line}: {r.message[:120]}"
        if r.category == "vectorization":
            if r.status == "applied":
                summary.vectorized_loops.append(label)
            elif r.status == "missed":
                summary.missed_vectorizations.append(label)
                reasons = _identify_missed_reasons(r.message)
                for reason in reasons:
                    if reason not in summary.reasons_missed:
                        summary.reasons_missed.append(reason)
            else:
                summary.other_remarks.append(label)
        elif r.category == "inlining":
            summary.inlining_decisions.append(label)
        elif r.category == "unrolling":
            summary.unrolling_decisions.append(label)
        elif r.category == "licm":
            summary.licm_decisions.append(label)
        else:
            summary.other_remarks.append(label)

    # Generate suggestions based on patterns
    for summary in groups.values():
        if summary.missed_vectorizations:
            summary.optimization_suggestions.append(
                f"{len(summary.missed_vectorizations)} loops not vectorized"
            )
            if "data_dependency" in summary.reasons_missed:
                summary.optimization_suggestions.append(
                    "Consider restrict pointers or data layout changes to break dependencies"
                )
            if "non_unit_stride" in summary.reasons_missed:
                summary.optimization_suggestions.append(
                    "Non-unit stride access detected; consider AoS-to-SoA or gather optimization"
                )
            if "aliasing" in summary.reasons_missed:
                summary.optimization_suggestions.append(
                    "Pointer aliasing prevents vectorization; add __restrict__ qualifiers"
                )

    return sorted(groups.values(), key=lambda s: s.start_line)


def identify_compiler_gaps(remarks: List[CompilerRemark]) -> List[str]:
    """Identify high-value 'compiler gap' patterns: loops that were NOT
    vectorized with specific reasons that a human could fix."""
    gaps: List[str] = []
    seen = set()
    for r in remarks:
        if r.category == "vectorization" and r.status == "missed":
            reasons = _identify_missed_reasons(r.message)
            key = (r.file, r.line, tuple(reasons))
            if key not in seen:
                seen.add(key)
                reason_str = ", ".join(reasons) if reasons else "unspecified reason"
                gaps.append(
                    f"{r.file}:{r.line} - vectorization missed ({reason_str}): "
                    f"{r.message[:100]}"
                )
    return gaps


def parse_and_aggregate(
    report_text: str,
    function_ranges: Optional[Dict[str, Dict[str, tuple]]] = None,
) -> StructuredCompilerRemarks:
    """Full pipeline: parse, aggregate, identify gaps."""
    remarks = parse_remarks(report_text)
    summaries = aggregate_by_function(remarks, function_ranges)
    gaps = identify_compiler_gaps(remarks)

    # Count by category
    counts: Dict[str, int] = {}
    for r in remarks:
        counts[r.category] = counts.get(r.category, 0) + 1

    return StructuredCompilerRemarks(
        remarks=remarks,
        function_summaries=summaries,
        category_counts=counts,
        compiler_gaps=gaps,
    )
