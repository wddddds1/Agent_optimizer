from orchestrator.graph import (
    _expand_snippets_for_symbols,
    _extract_context_symbols_from_missing_fields,
)


def test_extract_context_symbols_from_missing_fields() -> None:
    missing = [
        "bwt.h 中 bwt_t 结构体的完整定义",
        "__occ_aux4 宏完整定义",
        "bwt_occ_intv函数实现",
    ]
    symbols = _extract_context_symbols_from_missing_fields(missing)
    assert "bwt_t" in symbols
    assert "__occ_aux4" in symbols
    assert "bwt_occ_intv" in symbols


def test_expand_snippets_for_symbols_collects_header_and_source(tmp_path) -> None:
    src = tmp_path / "bwt.c"
    hdr = tmp_path / "bwt.h"
    src.write_text(
        '#include "bwt.h"\n'
        "int bwt_occ_intv(const bwt_t *bwt, int k) {\n"
        "  return __occ_aux4(bwt->cnt, k);\n"
        "}\n",
        encoding="utf-8",
    )
    hdr.write_text(
        "typedef struct { int cnt[4]; } bwt_t;\n"
        "#define __occ_aux4(cnt, k) ((cnt)[(k)&3])\n",
        encoding="utf-8",
    )
    snippets = _expand_snippets_for_symbols(
        repo_root=tmp_path,
        symbols=["bwt_t", "__occ_aux4", "bwt_occ_intv"],
        search_files=["bwt.c"],
        max_snippets=6,
        max_chars=12000,
    )
    assert snippets
    paths = {item.get("path") for item in snippets}
    assert "bwt.c" in paths
    assert "bwt.h" in paths
