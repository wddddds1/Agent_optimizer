import pytest

from orchestrator.agents.code_patch import try_disambiguate_edits
from orchestrator.graph import _parse_edit_failure_anchor
from schemas.patch_edit_ir import PatchEdit
from skills.patch_edit import StructuredEditError, apply_structured_edits


def test_try_disambiguate_edits_handles_old_text_not_unique(tmp_path) -> None:
    source = tmp_path / "bwt.c"
    source.write_text(
        "void f1() {\n"
        "  int x = 0;\n"
        "  x += 1;\n"
        "  return;\n"
        "}\n"
        "\n"
        "void f2() {\n"
        "  int x = 0;\n"
        "  x += 1;\n"
        "  return;\n"
        "}\n",
        encoding="utf-8",
    )
    old_block = "  int x = 0;\n  x += 1;\n  return;"
    edit = PatchEdit(
        file="bwt.c",
        op="replace",
        anchor="  x += 1;",
        old_text=old_block,
        new_text="  int x = 2;\n  x += 3;\n  return;",
    )

    with pytest.raises(StructuredEditError) as exc:
        apply_structured_edits(tmp_path, [edit], ["bwt.c"])
    message = str(exc.value)
    assert message.startswith("old_text_not_unique:")

    snippets = [{"path": "bwt.c", "start_line": 7, "end_line": 11}]
    assert try_disambiguate_edits([edit], snippets, tmp_path, message)

    applied = apply_structured_edits(tmp_path, [edit], ["bwt.c"])
    assert applied.patch_diff
    assert "int x = 2;" in applied.patch_diff
    assert "x += 3;" in applied.patch_diff


def test_parse_edit_failure_anchor_accepts_old_text_not_unique() -> None:
    encoded = (
        "CQljbnRrWzBdICs9IHgmMHhmZjsgY250a1sxXSArPSB4Pj44JjB4ZmY7IGNudGtbMl0gKz0geD4+"
        "MTYmMHhmZjsgY250a1szXSArPSB4Pj4yNDsKCQljbnRsWzBdICs9IHkmMHhmZjsgY250bFsxXSAr"
        "PSB5Pj44JjB4ZmY7IGNudGxbMl0gKz0geT4+MTYmMHhmZjsgY250bFszXSArPSB5Pj4yNDsKCX0="
    )
    item = f"edit_apply_failed: old_text_not_unique:third_party/bwa/bwt.c:b64:{encoded}"
    parsed = _parse_edit_failure_anchor([item])
    assert parsed is not None
    assert parsed[0] == "third_party/bwa/bwt.c"
    assert "cntk[0] +=" in parsed[1]


def test_try_disambiguate_edits_handles_old_text_not_found_with_identifier_drift(tmp_path) -> None:
    source = tmp_path / "bwt.c"
    source.write_text(
        "void f1() {\n"
        "\t\t\tif (ok[c].x[2] * 2 > max_intv) break;\n"
        "\t\t\tif (ok[c].x[2] < min_intv) break;\n"
        "}\n",
        encoding="utf-8",
    )
    edit = PatchEdit(
        file="bwt.c",
        op="replace",
        anchor="\t\t\tif (curr.x[2] * 2 > max_intv) break;",
        old_text=(
            "\t\t\tif (curr.x[2] * 2 > max_intv) break;\n"
            "\t\t\tif (curr.x[2] < min_intv) break;"
        ),
        new_text=(
            "\t\t\tif (ok[c].x[2] * 2 > max_intv) break;\n"
            "\t\t\tif (ok[c].x[2] <= min_intv) break;"
        ),
    )

    with pytest.raises(StructuredEditError) as exc:
        apply_structured_edits(tmp_path, [edit], ["bwt.c"])
    message = str(exc.value)
    assert message.startswith("old_text_not_found:")

    snippets = [{"path": "bwt.c", "start_line": 1, "end_line": 4}]
    assert try_disambiguate_edits([edit], snippets, tmp_path, message)

    applied = apply_structured_edits(tmp_path, [edit], ["bwt.c"])
    assert applied.patch_diff
    assert "<= min_intv" in applied.patch_diff
