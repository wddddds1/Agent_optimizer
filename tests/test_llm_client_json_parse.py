from orchestrator.llm_client import _safe_json_loads


def test_safe_json_loads_plain_object() -> None:
    data = _safe_json_loads('{"status":"OK","value":1}')
    assert isinstance(data, dict)
    assert data["status"] == "OK"
    assert data["value"] == 1


def test_safe_json_loads_fenced_json() -> None:
    data = _safe_json_loads("```json\n{\"status\":\"OK\"}\n```")
    assert isinstance(data, dict)
    assert data["status"] == "OK"


def test_safe_json_loads_with_extra_text() -> None:
    content = "Some text before\n{\"status\":\"OK\",\"edits\":[]}\nSome text after"
    data = _safe_json_loads(content)
    assert isinstance(data, dict)
    assert data["status"] == "OK"


def test_safe_json_loads_control_chars_in_string() -> None:
    # Literal tab/newline inside string content can appear in provider output.
    content = '{"status":"OK","anchor":"line1\tline2\nline3"}'
    data = _safe_json_loads(content)
    assert isinstance(data, dict)
    assert data["status"] == "OK"
    assert "line1" in data["anchor"]
