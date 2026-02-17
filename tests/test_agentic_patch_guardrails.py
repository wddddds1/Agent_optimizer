from pathlib import Path
from types import SimpleNamespace

from orchestrator.agent_llm import (
    AgentConfig,
    AgentLLMClient,
    AgentSession,
    MAX_TURNS_SENTINEL,
    TOOL_REPAIR_EXHAUSTED_SENTINEL,
    ToolCall,
    ToolDefinition,
)
from orchestrator.agents.agentic_code_patch import AgenticCodePatchAgent
from orchestrator.agents.code_optimizer_agent import CodeOptimizerAgent
from skills.agent_tools import CodeOptimizationTools


def test_tool_call_validation_reports_missing_required_fields() -> None:
    client = AgentLLMClient(AgentConfig(enabled=False))
    tool = ToolDefinition(
        name="create_patch",
        description="create patch",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "changes": {"type": "array"},
            },
            "required": ["file_path", "changes"],
        },
        handler=lambda **_: "ok",
    )
    tc = ToolCall(
        id="1",
        name="create_patch",
        arguments={},
        raw_arguments="{}",
        parse_error=None,
    )
    msg = client._validate_tool_call(tool, tc)
    assert msg is not None
    assert "Missing required fields: file_path, changes" in msg
    assert "Raw arguments: {}" in msg


def test_tools_reset_session_state_clears_patch_flags(tmp_path: Path) -> None:
    src = tmp_path / "a.c"
    src.write_text("int x = 1;\n", encoding="utf-8")
    tools = CodeOptimizationTools(tmp_path, tmp_path)
    result = tools._handle_create_patch(
        file_path="a.c",
        changes=[
            {
                "operation": "replace",
                "anchor": "int x = 1;",
                "old_code": "int x = 1;",
                "new_code": "int x = 2;",
            }
        ],
    )
    assert "Patch created" in result
    assert tools._patch_created_this_session is True
    assert tools._current_patch

    tools.reset_session_state()
    assert tools._patch_created_this_session is False
    assert tools._current_patch is None


def test_max_turns_no_patch_fallback_even_if_patch_exists(tmp_path: Path) -> None:
    agent = CodeOptimizerAgent(AgentConfig(enabled=False), repo_root=tmp_path, build_dir=tmp_path)
    session = AgentSession()

    agent.tools._current_patch = "--- a/a.c\n+++ b/a.c\n"
    agent.tools._patch_created_this_session = True
    result = agent._parse_result(MAX_TURNS_SENTINEL, session)
    assert result.status == "NEED_MORE_CONTEXT"
    assert result.patch_diff == ""


def test_missing_final_json_no_patch_fallback(tmp_path: Path) -> None:
    agent = CodeOptimizerAgent(AgentConfig(enabled=False), repo_root=tmp_path, build_dir=tmp_path)
    session = AgentSession()
    agent.tools._current_patch = "--- a/a.c\n+++ b/a.c\n"
    agent.tools._patch_created_this_session = True
    result = agent._parse_result("final answer without json", session)
    assert result.status == "NEED_MORE_CONTEXT"
    assert result.patch_diff == ""


def test_tool_repair_exhausted_maps_to_need_more_context(tmp_path: Path) -> None:
    agent = CodeOptimizerAgent(AgentConfig(enabled=False), repo_root=tmp_path, build_dir=tmp_path)
    session = AgentSession()
    result = agent._parse_result(
        f"{TOOL_REPAIR_EXHAUSTED_SENTINEL}: tool=create_patch invalid_streak=3 exceeds limit=2",
        session,
    )
    assert result.status == "NEED_MORE_CONTEXT"
    assert "repair budget exhausted" in result.rationale


def test_status_ok_without_patch_is_rejected(tmp_path: Path) -> None:
    agent = CodeOptimizerAgent(AgentConfig(enabled=False), repo_root=tmp_path, build_dir=tmp_path)
    session = AgentSession()
    result = agent._parse_result(
        '{"status":"OK","patch_diff":"","rationale":"done","confidence":0.9}',
        session,
    )
    assert result.status == "NEED_MORE_CONTEXT"
    assert result.patch_diff == ""


def test_tool_repair_limit_reason_triggers() -> None:
    client = AgentLLMClient(
        AgentConfig(
            enabled=False,
            max_invalid_tool_calls_total=5,
            max_invalid_tool_calls_per_tool=2,
        )
    )
    assert (
        client._tool_repair_limit_reason("create_patch", invalid_total=2, invalid_streak=3)
        is not None
    )
    assert (
        client._tool_repair_limit_reason("create_patch", invalid_total=6, invalid_streak=1)
        is not None
    )


def test_agentic_target_file_normalization(tmp_path: Path) -> None:
    orchestration_root = tmp_path
    agent_root = orchestration_root / "third_party" / "bwa"
    agent_root.mkdir(parents=True, exist_ok=True)
    (agent_root / "bwa.c").write_text("int main(){return 0;}\n", encoding="utf-8")

    agent = AgenticCodePatchAgent(repo_root=agent_root, build_dir=agent_root, enabled=False)
    normalized = agent._normalize_target_file("third_party/bwa/bwa.c", orchestration_root)
    assert normalized == "bwa.c"


def test_compile_single_fallback_without_compile_commands(tmp_path: Path) -> None:
    src = tmp_path / "a.c"
    src.write_text("int add(int a, int b) { return a + b; }\n", encoding="utf-8")
    tools = CodeOptimizationTools(tmp_path, tmp_path)
    output = tools._handle_compile_single("a.c")
    assert "fallback mode: no compile_commands.json" in output
    assert "compiles successfully" in output or "compiled (fallback mode" in output


def test_chat_reserves_final_turn_for_direct_answer() -> None:
    config = AgentConfig(enabled=False, max_turns=2)
    client = AgentLLMClient(config)
    calls: list[dict] = []

    def _make_response(with_tool_call: bool):
        if with_tool_call:
            message = SimpleNamespace(
                content="need tool",
                tool_calls=[
                    SimpleNamespace(
                        id="tool-1",
                        function=SimpleNamespace(name="echo_tool", arguments='{"value":"ok"}'),
                    )
                ],
            )
        else:
            message = SimpleNamespace(content='{"status":"OK"}', tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(total_tokens=1),
        )

    def _create(**kwargs):
        calls.append(kwargs)
        return _make_response(with_tool_call=("tools" in kwargs))

    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    session = client.create_session("system")
    client.register_tools(
        session,
        [
            ToolDefinition(
                name="echo_tool",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                handler=lambda value: f"tool:{value}",
            )
        ],
    )

    response = client.chat(session, user_message="start", auto_execute_tools=True)
    assert response == '{"status":"OK"}'
    assert len(calls) == 2
    assert "tools" in calls[0]
    assert "tools" not in calls[1]
