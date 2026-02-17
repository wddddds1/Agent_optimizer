"""Agentic LLM client with multi-turn conversation and tool use support."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import openai

from orchestrator.errors import LLMUnavailableError

MAX_TURNS_SENTINEL = "[Max turns reached]"
TOOL_REPAIR_EXHAUSTED_SENTINEL = "[Tool call repair exhausted]"


def _chat_api_compat_hint(exc: Exception, model: str) -> str:
    text = str(exc)
    lower = text.lower()
    if _needs_default_temperature(exc):
        return (
            f"{text} | model '{model}' only supports default temperature=1 "
            "for chat.completions."
        )
    if _needs_max_completion_tokens(exc):
        return (
            f"{text} | model '{model}' expects max_completion_tokens for "
            "chat.completions. Retrying with max_completion_tokens may fix this."
        )
    if (
        "only supported in v1/responses" in lower
        or ("v1/responses" in lower and "chat/completions" in lower)
    ):
        return (
            f"{text} | model '{model}' is Responses-API-only. "
            "Use a chat.completions-compatible model (e.g. gpt-4.1-mini) "
            "or migrate AgentLLMClient to Responses API."
        )
    return text


def _needs_max_completion_tokens(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "unsupported parameter" in text
        and "max_tokens" in text
        and "max_completion_tokens" in text
    )


def _promote_max_completion_tokens(payload: Dict[str, Any]) -> None:
    if "max_tokens" in payload and "max_completion_tokens" not in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")


def _needs_default_temperature(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "temperature" in text
        and "unsupported" in text
        and "default (1)" in text
    )


def _promote_default_temperature(payload: Dict[str, Any]) -> bool:
    if "temperature" in payload and payload.get("temperature") != 1:
        payload["temperature"] = 1
        return True
    return False


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., str]


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: Optional[str] = None
    parse_error: Optional[str] = None


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages
    name: Optional[str] = None  # Tool name for tool responses

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        msg: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name and self.role == "tool":
            msg["name"] = self.name
        return msg


@dataclass
class AgentConfig:
    """Configuration for the agent LLM client."""
    enabled: bool = True
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 4096
    max_turns: int = 15  # Maximum conversation turns
    max_tool_calls_per_turn: int = 5  # Limit tool calls per turn
    max_invalid_tool_calls_total: int = 5
    max_invalid_tool_calls_per_tool: int = 2
    strict_availability: bool = True


@dataclass
class AgentSession:
    """A conversation session with the agent."""
    messages: List[Message] = field(default_factory=list)
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)
    total_tokens: int = 0
    turn_count: int = 0

    def add_system_message(self, content: str) -> None:
        """Add or update system message."""
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = content
        else:
            self.messages.insert(0, Message(role="system", content=content))

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> None:
        """Add an assistant message."""
        self.messages.append(Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls
        ))

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None:
        """Add a tool result message."""
        self.messages.append(Message(
            role="tool",
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name,
        ))

    def get_api_messages(self) -> List[Dict[str, Any]]:
        """Get messages in API format."""
        return [m.to_api_format() for m in self.messages]

    def get_api_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions in API format."""
        if not self.tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in self.tools.values()
        ]

    def estimate_tokens(self) -> int:
        """Rough estimate of current token usage."""
        total = 0
        for msg in self.messages:
            if msg.content:
                total += len(msg.content) // 3  # Rough estimate
        return total


class AgentLLMClient:
    """LLM client with agentic capabilities: multi-turn conversation and tool use."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = None
        self._prefer_max_completion_tokens = False
        self._prefer_default_temperature = False

        if not config.enabled:
            return

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise LLMUnavailableError(f"Missing API key in env var {config.api_key_env}")

        client_cls = getattr(openai, "Client", None) or getattr(openai, "OpenAI", None)
        if client_cls is None:
            raise LLMUnavailableError("OpenAI client class not available")

        self.client = client_cls(
            api_key=api_key,
            base_url=config.base_url,
            max_retries=0,
        )

    def _chat_create(self, **kwargs: Any) -> Any:
        if not self.client:
            raise RuntimeError("Agent LLM client not initialized")
        req = dict(kwargs)
        if self._prefer_max_completion_tokens:
            _promote_max_completion_tokens(req)
        if self._prefer_default_temperature:
            _promote_default_temperature(req)
        last_exc: Optional[Exception] = None
        for _ in range(3):
            try:
                return self.client.chat.completions.create(**req)
            except Exception as exc:
                last_exc = exc
                changed = False
                if _needs_max_completion_tokens(exc):
                    before = dict(req)
                    _promote_max_completion_tokens(req)
                    if req != before:
                        self._prefer_max_completion_tokens = True
                        changed = True
                if _needs_default_temperature(exc):
                    if _promote_default_temperature(req):
                        self._prefer_default_temperature = True
                        changed = True
                if not changed:
                    raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")

    def create_session(self, system_prompt: str = "") -> AgentSession:
        """Create a new conversation session."""
        session = AgentSession()
        if system_prompt:
            session.add_system_message(system_prompt)
        return session

    def register_tools(self, session: AgentSession, tools: List[ToolDefinition]) -> None:
        """Register tools for a session."""
        for tool in tools:
            session.tools[tool.name] = tool

    def _validate_tool_call(self, tool: ToolDefinition, tc: ToolCall) -> Optional[str]:
        """Validate tool call arguments against required schema fields."""
        if not isinstance(tc.arguments, dict):
            return (
                f"[Tool Error: Invalid arguments for {tool.name}. "
                f"Expected object, got {type(tc.arguments).__name__}]"
            )

        schema = tool.parameters if isinstance(tool.parameters, dict) else {}
        required = schema.get("required", [])
        if not isinstance(required, list):
            required = []

        missing = [key for key in required if key not in tc.arguments]
        if not missing and not tc.parse_error:
            return None

        pieces: List[str] = [
            f"[Tool Error: Invalid arguments for {tool.name}.",
        ]
        if missing:
            pieces.append(f"Missing required fields: {', '.join(missing)}.")
        pieces.append(f"Received parsed args: {json.dumps(tc.arguments, ensure_ascii=True)}.")
        if tc.raw_arguments is not None:
            pieces.append(f"Raw arguments: {tc.raw_arguments}")
        if tc.parse_error:
            pieces.append(f"Parse error: {tc.parse_error}")
        properties = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        if missing and properties:
            sample = {
                field: f"<{properties.get(field, {}).get('type', 'value')}>"
                for field in missing
            }
            pieces.append(
                "Please call this tool again with a JSON object containing required fields, "
                f"for example: {json.dumps(sample, ensure_ascii=True)}"
            )
        return " ".join(pieces)

    def _tool_repair_limit_reason(
        self,
        tool_name: str,
        invalid_total: int,
        invalid_streak: int,
    ) -> Optional[str]:
        """Return reason when invalid tool-call repair budget is exhausted."""
        if invalid_streak > self.config.max_invalid_tool_calls_per_tool:
            return (
                f"tool={tool_name} invalid_streak={invalid_streak} exceeds "
                f"limit={self.config.max_invalid_tool_calls_per_tool}"
            )
        if invalid_total > self.config.max_invalid_tool_calls_total:
            return (
                f"invalid_total={invalid_total} exceeds "
                f"limit={self.config.max_invalid_tool_calls_total}"
            )
        return None

    def chat(
        self,
        session: AgentSession,
        user_message: Optional[str] = None,
        auto_execute_tools: bool = True,
    ) -> str:
        """Send a message and get a response, optionally executing tools.

        Args:
            session: The conversation session
            user_message: The user message to send (None to continue after tool results)
            auto_execute_tools: If True, automatically execute tool calls and continue

        Returns:
            The final assistant response (after all tool calls are resolved)
        """
        if not self.client:
            if self.config.enabled and self.config.strict_availability:
                raise LLMUnavailableError("LLM client not initialized")
            return ""

        if user_message:
            session.add_user_message(user_message)

        tools_for_api = session.get_api_tools() if session.tools else None
        invalid_total = 0
        invalid_streak_by_tool: Dict[str, int] = {}

        while session.turn_count < self.config.max_turns:
            remaining_turns = self.config.max_turns - session.turn_count
            reserve_final_turn = bool(
                auto_execute_tools
                and tools_for_api
                and remaining_turns == 1
            )
            if reserve_final_turn:
                # Final turn must be a direct assistant answer, not another tool loop.
                session.add_user_message(
                    "FINAL TURN: do not call tools. Provide your best final answer now. "
                    "If prior instructions require JSON output, output valid JSON now."
                )
            session.turn_count += 1

            try:
                kwargs: Dict[str, Any] = {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "messages": session.get_api_messages(),
                }
                if tools_for_api and not reserve_final_turn:
                    kwargs["tools"] = tools_for_api
                    kwargs["tool_choice"] = "auto"
                response = self._chat_create(**kwargs)
            except Exception as e:
                if self.config.strict_availability:
                    hint = _chat_api_compat_hint(e, self.config.model)
                    raise LLMUnavailableError(f"Agent chat API error: {hint}") from e
                return f"[API Error: {e}]"

            choice = response.choices[0]
            message = choice.message

            # Update token count
            if hasattr(response, "usage") and response.usage:
                session.total_tokens += response.usage.total_tokens

            # Check for tool calls
            if message.tool_calls and auto_execute_tools:
                # Parse tool calls
                tool_calls = []
                for tc in message.tool_calls:
                    raw_args = tc.function.arguments if tc.function.arguments is not None else ""
                    parse_error = None
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                        if not isinstance(args, dict):
                            parse_error = (
                                "arguments must decode to a JSON object "
                                f"(got {type(args).__name__})"
                            )
                            args = {}
                    except json.JSONDecodeError as exc:
                        args = {}
                        parse_error = str(exc)
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                        raw_arguments=raw_args,
                        parse_error=parse_error,
                    ))

                # Add assistant message with tool calls
                session.add_assistant_message(
                    content=message.content,
                    tool_calls=tool_calls,
                )

                # Execute tools and add results
                for tc in tool_calls[:self.config.max_tool_calls_per_turn]:
                    tool = session.tools.get(tc.name)
                    is_invalid_call = False
                    if tool:
                        validation_error = self._validate_tool_call(tool, tc)
                        if validation_error:
                            is_invalid_call = True
                            result = validation_error
                        else:
                            try:
                                result = tool.handler(**tc.arguments)
                            except Exception as e:
                                result = f"[Tool Error: {e}]"
                    else:
                        result = f"[Unknown tool: {tc.name}]"

                    if is_invalid_call:
                        invalid_total += 1
                        invalid_streak = invalid_streak_by_tool.get(tc.name, 0) + 1
                        invalid_streak_by_tool[tc.name] = invalid_streak
                        limit_reason = self._tool_repair_limit_reason(
                            tc.name, invalid_total, invalid_streak
                        )
                        if limit_reason:
                            result += (
                                " [Tool Repair Exhausted: stop issuing invalid tool calls and "
                                "return final JSON with status=\"NEED_MORE_CONTEXT\".]"
                            )
                            session.add_tool_result(tc.id, tc.name, result)
                            return (
                                f"{TOOL_REPAIR_EXHAUSTED_SENTINEL}: {limit_reason}"
                            )
                    else:
                        invalid_streak_by_tool[tc.name] = 0

                    session.add_tool_result(tc.id, tc.name, result)

                # Continue the loop to get next response
                continue

            # No tool calls - we have a final response
            content = message.content or ""
            session.add_assistant_message(content=content)
            return content

        return MAX_TURNS_SENTINEL

    def chat_for_json(
        self,
        session: AgentSession,
        user_message: str,
        auto_execute_tools: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Chat and parse the response as JSON."""
        response = self.chat(session, user_message, auto_execute_tools)
        return self._parse_json(response)

    def _parse_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from response, handling markdown code blocks."""
        content = content.strip()

        # Remove markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in the content
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None


# Convenience function for quick single-turn requests
def quick_request(
    config: AgentConfig,
    system_prompt: str,
    user_message: str,
) -> str:
    """Make a quick single-turn request without tools."""
    client = AgentLLMClient(config)
    session = client.create_session(system_prompt)
    return client.chat(session, user_message, auto_execute_tools=False)
