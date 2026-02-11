"""Agentic LLM client with multi-turn conversation and tool use support."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import openai

MAX_TURNS_SENTINEL = "[Max turns reached]"


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

        if not config.enabled:
            return

        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in env var {config.api_key_env}")

        client_cls = getattr(openai, "Client", None) or getattr(openai, "OpenAI", None)
        if client_cls is None:
            raise RuntimeError("OpenAI client class not available")

        self.client = client_cls(api_key=api_key, base_url=config.base_url)

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
            return ""

        if user_message:
            session.add_user_message(user_message)

        tools_for_api = session.get_api_tools() if session.tools else None

        while session.turn_count < self.config.max_turns:
            session.turn_count += 1

            try:
                kwargs: Dict[str, Any] = {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "messages": session.get_api_messages(),
                }
                if tools_for_api:
                    kwargs["tools"] = tools_for_api
                    kwargs["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**kwargs)
            except Exception as e:
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
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))

                # Add assistant message with tool calls
                session.add_assistant_message(
                    content=message.content,
                    tool_calls=tool_calls,
                )

                # Execute tools and add results
                for tc in tool_calls[:self.config.max_tool_calls_per_turn]:
                    tool = session.tools.get(tc.name)
                    if tool:
                        try:
                            result = tool.handler(**tc.arguments)
                        except Exception as e:
                            result = f"[Tool Error: {e}]"
                    else:
                        result = f"[Unknown tool: {tc.name}]"

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
