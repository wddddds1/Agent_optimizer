"""Restricted shell execution tool for platform exploration.

Provides a safe subset of shell commands for LLM agents to explore
the platform's hardware, compilers, and system configuration.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from orchestrator.agent_llm import ToolDefinition

# Commands allowed for exploration
ALLOWED_COMMANDS = {
    # Platform/hardware info
    "uname", "sysctl", "lscpu", "nproc", "free", "lsb_release",
    "numactl", "hwloc-ls", "hwloc-info",
    "nvidia-smi", "rocm-smi", "nvcc",
    # Compiler/toolchain (for version/capability queries)
    "gcc", "g++", "clang", "clang++", "icx", "icpx",
    "cmake", "make",
    "mpirun", "mpiexec", "ompi_info",
    # File viewing (read-only)
    "cat", "ls", "head", "tail", "wc", "file",
    # Process info
    "getconf",
}

# Patterns that are forbidden (shell metacharacters, injection attempts)
FORBIDDEN_PATTERNS = [
    r"\|",       # pipe
    r"[><]",     # redirect
    r"[;&]",     # command chaining
    r"`",        # backtick
    r"\$\(",     # command substitution
    r"\$\{",     # variable expansion with braces
    r"&&",       # logical and
    r"\|\|",     # logical or
]

# Allowed path prefixes for file reading commands
ALLOWED_READ_PREFIXES = [
    "/proc/",
    "/sys/",
    "/etc/os-release",
    "/etc/lsb-release",
    "/etc/redhat-release",
]

# Maximum output size (64KB)
MAX_OUTPUT_SIZE = 65536

# Default timeout (30 seconds)
DEFAULT_TIMEOUT = 30


class ShellToolError(Exception):
    """Error from shell tool execution."""
    pass


class ShellTool:
    """Restricted shell execution tool for platform exploration.

    Provides a safe subset of shell commands that LLM agents can use
    to explore the platform without risk of harmful side effects.
    """

    def __init__(self, cwd: Optional[Path] = None, timeout: int = DEFAULT_TIMEOUT):
        """Initialize the shell tool.

        Args:
            cwd: Working directory for command execution.
            timeout: Command timeout in seconds.
        """
        self.cwd = cwd
        self.timeout = timeout

    def get_tool_definition(self) -> ToolDefinition:
        """Return the tool definition for registration with an LLM agent."""
        return ToolDefinition(
            name="run_shell",
            description=(
                "Execute a shell command to explore the platform's hardware, "
                "compilers, and system configuration. Only read-only exploration "
                "commands are allowed. Use this to query CPU topology, memory, "
                "compiler versions, etc.\n\n"
                "Allowed commands: " + ", ".join(sorted(ALLOWED_COMMANDS)) + "\n\n"
                "Examples:\n"
                "- 'uname -a' - OS and kernel info\n"
                "- 'sysctl -n hw.ncpu' (macOS) or 'nproc' (Linux) - CPU count\n"
                "- 'sysctl hw.cachelinesize' (macOS) - cache line size\n"
                "- 'clang++ --version' - compiler version\n"
                "- 'cat /proc/cpuinfo' (Linux) - detailed CPU info"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            },
            handler=self.handle,
        )

    def handle(self, command: str) -> str:
        """Execute a command and return the output.

        Args:
            command: The shell command to execute.

        Returns:
            Command output (stdout + stderr).

        Raises:
            ShellToolError: If the command is not allowed.
        """
        # Validate command
        is_allowed, reason = self._validate_command(command)
        if not is_allowed:
            return f"Error: {reason}"

        # Parse and execute
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            return f"Error: Invalid command syntax: {e}"

        if not tokens:
            return "Error: Empty command"

        try:
            result = subprocess.run(
                tokens,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += result.stderr

            # Truncate if too long
            if len(output) > MAX_OUTPUT_SIZE:
                output = output[:MAX_OUTPUT_SIZE] + f"\n... (truncated, {len(output)} bytes total)"

            if result.returncode != 0 and not output:
                output = f"Command exited with code {result.returncode}"

            return output if output else "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.timeout} seconds"
        except FileNotFoundError:
            return f"Error: Command not found: {tokens[0]}"
        except PermissionError:
            return f"Error: Permission denied: {tokens[0]}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def _validate_command(self, command: str) -> Tuple[bool, str]:
        """Validate a command against security rules.

        Args:
            command: The command to validate.

        Returns:
            Tuple of (is_allowed, reason).
        """
        # Check for forbidden patterns
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, command):
                return False, f"Forbidden pattern in command: {pattern}"

        # Parse command
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False, "Invalid command syntax"

        if not tokens:
            return False, "Empty command"

        # Check base command
        base_cmd = Path(tokens[0]).name  # Handle /usr/bin/xxx paths
        if base_cmd not in ALLOWED_COMMANDS:
            return False, f"Command not in allowlist: {base_cmd}. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}"

        # Special validation for file reading commands
        if base_cmd in ("cat", "head", "tail"):
            return self._validate_file_read_command(tokens)

        return True, ""

    def _validate_file_read_command(self, tokens: List[str]) -> Tuple[bool, str]:
        """Validate file reading commands to ensure they only access allowed paths.

        Args:
            tokens: Parsed command tokens.

        Returns:
            Tuple of (is_allowed, reason).
        """
        # Extract file arguments (skip flags)
        file_args = [t for t in tokens[1:] if not t.startswith("-")]

        if not file_args:
            return False, "No file specified for read command"

        for path in file_args:
            # Check against allowed prefixes
            allowed = any(path.startswith(prefix) for prefix in ALLOWED_READ_PREFIXES)
            if not allowed:
                return False, (
                    f"File path not in allowed prefixes: {path}. "
                    f"Allowed: {', '.join(ALLOWED_READ_PREFIXES)}"
                )

        return True, ""


def create_shell_tool(cwd: Optional[Path] = None) -> ShellTool:
    """Create a shell tool instance.

    Args:
        cwd: Working directory for command execution.

    Returns:
        Configured ShellTool instance.
    """
    return ShellTool(cwd=cwd)
