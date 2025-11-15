# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for working with AWS MCP servers more easily from the notebook"""

# Python Built-Ins:
from contextlib import contextmanager, ExitStack
from logging import getLogger
import os
from typing import cast, Generator, Sequence

# External Dependencies:
from mcp.client.stdio import DEFAULT_INHERITED_ENV_VARS
from strands.tools.mcp import MCPClient
from strands.types.tools import AgentTool

logger = getLogger(__name__)

INHERITED_ENV_VARS = DEFAULT_INHERITED_ENV_VARS + [
    "AWS_ACCESS_KEY_ID",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
]


def get_aws_mcp_env() -> dict[str, str]:
    """Get environment variables that local MCP servers should inherit, incl. AWS credentials

    Inspired by `mcp.client.stdio.get_default_environment()`, but expanded to include AWS CLI
    configurations and credentials.
    """
    env: dict[str, str] = {}

    for key in INHERITED_ENV_VARS:
        value = os.environ.get(key)
        if value is None:
            continue

        if value.startswith("()"):
            # Skip functions, which are a security risk
            continue

        env[key] = value

    return env


@contextmanager
def mcp_all_tools(
    mcp_clients: MCPClient | Sequence[MCPClient],
) -> Generator[list[AgentTool], None, None]:
    """Context manager to activate a set of MCP servers and list all their tools

    Parameters
    ----------
    mcp_clients :
        Either a Strands MCPClient or a list/tuple/Sequence of same.

    Usage
    -----

    ```python
    with mcp_all_tools(aws_docs_mcp, aws_pricing_mcp) as tools:
        agent = Agent(
            model=...,
            tools=tools
        )
        agent("Hello!")
    ```
    """

    clients_list = cast(
        list[MCPClient],
        [mcp_clients] if hasattr(mcp_clients, "list_tools_sync") else mcp_clients,
    )

    with ExitStack() as cstack:
        open_mcps = [cstack.enter_context(c) for c in clients_list]
        logger.info("Initialized %s MCP clients", len(open_mcps))
        tools = [t for c in open_mcps for t in c.list_tools_sync()]
        logger.info("Got %s total tools from MCP", len(tools))
        yield tools
