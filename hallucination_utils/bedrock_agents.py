# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Utilities for working with Amazon Bedrock Agents"""

# Python Built-Ins:
import json
from logging import getLogger
import time

# External Dependencies:
import boto3

logger = getLogger(__name__)


def invoke_bedrock_agent(
    agentId: str,
    sessionId: str,
    agentAliasId: str = "TSTALIASID",
    bedrock_agent_runtime_client=None,
    **kwargs,
) -> str:
    """Convenience method to simplify invoking Amazon Bedrock Agents

    Since the Bedrock InvokeAgent API only supports streaming responses, this function wraps around
    it to provide a simpler interface.

    Parameters
    ----------
    agentId :
        Unique ID of the Amazon Bedrock Agent to be invoked
    sessionId :
        Unique ID of the user session with the agent
    agentAliasId :
        Will be defaulted to the under-construction 'TSTALIASID' by default
    bedrock_agent_runtime_client :
        An optional boto3 'bedrock-agent-runtime' Client. If not provided, one will be created with
        default settings.
    **kwargs :
        Other function parameters are in line with boto3 invoke_agent. See:
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/invoke_agent.html
    """
    if not bedrock_agent_runtime_client:
        bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime")

    params = {
        "agentId": agentId,
        "agentAliasId": agentAliasId,
        "sessionId": sessionId,
        **kwargs,
    }

    # invoke the agent API
    agentResponse = bedrock_agent_runtime_client.invoke_agent(**params)

    # logger.info(pprint.pprint(agentResponse))
    event_stream = agentResponse["completion"]
    final_answer = ""

    for event in event_stream:
        if "chunk" in event:
            data = event["chunk"]["bytes"]
            final_answer += data.decode("utf8")
        elif "trace" in event:
            logger.info(json.dumps(event["trace"], indent=2))
        else:
            raise RuntimeError(f"unexpected event: {event}")

    return final_answer


def wait_for_agent_prepare(
    agent_id: str,
    bedrock_agent_client=None,
    poll_secs: float = 30,
) -> None:
    """Wait for an Amazon Bedrock Agent to become ready (finish 'preparing' or etc.)

    Parameters
    ----------
    agent_id :
        Unique ID (*not* name) of the Agent
    bedrock_agent_client :
        An optional boto3 'bedrock-agent' Client object. If not provided, one will be created with
        default settings.
    poll_secs :
        Number of seconds to wait between 'get_agent' calls to check agent status.
    """
    if not bedrock_agent_client:
        bedrock_agent_client = boto3.client("bedrock-agent")

    logger.info(f"Waiting for agent {agent_id} to be prepared...")
    agent_status = bedrock_agent_client.get_agent(agentId=agent_id)["agent"][
        "agentStatus"
    ]
    while agent_status in ("CREATING", "PREPARING", "UPDATING", "VERSIONING"):
        time.sleep(poll_secs)
        agent_status = bedrock_agent_client.get_agent(agentId=agent_id)["agent"][
            "agentStatus"
        ]
    if agent_status != "PREPARED":
        raise ValueError(f"Agent {agent_id} entered unexpected status: {agent_status}")
    logger.info("Done!")


def wait_for_agent_alias(
    agent_id: str,
    alias_id: str,
    bedrock_agent_client=None,
    poll_secs: float = 30,
) -> None:
    """Wait for an Amazon Bedrock Agent Alias to become ready (finish creating, etc.)

    Parameters
    ----------
    agent_id :
        Unique ID (*not* name) of the Agent
    alias_id :
        Unique ID (*not* name) of the Agent's Alias
    bedrock_agent_client :
        An optional boto3 'bedrock-agent' Client object. If not provided, one will be created with
        default settings.
    poll_secs :
        Number of seconds to wait between 'get_agent_alias' calls to check status.
    """
    if not bedrock_agent_client:
        bedrock_agent_client = boto3.client("bedrock-agent")

    logger.info(f"Waiting for alias {alias_id} of agent {agent_id} to be ready...")
    status = bedrock_agent_client.get_agent_alias(
        agentId=agent_id,
        agentAliasId=alias_id,
    )["agentAlias"]["agentAliasStatus"]
    while status in ("CREATING", "PREPARING", "UPDATING", "VERSIONING"):
        time.sleep(poll_secs)
        status = bedrock_agent_client.get_agent_alias(
            agentId=agent_id,
            agentAliasId=alias_id,
        )["agentAlias"]["agentAliasStatus"]
    if status != "PREPARED":
        raise ValueError(
            f"Alias {alias_id} of agent {agent_id} entered unexpected status: {status}"
        )
    logger.info("Done!")
