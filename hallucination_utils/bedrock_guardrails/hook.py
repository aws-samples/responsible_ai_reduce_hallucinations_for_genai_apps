# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Custom Strands-Agents hook for using Amazon Bedrock Guardrails"""

# Python Built-Ins:
from contextlib import nullcontext
import json
from logging import getLogger
from typing import Any, cast

# External Dependencies:
import boto3
import opentelemetry.trace as trace_api
from strands.hooks import (
    AfterModelCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.telemetry import get_tracer
from strands.types.content import ContentBlock, Message

# Local Dependencies:
from .api import ApiApplyGuardrailResponse, ApiGuardrailAction


logger = getLogger("guardrails")


class GuardrailFailedError(ValueError):
    """Error raised when user's message fails guardrail check(s)"""

    guardrail_response: ApiApplyGuardrailResponse

    def __init__(
        self, message: str, guardrail_response: ApiApplyGuardrailResponse
    ) -> None:
        super().__init__(message)
        self.guardrail_response = guardrail_response


def _guardrail_trace_span(is_input: bool, input_content: Any | None = None):
    """Context manager to generate a tracing span for the guardrail (if tracing is enabled)"""
    tracer = get_tracer()
    parent_span = trace_api.get_current_span()
    if parent_span:
        attributes = {
            # https://opentelemetry.io/docs/specs/semconv/gen-ai/
            # https://github.com/langfuse/langfuse-python/blob/06912ce1581e9aedc5ae5951310d1c1469090808/langfuse/_client/attributes.py#L39
            "langfuse.observation.type": "guardrail",
            "gen_ai.guardrail.action": "input" if is_input else "output",
        }
        if input_content:
            attributes["gen_ai.input.messages"] = json.dumps(input_content)
            attributes["gen_ai.input.type"] = "json"
        return tracer.tracer.start_as_current_span(
            "bedrock_applyguardrail",
            attributes=attributes,
        )
    else:
        return nullcontext()


class BedrockGuardrailHook(HookProvider):
    """Custom Strands-Agents 'Hook' using the Bedrock ApplyGuardrail API

    This hook uses AfterModelCallEvent to intercept LLM outputs as early as possible, but uses
    MessageAddedEvent to intercept user inputs because at the time of writing, Strands'
    BeforeInvocationEvent does not include incoming message information.

    As a result, it's important you apply this hook to your agent *before* memory hooks (such as
    AgentCoreMemorySessionManager) to prevent unwanted user inputs leaking into agent memory. For
    more discussion see: https://github.com/strands-agents/sdk-python/issues/1006
    """

    def __init__(
        self,
        guardrail_id: str,
        guardrail_version: str = "DRAFT",
        boto_session: boto3.Session | None = None,
        error_on_intervention: bool = True,
    ):
        """Create a BedrockGuardrailHook

        Parameters
        ----------
        guardrail_id :
            Unique ID of the Amazon Bedrock Guardrail to be applied
        guardrail_version :
            Specific version of the Amazon Bedrock Guardrail to apply. Default: current working
            'DRAFT'
        boto_session :
            Existing boto3 Session to use for interacting with Bedrock. Useful if you want to call
            a guardrail in a different AWS Region, for example.
        error_on_intervention :
            By default (True) a `GuardrailFailedError` will be raised to interrupt the agent loop
            if the Bedrock Guardrail intervenes. Set `False` to instead just redact the message and
            proceed with the event loop.
        """
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.bedrock_client = (boto_session or boto3).client("bedrock-runtime")
        self.error_on_intervention = error_on_intervention

    def register_hooks(self, registry: HookRegistry, **_) -> None:
        registry.add_callback(MessageAddedEvent, self.on_message_added)
        registry.add_callback(AfterModelCallEvent, self.after_model_call)

    def evaluate_content(
        self, content: list[dict], is_input: bool = True
    ) -> ApiApplyGuardrailResponse:
        """Evaluate content using Bedrock ApplyGuardrail API"""
        return ApiApplyGuardrailResponse(
            **self.bedrock_client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source="INPUT" if is_input else "OUTPUT",
                content=content,
            )
        )

    def on_user_message_added(self, event: MessageAddedEvent) -> None:
        """Check a newly-added user message and redact it if necessary"""
        guard_content = []
        content = event.message.get("content", [])
        for block in content:
            if "toolResult" in block:
                # This is a tool response, not a real user message
                continue
            elif "text" in block:
                guard_content.append(
                    {
                        "text": {
                            "text": block["text"],
                            "qualifiers": ["query", "guard_content"],
                        },
                    }
                )
            else:
                raise ValueError(
                    f"User content type not implemented by BedrockGuardrailHook: {block}"
                )
        with _guardrail_trace_span(is_input=True, input_content=guard_content) as span:
            if not guard_content:
                msg = "Input guardrail skipped - no user content to analyze"
                logger.debug(msg)
                if span:
                    span.set_attributes(
                        {
                            "gen_ai.output.messages": json.dumps({"info": msg}),
                            "gen_ai.output.type": "json",
                        }
                    )
                return
            guardrail_resp = self.evaluate_content(guard_content, is_input=True)
            guardrail_intervened = guardrail_resp.action != ApiGuardrailAction.NONE
            if span:
                span.set_attributes(
                    {
                        "gen_ai.output.messages": guardrail_resp.model_dump_json(),
                        "gen_ai.output.type": "json",
                        "gen_ai.guardrail.bedrock_guardrail.intervened": guardrail_intervened,
                    }
                )
            if guardrail_intervened:
                logger.warning(
                    "Redacting last message from history due to failed input guardrail"
                )
                event.agent.messages.pop()
                event.agent.messages.append(
                    Message(
                        content=[
                            ContentBlock(
                                text="[Message redacted for failing input guardrail check]"
                            )
                        ],
                        role="user",
                    )
                )
                if self.error_on_intervention:
                    raise GuardrailFailedError("Input guardrail failed", guardrail_resp)
            else:
                logger.debug("Input guardrail passed")

    def after_model_call(self, event: AfterModelCallEvent) -> None:
        """Check an outbound LLM response and redact it if necessary"""
        if not event.stop_response:
            logger.info("No message received - skipping post-LLM guardrail")
            return
        current_message = event.stop_response.message
        # We'll loop *backwards* through `event.agent.messages`, and reverse the content later:
        guard_content_rev = []
        has_grounding_source = False
        # (Note that the current event's message has *not yet* been added to
        # `event.agent.messages` when this handler is called)
        for message in [current_message] + event.agent.messages[::-1]:
            message_role = message.get("role")
            for block in message.get("content", []):
                if "text" in block:
                    if message_role == "user":
                        guard_content_rev.append(
                            {
                                "text": {
                                    "text": block["text"],
                                    "qualifiers": ["query"],
                                },
                            }
                        )
                        # Don't include anything past the most recent user message in the
                        # guardrail context:
                        break
                    elif message_role == "assistant":
                        guard_content_rev.append(
                            {
                                "text": {
                                    "text": block["text"],
                                    "qualifiers": ["guard_content"],
                                },
                            }
                        )
                    else:
                        raise ValueError(f"Unknown text message role {message_role}")
                elif "toolResult" in block:
                    # In this case we'll treat any tool result as a 'grounding context' source.
                    # You could also choose to e.g. switch on tool_block["name"] for tool name
                    # For more info see:
                    # https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html
                    for tool_block in block["toolResult"].get("content", [])[::-1]:
                        if "text" in tool_block:
                            guard_content_rev.append(
                                {
                                    "text": {
                                        "text": tool_block["text"],
                                        "qualifiers": ["grounding_source"],
                                    },
                                }
                            )
                            has_grounding_source = True
                        else:
                            raise ValueError(
                                f"toolResult content block didn't have 'text'. Got: {tool_block}"
                            )
                elif "toolUse" in block:
                    continue  # Ignore tool requests
                else:
                    raise ValueError(
                        f"Content type not implemented by BedrockGuardrailHook: {block}"
                    )

        if guard_content_rev:
            if not has_grounding_source:
                # Contextual grounding guardrail will raise an error if there's no grounding
                # source, so add an empty one:
                guard_content_rev.append(
                    {
                        "text": {"qualifiers": ["grounding_source"], "text": ""},
                    }
                )
            guard_content = guard_content_rev[::-1]
        else:
            guard_content = []
        with _guardrail_trace_span(is_input=False, input_content=guard_content) as span:
            if not guard_content:
                msg = "Output guardrail skipped - no assistant content to analyze"
                logger.debug(msg)
                if span:
                    span.set_attributes(
                        {
                            "gen_ai.output.messages": json.dumps({"info": msg}),
                            "gen_ai.output.type": "json",
                        }
                    )
                return
            guardrail_resp = self.evaluate_content(guard_content, is_input=False)
            guardrail_intervened = guardrail_resp.action != ApiGuardrailAction.NONE
            if span:
                span.set_attributes(
                    {
                        "gen_ai.output.messages": guardrail_resp.model_dump_json(),
                        "gen_ai.output.type": "json",
                        "gen_ai.guardrail.bedrock_guardrail.intervened": guardrail_intervened,
                    }
                )
            if guardrail_intervened:
                logger.warning(
                    "Redacting outbound message due to failed input guardrail"
                )
                current_message["content"] = (
                    [cast(ContentBlock, m.model_dump()) for m in guardrail_resp.outputs]
                    if guardrail_resp.outputs
                    else [
                        ContentBlock(text="Sorry, I can't help you with that question.")
                    ]
                )
                if self.error_on_intervention:
                    raise GuardrailFailedError(
                        "Output guardrail failed", guardrail_resp
                    )
            else:
                logger.debug("Output guardrail passed")

    def on_message_added(self, event: MessageAddedEvent) -> None:
        """If the newly-added message is from a user, call the input-side guardrail"""
        if event.message.get("role") == "user":
            return self.on_user_message_added(event)
