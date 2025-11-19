# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""A Strands SageMakerAIModel-like class extended to support callback-based guardrail checks"""

# Python Built-Ins:
from __future__ import annotations
import abc
from contextlib import nullcontext
import json
from logging import getLogger
from typing import Any, AsyncGenerator, Optional, Protocol, TypeVar, Union

# External Dependencies:
from boto3 import Session
from botocore.config import Config as BotocoreConfig
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletion, ChatCompletionChunk
import opentelemetry.trace as trace_api
from pydantic import BaseModel
from strands.models._validation import warn_on_tool_choice_not_supported
from strands.models.sagemaker import SageMakerAIModel, ToolCall, UsageMetadata
from strands.telemetry import get_tracer
from strands.types.content import Message, Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

# Local Dependencies:
from ..types.tracing import TraceAttributes


logger = getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


class OverallResponseChecker(Protocol):
    """Non-streaming guardrail check

    You don't need to subclass this protocol - just define a compatible async function.

    Parameters
    ----------
    response :
        The original response from the LLM

    Return Values
    -------------
    final_response :
        The updated response after applying the guardrail check
    trace_attributes :
        Optionally, also provide a mapping of metadata to attach to this check's span in the active
        tracing tool.
    """

    async def __call__(
        self, response: ChatCompletion
    ) -> Union[ChatCompletion, tuple[ChatCompletion, TraceAttributes]]: ...


class ResponseChunkChecker(Protocol):
    """Streaming guardrail check (called for each chunk as it's generated)

    You don't need to subclass this protocol - just define a compatible async function.

    Parameters
    ----------
    chunk :
        The current chunk of response data from the LLM
    state :
        A state aggregating all chunks so far (you can use this to fetch a snapshot of the overall
        response if needed)

    Return Values
    -------------
    edited_chunk :
        The updated current chunk after applying the guardrail check
    trace_attributes :
        Optionally, also provide a mapping of metadata to attach to this check's span in the active
        tracing tool.
    """

    async def __call__(
        self, chunk: ChatCompletionChunk, state: ChatCompletionStreamState
    ) -> Union[ChatCompletionChunk, tuple[ChatCompletionChunk, TraceAttributes]]: ...


class _StreamingCheckManager:
    """Class to manage calling streaming guardrail checks

    Since this process requires some state to be tracked over the course of the LLM response
    generation, we factor it out into a separate class to keep the core
    `SageMakerAIModelWithChecks.stream` implementation as close as possible to the Strands base.
    """

    def __init__(
        self,
        checkers: list[ResponseChunkChecker],
        tool_specs: list[ToolSpec] | None = None,
    ):
        self.checkers = checkers
        self.finalized_chunks: list[ChatCompletionChunk] = []
        self.finalized_stream_state = self._create_stream_state()
        self._tool_specs = tool_specs
        self._working_stream_state = self._create_stream_state()

    def _create_stream_state(
        self, initial_chunks: list[ChatCompletionChunk] | None = None
    ):
        state = ChatCompletionStreamState(
            # TODO: Tools get kind of complicated because Strands' schema's different
            # input_tools=self._tool_specs,
        )
        if initial_chunks:
            for chunk in initial_chunks:
                state.handle_chunk(chunk)
        return state

    async def process_chunk(self, chunk: dict) -> dict:
        curr_openai_chunk = ChatCompletionChunk(**chunk)
        self._working_stream_state.handle_chunk(curr_openai_chunk)
        tracer = get_tracer()
        parent_span = trace_api.get_current_span()
        for checker in self.checkers:
            with (
                tracer.tracer.start_as_current_span(
                    getattr(checker, "__name__", "response_chunk_checker"),
                    attributes={
                        # https://opentelemetry.io/docs/specs/semconv/gen-ai/
                        # https://github.com/langfuse/langfuse-python/blob/06912ce1581e9aedc5ae5951310d1c1469090808/langfuse/_client/attributes.py#L39
                        "langfuse.observation.type": "guardrail",
                        "gen_ai.input.messages": curr_openai_chunk.model_dump_json(),
                        "gen_ai.input.type": "json",
                    },
                )
                if parent_span
                else nullcontext()
            ) as span:
                # Run the checker:
                check_response = await checker(
                    chunk=curr_openai_chunk,
                    state=self._working_stream_state,
                )
                # Handle multiple return type options:
                if isinstance(check_response, tuple):
                    edited_openai_chunk, metadata = check_response
                else:
                    edited_openai_chunk = check_response
                    metadata = {}

                # Check if the checker edited the chunk, and handle the change if so:
                # (So that further checkers in the list, or checker on future chunks, will see a
                # state that's in line with what actually got accepted)
                if edited_openai_chunk != curr_openai_chunk:
                    self._working_stream_state = self._create_stream_state(
                        self.finalized_chunks + [edited_openai_chunk]
                    )
                    curr_openai_chunk = edited_openai_chunk

                # Add metadata to the trace
                if span:
                    span.set_attributes(
                        {
                            "gen_ai.output.messages": curr_openai_chunk.model_dump_json(),
                            "gen_ai.output.type": "json",
                            **metadata,
                        }
                    )
        # All checkers now ran - update for and return the finalized chunk:
        self.finalized_chunks.append(curr_openai_chunk)
        self.finalized_stream_state.handle_chunk(curr_openai_chunk)
        return curr_openai_chunk.model_dump()


class SageMakerAIModelWithChecks(SageMakerAIModel, abc.ABC):
    """Customized Strands SageMakerAIModel class to support additional checker/guardrail callbacks

    This implementation assumes you're using a model that supports OpenAI-compatible Chat
    Completions API.

    Parameters
    ----------
    endpoint_config :
        As per Strands SageMakerAIModel
    payload_config :
        As per Strands SageMakerAIModel
    boto_session :
        As per Strands SageMakerAIModel
    boto_client_config :
        As per Strands SageMakerAIModel
    overall_response_checkers :
        A list of callback functions to validate (and possibly edit) the overall LLM response.
        If the model is called in streaming mode, these will still be called at the end of the
        response stream but an error will be raised if they try to edit the response - as the
        data chunks will already have been propagated through to the Strands event loop.
    response_chunk_checkers :
        A list of callback functions to validate individual chunks of LLM response as they're
        generated in streaming mode. If the model is called in non-streaming mode, these will
        be ignored.
    """

    def __init__(
        self,
        endpoint_config: SageMakerAIModel.SageMakerAIEndpointConfig,
        payload_config: SageMakerAIModel.SageMakerAIPayloadSchema,
        boto_session: Session | None = None,
        boto_client_config: BotocoreConfig | None = None,
        overall_response_checkers: list[OverallResponseChecker] | None = None,
        response_chunk_checkers: list[ResponseChunkChecker] | None = None,
    ):
        super().__init__(
            endpoint_config, payload_config, boto_session, boto_client_config
        )
        self.overall_response_checkers = overall_response_checkers or []
        self.response_chunk_checkers = response_chunk_checkers or []

    async def _run_response_checks(self, response: dict) -> dict:
        """Handle running non-streaming checks with tracing metadata

        Returns the final response after all checks have been called
        """
        # Log tracing info if available:
        tracer = get_tracer()
        parent_span = trace_api.get_current_span()
        curr_response_parsed = ChatCompletion(**response)
        for checker in self.overall_response_checkers:
            with (
                tracer.tracer.start_as_current_span(
                    getattr(checker, "__name__", "overall_response_checker"),
                    attributes={
                        # https://opentelemetry.io/docs/specs/semconv/gen-ai/
                        # https://github.com/langfuse/langfuse-python/blob/06912ce1581e9aedc5ae5951310d1c1469090808/langfuse/_client/attributes.py#L39
                        "langfuse.observation.type": "guardrail",
                        "gen_ai.input.messages": curr_response_parsed.model_dump_json(),
                        "gen_ai.input.type": "json",
                    },
                )
                if parent_span
                else nullcontext()
            ) as span:
                # Run the checker:
                check_response = await checker(curr_response_parsed)
                # Handle multiple return type options:
                if isinstance(check_response, tuple):
                    curr_response_parsed, metadata = check_response
                else:
                    curr_response_parsed = check_response
                    metadata = {}

                # Add metadata to the trace:
                if span:
                    span.set_attributes(
                        {
                            "gen_ai.output.messages": curr_response_parsed.model_dump_json(),
                            "gen_ai.output.type": "json",
                            # Note setting top-level trace & observation metadata seems to work, but can't
                            # dashboard them as metrics:
                            # "langfuse.trace.metadata.semantic_similarity": 0.456789,
                            # "langfuse.observation.metadata.semantic_similarity": 0.456789,
                            # "langfuse.observation.status_message": "Guardrail intervened",
                            # "semantic_similarity": 0.12345678,
                            **metadata,
                        }
                    )
        # All checkers now ran - return the finalized chunk:
        final_response = curr_response_parsed.model_dump()
        # TODO: Potential bug report?
        # strands.models.sagemaker.UsageMetadata doesn't support this key:
        (final_response.get("usage") or {}).pop("completion_tokens_details", None)
        return final_response

    async def stream(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Customized from Strands SageMakerAIModel.stream to implement guardrail callbacks"""
        warn_on_tool_choice_not_supported(tool_choice)

        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug("formatted request=<%s>", request)

        logger.debug("invoking model")

        try:
            if self.payload_config.get("stream", True):
                response = self.client.invoke_endpoint_with_response_stream(**request)

                logger.debug("Initialising manager for chunk-level check callbacks")
                chunk_checker = _StreamingCheckManager(
                    checkers=self.response_chunk_checkers,
                    tool_specs=tool_specs,
                )

                # Message start
                yield self.format_chunk({"chunk_type": "message_start"})

                # Parse the content
                finish_reason = ""
                partial_content = ""
                tool_calls: dict[int, list[Any]] = {}
                has_text_content = False
                text_content_started = False
                reasoning_content_started = False

                for event in response["Body"]:
                    chunk = event["PayloadPart"]["Bytes"].decode("utf-8")
                    partial_content += (
                        chunk[6:] if chunk.startswith("data: ") else chunk
                    )  # TGI fix
                    logger.info("chunk=<%s>", partial_content)
                    try:
                        content = json.loads(partial_content)

                        # Chunk-level guardrail checks:
                        content = await chunk_checker.process_chunk(content)

                        partial_content = ""
                        choice = content["choices"][0]
                        logger.info("choice=<%s>", json.dumps(choice, indent=2))

                        # Handle text content
                        if choice["delta"].get("content", None):
                            if not text_content_started:
                                yield self.format_chunk(
                                    {"chunk_type": "content_start", "data_type": "text"}
                                )
                                text_content_started = True
                            has_text_content = True
                            yield self.format_chunk(
                                {
                                    "chunk_type": "content_delta",
                                    "data_type": "text",
                                    "data": choice["delta"]["content"],
                                }
                            )

                        # Handle reasoning content
                        if choice["delta"].get("reasoning_content", None):
                            if not reasoning_content_started:
                                yield self.format_chunk(
                                    {
                                        "chunk_type": "content_start",
                                        "data_type": "reasoning_content",
                                    }
                                )
                                reasoning_content_started = True
                            yield self.format_chunk(
                                {
                                    "chunk_type": "content_delta",
                                    "data_type": "reasoning_content",
                                    "data": choice["delta"]["reasoning_content"],
                                }
                            )

                        # Handle tool calls (Treat both missing & =None as empty list)
                        generated_tool_calls = (
                            choice["delta"].get("tool_calls", []) or []
                        )
                        if not isinstance(generated_tool_calls, list):
                            generated_tool_calls = [generated_tool_calls]
                        for tool_call in generated_tool_calls:
                            tool_calls.setdefault(tool_call["index"], []).append(
                                tool_call
                            )

                        if choice["finish_reason"] is not None:
                            finish_reason = choice["finish_reason"]
                            break

                        if choice.get("usage", None):
                            yield self.format_chunk(
                                {
                                    "chunk_type": "metadata",
                                    "data": UsageMetadata(**choice["usage"]),
                                }
                            )

                    except json.JSONDecodeError:
                        # Continue accumulating content until we have valid JSON
                        continue

                overall_resp = chunk_checker.finalized_stream_state.get_final_completion().model_dump()
                edited_overall = await self._run_response_checks(overall_resp)
                if edited_overall != overall_resp:
                    raise ValueError(
                        "An overall_response_checker tried to edit the LLM response, but this is "
                        "not supported in streaming mode as events were already emitted!"
                    )

                # Close reasoning content if it was started
                if reasoning_content_started:
                    yield self.format_chunk(
                        {"chunk_type": "content_stop", "data_type": "reasoning_content"}
                    )

                # Close text content if it was started
                if text_content_started:
                    yield self.format_chunk(
                        {"chunk_type": "content_stop", "data_type": "text"}
                    )

                # Handle tool calling
                logger.info("tool_calls=<%s>", json.dumps(tool_calls, indent=2))
                for tool_deltas in tool_calls.values():
                    if not tool_deltas[0]["function"].get("name", None):
                        raise Exception("The model did not provide a tool name.")
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_start",
                            "data_type": "tool",
                            "data": ToolCall(**tool_deltas[0]),
                        }
                    )
                    for tool_delta in tool_deltas:
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "tool",
                                "data": ToolCall(**tool_delta),
                            }
                        )
                    yield self.format_chunk(
                        {"chunk_type": "content_stop", "data_type": "tool"}
                    )

                # If no content was generated at all, ensure we have empty text content
                if not has_text_content and not tool_calls:
                    yield self.format_chunk(
                        {"chunk_type": "content_start", "data_type": "text"}
                    )
                    yield self.format_chunk(
                        {"chunk_type": "content_stop", "data_type": "text"}
                    )

                # Message close
                yield self.format_chunk(
                    {"chunk_type": "message_stop", "data": finish_reason}
                )

            else:
                # Not all SageMaker AI models support streaming!
                response = self.client.invoke_endpoint(**request)  # type: ignore[assignment]
                final_response_json = json.loads(
                    response["Body"].read().decode("utf-8")
                )  # type: ignore[attr-defined]
                logger.info("response=<%s>", json.dumps(final_response_json, indent=2))

                # Run non-streaming checks here:
                final_response_json = await self._run_response_checks(
                    final_response_json
                )

                # Obtain the key elements from the response
                message = final_response_json["choices"][0]["message"]
                message_stop_reason = final_response_json["choices"][0]["finish_reason"]

                # Message start
                yield self.format_chunk({"chunk_type": "message_start"})

                # Handle text
                if message.get("content", ""):
                    yield self.format_chunk(
                        {"chunk_type": "content_start", "data_type": "text"}
                    )
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "text",
                            "data": message["content"],
                        }
                    )
                    yield self.format_chunk(
                        {"chunk_type": "content_stop", "data_type": "text"}
                    )

                # Handle reasoning content
                if message.get("reasoning_content", None):
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_start",
                            "data_type": "reasoning_content",
                        }
                    )
                    yield self.format_chunk(
                        {
                            "chunk_type": "content_delta",
                            "data_type": "reasoning_content",
                            "data": message["reasoning_content"],
                        }
                    )
                    yield self.format_chunk(
                        {"chunk_type": "content_stop", "data_type": "reasoning_content"}
                    )

                # Handle the tool calling, if any
                if (
                    message.get("tool_calls", None)
                    or message_stop_reason == "tool_calls"
                ):
                    if not isinstance(message["tool_calls"], list):
                        message["tool_calls"] = [message["tool_calls"]]
                    for tool_call in message["tool_calls"]:
                        # if arguments of tool_call is not str, cast it
                        if not isinstance(tool_call["function"]["arguments"], str):
                            tool_call["function"]["arguments"] = json.dumps(
                                tool_call["function"]["arguments"]
                            )
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_start",
                                "data_type": "tool",
                                "data": ToolCall(**tool_call),
                            }
                        )
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": "tool",
                                "data": ToolCall(**tool_call),
                            }
                        )
                        yield self.format_chunk(
                            {"chunk_type": "content_stop", "data_type": "tool"}
                        )
                    message_stop_reason = "tool_calls"

                # Message close
                yield self.format_chunk(
                    {"chunk_type": "message_stop", "data": message_stop_reason}
                )
                # Handle usage metadata
                if final_response_json.get("usage", None):
                    yield self.format_chunk(
                        {
                            "chunk_type": "metadata",
                            "data": UsageMetadata(
                                **final_response_json.get("usage", None)
                            ),
                        }
                    )
        except (
            self.client.exceptions.InternalFailure,
            self.client.exceptions.ServiceUnavailable,
            self.client.exceptions.ValidationError,
            self.client.exceptions.ModelError,
            self.client.exceptions.InternalDependencyException,
            self.client.exceptions.ModelNotReadyException,
        ) as e:
            logger.error("SageMaker error: %s", str(e))
            raise e

        logger.debug("finished streaming response from model")

    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """TEMPORARY BUGFIX OVERRIDE for additional_args

        See: https://github.com/strands-agents/sdk-python/pull/983

        TODO: Remove this override when linked PR is merged and upstream updated
        """
        formatted_messages = self.format_request_messages(messages, system_prompt)

        payload = {
            "messages": formatted_messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            # Add payload configuration parameters
            **{
                k: v
                for k, v in self.payload_config.items()
                if k not in ["additional_args", "tool_results_as_user_messages"]
            },
        }

        payload_additional_args = self.payload_config.get("additional_args")
        if payload_additional_args:
            payload.update(payload_additional_args)

        # Remove tools and tool_choice if tools = []
        if not payload["tools"]:
            payload.pop("tools")
            payload.pop("tool_choice", None)
        else:
            # Ensure the model can use tools when available
            payload["tool_choice"] = "auto"

        for message in payload["messages"]:  # type: ignore
            # Assistant message must have either content or tool_calls, but not both
            if (
                message.get("role", "") == "assistant"
                and message.get("tool_calls", []) != []
            ):
                message.pop("content", None)
            if message.get("role") == "tool" and self.payload_config.get(
                "tool_results_as_user_messages", False
            ):
                # Convert tool message to user message
                tool_call_id = message.get("tool_call_id", "ABCDEF")
                content = message.get("content", "")
                message = {
                    "role": "user",
                    "content": f"Tool call ID '{tool_call_id}' returned: {content}",
                }
            # Cannot have both reasoning_text and text - if "text", content becomes an array of content["text"]
            for c in message.get("content", []):
                if "text" in c:
                    message["content"] = [c]
                    break
            # Cast message content to string for TGI compatibility
            # message["content"] = str(message.get("content", ""))

        logger.info("payload=<%s>", json.dumps(payload, indent=2))
        # Format the request according to the SageMaker Runtime API requirements
        request = {
            "EndpointName": self.endpoint_config["endpoint_name"],
            "Body": json.dumps(payload),
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        # Add optional SageMaker parameters if provided
        inf_component_name = self.endpoint_config.get("inference_component_name")
        if inf_component_name:
            request["InferenceComponentName"] = inf_component_name
        target_model = self.endpoint_config.get("target_model")
        if target_model:
            request["TargetModel"] = target_model
        target_variant = self.endpoint_config.get("target_variant")
        if target_variant:
            request["TargetVariant"] = target_variant

        # Add additional request args if provided
        additional_args = self.endpoint_config.get("additional_args")
        if additional_args:
            request.update(additional_args)

        return request
