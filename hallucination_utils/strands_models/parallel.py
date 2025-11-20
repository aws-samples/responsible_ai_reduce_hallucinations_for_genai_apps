# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Wrappers for calling multiple LLMs in parallel in Strands-Agents and consolidating the results"""

# Python Built-Ins:
from __future__ import annotations
import abc
from contextlib import nullcontext
import json
from logging import getLogger
from typing import Any, AsyncGenerator, Generator, Mapping, Sequence, TypeVar

# External Dependencies:
import opentelemetry.trace as trace_api
from pydantic import BaseModel
from strands.event_loop.streaming import process_stream
from strands.models.model import Model
from strands.telemetry import get_tracer
from strands.types.content import Message
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from strands.types.traces import AttributeValue
from strands.types.event_loop import Metrics, StopReason, Usage

# Local Dependencies:
from ..streaming import collect_async_iterables, list_to_async_iterable
from ..types.tracing import TraceAttributes


logger = getLogger(__name__)


def _aggregate_metrics(*metrics: Metrics) -> Metrics:
    return Metrics(
        latencyMs=sum(m["latencyMs"] for m in metrics),
    )


def _aggregate_usages(*usages: Usage) -> Usage:
    return Usage(
        inputTokens=sum(u["inputTokens"] for u in usages),
        outputTokens=sum(u["outputTokens"] for u in usages),
        totalTokens=sum(u["totalTokens"] for u in usages),
        cacheReadInputTokens=sum(u.get("cacheReadInputTokens", 0) for u in usages),
        cacheWriteInputTokens=sum(u.get("cacheWriteInputTokens", 0) for u in usages),
    )


class ModelResponse(BaseModel):
    """Custom class representing a (non-streaming) response from a LLM in Strands-Agents

    You probably want to use the `.from_alternatives(...)` constructor in the workshop.

    Strands is stream-based but we want to hide that complexity to focus on what's important for
    the workshop, so this class holds a snapshot representation of an overall LLM response on
    completion.
    """

    message: Message
    metrics: Metrics
    stop_reason: StopReason
    usage: Usage

    @classmethod
    def from_alternatives(
        cls,
        alternatives: list[ModelResponse],
        final_message: Message,
        override_metrics: Metrics | None = None,
        override_stop_reason: StopReason | None = None,
        override_usage: Usage | None = None,
    ) -> ModelResponse:
        """Factory method to create a final ModelResponse from a list of candidates

        Parameters
        ----------
        alternatives :
            Raw ModelResponses from your LLM call(s)
        final_message :
            The Message you chose (or otherwise consolidated) from the list of alternatives
        override_metrics :
            By default, the metrics from `alternatives` will be combined taking latency as the max
            latency across each model. You can set `override_metrics` to override this behaviour.
        override_stop_reason :
            By default, if all `alternatives` stop reasons match it will be propagated or else an
            error will be thrown. You can set `override_stop_reason` to override this behaviour.
        override_usage :
            By default, the usage metrics from all `alternatives` will be summed together. You can
            set `override_usage` to override this behaviour.
        """
        message = final_message
        metrics = override_metrics or _aggregate_metrics(
            *[alt.metrics for alt in alternatives]
        )
        if override_stop_reason:
            stop_reason = override_stop_reason
        else:
            if not alternatives:
                raise ValueError(
                    "Must provide either `override_stop_reason` or input `alternatives`"
                )
            elif any(
                a.stop_reason != alternatives[0].stop_reason for a in alternatives
            ):
                raise ValueError(
                    "Can only default stop_reason when all alternatives had the same. Got: %s"
                    % [a.stop_reason for a in alternatives]
                )
            stop_reason = alternatives[0].stop_reason
        usage = override_usage or _aggregate_usages(
            *[alt.usage for alt in alternatives]
        )
        return cls(
            message=message, stop_reason=stop_reason, usage=usage, metrics=metrics
        )

    def stream_events(self) -> Generator[StreamEvent]:
        """Generate Strands streaming format events representing this (non-streaming) response

        Inspired by `strands.models.bedrock.BedrockModel._convert_non_streaming_to_streaming()`
        https://github.com/strands-agents/sdk-python/blob/921ca89f6f0f5e7874c1aa92be83354fc73eb1d4/src/strands/models/bedrock.py#L749
        """
        # Yield messageStart event
        yield {"messageStart": {"role": self.message["role"]}}

        # Process content blocks
        for content in self.message["content"]:
            # Yield contentBlockStart event if needed
            if "toolUse" in content:
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": content["toolUse"]["toolUseId"],
                                "name": content["toolUse"]["name"],
                            }
                        },
                    }
                }

                # For tool use, we need to yield the input as a delta
                input_value = json.dumps(content["toolUse"]["input"])

                yield {
                    "contentBlockDelta": {"delta": {"toolUse": {"input": input_value}}}
                }
            elif "text" in content:
                # Then yield the text as a delta
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": content["text"]},
                    }
                }
            elif "reasoningContent" in content:
                # Then yield the reasoning content as a delta
                reasoning_content = content["reasoningContent"].get("reasoningText")

                if reasoning_content and "text" in reasoning_content:
                    yield {
                        "contentBlockDelta": {
                            "delta": {
                                "reasoningContent": {"text": reasoning_content["text"]}
                            }
                        }
                    }

                # TODO: How to deal with content["reasoningContent"]["redactedContent"]?

                if reasoning_content and "signature" in reasoning_content:
                    yield {
                        "contentBlockDelta": {
                            "delta": {
                                "reasoningContent": {
                                    "signature": reasoning_content["signature"]
                                }
                            }
                        }
                    }
            elif "citationsContent" in content:
                # For non-streaming citations, emit text and metadata deltas in sequence
                # to match streaming behavior where they flow naturally
                if "content" in content["citationsContent"]:
                    text_content = "".join(
                        [
                            content.get("text", "")
                            for content in content["citationsContent"]["content"]
                        ]
                    )
                    yield {
                        "contentBlockDelta": {"delta": {"text": text_content}},
                    }

                for citation in content["citationsContent"].get("citations", []):
                    yield {"contentBlockDelta": {"delta": {"citation": citation}}}

            # Yield contentBlockStop event
            yield {"contentBlockStop": {}}

        # Yield messageStop event
        # Fix stopReason for models that return end_turn when they should return tool_use on
        # non-streaming side
        current_stop_reason = self.stop_reason
        if current_stop_reason == "end_turn":
            message_content = self.message["content"]
            if any("toolUse" in content for content in message_content):
                current_stop_reason = "tool_use"
                logger.warning("Override stop reason from end_turn to tool_use")

        yield {
            "messageStop": {
                "stopReason": current_stop_reason,
                "additionalModelResponseFields": self.message.get(
                    "additionalModelResponseFields"
                ),
            }
        }

        # Yield metadata event
        if self.metrics or self.usage or "trace" in self.message:
            metadata: StreamEvent = {"metadata": {}}
            if self.usage:
                metadata["metadata"]["usage"] = self.usage
            if self.metrics:
                metadata["metadata"]["metrics"] = self.metrics
            if "trace" in self.message:
                metadata["metadata"]["trace"] = self.message["trace"]
            yield metadata


T = TypeVar("T", bound=BaseModel)


class ParallelModelWithConsolidation(Model, abc.ABC):
    """(Abstract) wrapper to call multiple Strands Models in parallel and consolidate the results

    You can use this class like a `strands.models.Model`, but you'll need to subclass it and
    implement your `consolidate()` method first.
    """

    def __init__(self, models: Sequence[Model]):
        """Create a ParallelModelWithConsolidation

        Parameters
        ----------
        models :
            Strands Models to parallelize. This can include repeated representations of the same
            model if you want to just call one model multiple times, for example:
            `models=[my_model] * 5`
        """
        if not models:
            raise ValueError("Must provide at least one strands.models.Model to wrap")
        self.models = models

    def get_config(self) -> list[Any]:
        return [m.get_config() for m in self.models]

    def update_config(self, **_: Any) -> None:
        # (We need to implement this for compliance with strands.models.Model)
        raise ValueError(
            "Can't update config of ParallelModelWithConsolidation - Update the .models list instead"
        )

    @abc.abstractmethod
    def consolidate(
        self, alternatives: list[ModelResponse]
    ) -> ModelResponse | tuple[ModelResponse, TraceAttributes]:
        pass

    async def stream(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        raw_events = await collect_async_iterables(
            *[
                model.stream(
                    messages,
                    tool_specs=tool_specs,
                    system_prompt=system_prompt,
                    tool_choice=tool_choice,
                    **kwargs,
                )
                for model in self.models
            ]
        )

        # Log tracing info if available:
        tracer = get_tracer()
        parent_span = trace_api.get_current_span()
        # TODO: For some reason, adding events don't seem to work? So we're just using a span below
        # parent_span.add_event(
        #     "semantic_similarity_event",
        #     attributes={
        #         "langfuse.observation.type": "guardrail",
        #         "event_based_ssscore": 0.79123
        #     }
        # )
        with (
            tracer.tracer.start_as_current_span(
                "semantic_similarity",
                attributes={
                    # https://opentelemetry.io/docs/specs/semconv/gen-ai/
                    # https://github.com/langfuse/langfuse-python/blob/06912ce1581e9aedc5ae5951310d1c1469090808/langfuse/_client/attributes.py#L39
                    "langfuse.observation.type": "guardrail",
                },
            )
            if parent_span
            else nullcontext()
        ) as span:
            # Use Strands' built-in stream processor to assemble the raw LLM chunk streams into
            # event streams:
            gen_processed_events = await collect_async_iterables(
                *[
                    process_stream(list_to_async_iterable(stream))
                    for stream in raw_events
                ]
            )
            # Recover each generation's actual output message from the final (ModelStopReason)
            # event in each alternative stream:
            # ModelStopReason is {"stop": (stop_reason, message, usage, metrics)}
            gen_alternatives = [
                ModelResponse(
                    message=evts[-1]["stop"][1],
                    metrics=evts[-1]["stop"][3],
                    stop_reason=evts[-1]["stop"][0],
                    usage=evts[-1]["stop"][2],
                )
                for evts in gen_processed_events
            ]

            consolidation_result = self.consolidate(gen_alternatives)
            if isinstance(consolidation_result, tuple):
                consolidated, metadata = consolidation_result
            else:
                consolidated = consolidation_result
                metadata = {}

            if span:
                span.set_attributes(
                    {
                        "gen_ai.input.messages": json.dumps(
                            [alt.message for alt in gen_alternatives]
                        ),
                        "gen_ai.input.type": "json",
                        "gen_ai.output.messages": json.dumps(consolidated.message),
                        "gen_ai.output.type": "json",
                        "gen_ai.request.choice.count": len(self.models),
                        # Note setting top-level trace & observation metadata seems to work, but can't
                        # dashboard them as metrics:
                        # "langfuse.trace.metadata.semantic_similarity": 0.456789,
                        # "langfuse.observation.metadata.semantic_similarity": 0.456789,
                        # "langfuse.observation.status_message": "Guardrail intervened",
                        # "semantic_similarity": 0.12345678,
                        **metadata,
                    }
                )

        # TODO: Could we support redactContent in NonStreamingResponse?
        # raw_events[0].append(
        #     {
        #         "redactContent": {
        #             "redactAssistantContentMessage": self.config.get(
        #                 "guardrail_redact_output_message",
        #                 "[Output redacted by ParallelModel]",
        #             )
        #         }
        #     }
        # )

        for event in consolidated.stream_events():
            yield event

    async def structured_output(
        self,
        output_model: type[T],
        prompt: list[Message],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        # TODO: Validate the workshop notebooks set logging up, or else stick to print?
        print("WARNING: TODO structured_output is not properly implemented")
        logger.warning(
            "%s does not implement parallel/consolidating structured_output yet... Using first "
            "model only.",
            self.__class__.__name__,
        )
        return self.models[0].structured_output(
            output_model, prompt, system_prompt=system_prompt, **kwargs
        )
        # generations = await collect_async_iterators(*[
        #     self.model.structured_output(output_model, prompt, system_prompt=system_prompt, **kwargs)
        #     for _ in range(self.config["n"])
        # ])

        # for val in generations[0]:
        #     yield val
