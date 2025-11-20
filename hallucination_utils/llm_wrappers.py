# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Simple, boto3-based, LangChain-like wrappers for invoking FMs on AWS

From Bedrock to SageMaker to other Foundation Model inference services, APIs differ because
different functionality is on offer. Often though, we'd like to skim over that complexity when
doing something basic - like just sending a text message to an LLM and getting a text response.

Popular Open Source libraries like LangChain and LiteLLM provide these generic translation layers
for a very wide variety of models, but can be more bloated than necessary, in cases where only a
smaller set of models or functionality is needed.

The LLM wrappers we provide here are lightweight, based directly on the AWS SDK for Python (boto3),
and implement a subset of LangChain's LLM.invoke(...) API: Providing familiarity for LangChain
users, but demonstrating that bringing in a dependency like this is not strictly necessary.
"""

# Python Built-Ins:
from dataclasses import dataclass
import json
from typing import Any

# External Dependencies:
import boto3  # AWS SDK for Python
from botocore.client import BaseClient as Boto3Client


@dataclass
class BedrockLLM:
    """Cut-down, LangChain-style wrapper to fetch text responses from an LLM on Amazon Bedrock

    For more fully-featured alternatives, check out langchain-aws, litellm, or other libraries.
    langchain_aws.llms.BedrockLLM should be a drop-in replacement, but we'd recommend starting
    with langchain_aws.chat_models.ChatBedrockConverse instead.
    """

    model_id: str
    client: Boto3Client | None = None
    max_tokens: int = 200
    temperature: float = 0.7

    def __post_init__(self):
        if not self.client:
            self.client = boto3.client("bedrock-runtime")

    def invoke(self, prompt: str) -> str:
        """Invoke the model with a text prompt, and return the text output"""
        # Call the model on Bedrock:
        response = self.client.converse(
            modelId=self.model_id,
            # See Conversation messages format for Converse API:
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        # Extract the response text from the bedrock Converse API response:
        return response["output"]["message"]["content"][0]["text"]


@dataclass
class SageMakerVLLM:
    """Cut-down, LangChain-style wrapper for ChatCompletions-compatible FMs on SageMaker

    For more fully-featured alternatives, check out langchain-aws, litellm, or other libraries.
    langchain_aws.llms.SageMakerEndpoint should be a drop-in replacement, but we'd recommend
    starting with langchain_aws.chat_models.ChatSageMakerEndpoint instead.
    """

    endpoint_name: str
    client: Boto3Client | None = None
    inference_component_name: str | None = None
    # Note GPT-OSS max_tokens *includes reasoning*! Can lead to empty responses when too small
    max_tokens: int | None = None
    temperature: float | None = 0.7

    def __post_init__(self):
        if not self.client:
            self.client = boto3.client("sagemaker-runtime")

    def invoke(self, prompt: str) -> str:
        """Invoke the model with a text prompt, and return the text output"""
        # Prepare SageMaker input arguments:
        req_body = {
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.max_tokens:
            req_body["max_tokens"] = self.max_tokens
        if self.temperature:
            req_body["temperature"] = self.temperature
        req = {
            "EndpointName": self.endpoint_name,
            "Body": json.dumps(req_body),
        }
        if self.inference_component_name:
            req["InferenceComponentName"] = self.inference_component_name

        # Invoke the SageMaker endpoint:
        response = self.client.invoke_endpoint(**req)

        # Extract plain text from the response:
        result = json.loads(response["Body"].read().decode())
        if "choices" in result and len(result["choices"]) > 0:
            text_response = result["choices"][0]["message"]["content"]
        elif "generated_text" in result:
            text_response = result["generated_text"]
        else:
            raise ValueError(
                "Expected SageMaker response to contain either 'choices' or 'generated_text'. "
                "Got: %s" % response
            )
        if text_response is None:
            raise ValueError(
                "Output content was empty. Maybe a low max_tokens cut off generation while "
                "the model was still reasoning? Got: %s" % result
            )
        return text_response
