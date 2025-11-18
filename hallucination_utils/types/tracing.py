# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Type definitions related to tracing and instrumentation"""

# Python Built-Ins:
from typing import Mapping, TypeAlias

# External Dependencies:
from strands.types.traces import AttributeValue

# Type alias for tracing attributes (compatible with Python 3.10+)
TraceAttributes: TypeAlias = Mapping[str, AttributeValue]
