# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Python Built-Ins:
import base64
from getpass import getpass
import os

# External Dependencies:
from dotenv import load_dotenv
from strands.telemetry import StrandsTelemetry


def set_up_notebook_langfuse(
    dotenv_filepath: str | None = None,
    refresh: bool = False,
) -> StrandsTelemetry | None:
    """Set up Strands tracing in LangFuse via environment variables and saved .env file

    This utility configures Strands Agents SDK to send traces to Langfuse via OpenTelemetry.
    Langfuse credentials are read from the LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, and
    LANGFUSE_SECRET_KEY environment variables - or a local '.env' file.

    If credentials are not found (or a `refresh` is forced), you'll be prompted to enter your
    details via the secure Python built-in `getpass` utility. These will be saved in your local
    .env file, to avoid needing to re-enter them for every notebook / every kernel restart.

    ⚠️ NOTE: Strands OpenTelemetry can only be configured ONCE per session, so if you need to
    change your LangFuse credentials (e.g. using 'refresh'), you'll need to restart your notebook
    kernel first for the change to take effect!

    Parameters
    ----------
    dotenv_filepath :
        Path where the '.env' file (which contains sensitive secrets!) is / should be saved.
        Default: '.env' in the root of the sample repository.
    refresh :
        Set True to force re-prompting for updated Langfuse credentials, instead of using existing.
    """
    dotenv_filepath = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", ".env")
    )
    load_dotenv(dotenv_filepath)

    LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "cloud.langfuse.com")

    if refresh or not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        print("".join(("🔎 ", "-" * 75, " 🔎")))
        print(
            "\n".join(
                (
                    "This utility will set up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and ",
                    "LANGFUSE_HOST environment variables in your notebook, and also store these ",
                    f"values ***unencrypted*** in file:\n{dotenv_filepath}",
                    "",
                    "If you don't want this, interrupt the cell and set up your environment variables",
                    "manually instead.",
                )
            )
        )
        LANGFUSE_SECRET_KEY = getpass(prompt="LANGFUSE_SECRET_KEY:")
        LANGFUSE_PUBLIC_KEY = getpass(prompt="LANGFUSE_PUBLIC_KEY:")
        LANGFUSE_HOST = input("LANGFUSE_HOST_URL:")
        LANGFUSE_HOST = LANGFUSE_HOST or "https://cloud.langfuse.com"

        os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST
        os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
        
        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_HOST:
            with open(dotenv_filepath, "a") as env_file:
                env_file.writelines(
                    [
                        "\n",
                        f"LANGFUSE_HOST={LANGFUSE_HOST}\n",
                        f"LANGFUSE_PUBLIC_KEY={LANGFUSE_PUBLIC_KEY}\n",
                        f"LANGFUSE_SECRET_KEY={LANGFUSE_SECRET_KEY}\n",
                    ]
                )
            print(".env file saved")
        else:
            print("⚠️ Variables not set - aborting")
            return

    # Build Basic Auth header.
    LANGFUSE_AUTH = base64.b64encode(
        f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()

    # Configure OpenTelemetry endpoint & headers
    LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "cloud.langfuse.com")
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{LANGFUSE_HOST}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    # Configure the telemetry
    # (Creates new tracer provider and sets it as global)
    strands_telemetry = StrandsTelemetry().setup_otlp_exporter()
    print("🔎 Langfuse tracing configured")
    return strands_telemetry
