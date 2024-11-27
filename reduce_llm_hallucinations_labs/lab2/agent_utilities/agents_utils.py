from IPython.display import display, HTML
import pandas as pd

import pprint
import logging
import json
from IPython.display import JSON
import os, shutil

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
from langchain_core.messages import HumanMessage
import boto3

logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)


embedding_model_id = "amazon.titan-embed-text-v2:0"
llm_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

bedrock_runtime_client = boto3.client("bedrock-runtime")

model_kwargs = {
    "max_tokens": 4000,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}


def test_llm_call(input_prompt):
    llm = BedrockChat(
        client=bedrock_runtime_client, model_id=llm_model_id, model_kwargs=model_kwargs
    )
    messages = [HumanMessage(content=f"{input_prompt}")]
    response = llm(messages)

    if str(type(response)) == "<class 'langchain_core.messages.ai.AIMessage'>":
        response = response.content
        response = response.strip()

    return response


def clean_up_trace_files(trace_file_path):
    # cleanup trace files to avoid issues
    if os.path.isdir(trace_file_path):
        shutil.rmtree(trace_file_path)
    os.mkdir(trace_file_path)


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))  # .replace("\\n","<p>")


def format_final_response(
    question_id, question, final_answer, lab_number, turn_number, show_detailed=True
):
    question_id_list = [question_id]
    question_list = [question]
    agent_answer_list = [final_answer]
    finalAPIResponse = None
    kbResponse = None
    hallucinationScore = None

    if show_detailed is True:

        generated_sql = list()
        trace_file_name = f"trace_files/actionGroupInvocationOutput_lab{lab_number}_hallucination_agent_trace_{turn_number}.log"
        with open(trace_file_name, "r") as agent_trace_fp:

            # file_text = agent_trace_fp.read().replace("\"", "\'")
            # print(file_text)
            file_json = json.load(agent_trace_fp)
            # print(file_json)
            # print(type(file_json))
            file_json["text"] = file_json["text"].replace('"', "")
            file_json["text"] = file_json["text"].replace('\\"', "")
            file_json["text"] = file_json["text"].replace("'", '"')
            # file_json["text"] = file_json["text"].replace("\'", "\"")
            file_json["text"] = file_json["text"].replace('"s', "s")
            file_json["text"] = file_json["text"].replace("'s", "s")
            file_json["text"] = file_json["text"].replace("(", "")
            file_json["text"] = file_json["text"].replace(")", "")
            # print(file_json["text"])

            file_json = json.loads(file_json["text"])
            # print(file_json)
            # print(file_json["response"])
            finalAPIResponse = [file_json["response"]["finalAPIResponse"]]
            # print(f"finalAPIResponse >> {finalAPIResponse}")
            kbResponse = [file_json["response"]["kbResponse"]]
            # print(f"kbResponse >> {kbResponse}")
            hallucinationScore = [file_json["response"]["hallucinationScore"]]
            # print(f"hallucinationScore >> {hallucinationScore}")

        # Store and print as a dataframe
        response_df = pd.DataFrame(
            list(
                zip(
                    question_id_list,
                    question_list,
                    finalAPIResponse,
                    kbResponse,
                    hallucinationScore,
                )
            ),
            columns=[
                "Question ID",
                "User Question",
                "Agent/Chatbot Response",
                "KB Response",
                "Answer Score",
            ],
        )
        response_df.style.set_properties(
            **{"text-align": "left", "border": "1px solid black"}
        )
        with pd.option_context("display.max_colwidth", None):
            pretty_print(response_df)
    else:
        # Store and print as a dataframe
        response_df = pd.DataFrame(
            list(zip(question_id_list, question_list, agent_answer_list)),
            columns=["Question ID", "User Question", "Agent Response"],
        )
        response_df.style.set_properties(
            **{"text-align": "left", "border": "1px solid black"}
        )
        response_df.to_string(justify="left", index=False)
        with pd.option_context("display.max_colwidth", None):
            pretty_print(response_df)


def invoke_agent_generate_response(
    bedrock_agent_runtime_client,
    input_text,
    agent_id,
    agent_alias_id,
    session_id,
    enable_trace,
    end_session,
    trace_filename_prefix,
    turn_number,
):

    # invoke the agent API
    agentResponse = bedrock_agent_runtime_client.invoke_agent(
        inputText=input_text,
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,
        enableTrace=enable_trace,
        endSession=end_session,
    )

    # logger.info(pprint.pprint(agentResponse))
    event_stream = agentResponse["completion"]
    final_answer = None
    try:
        for event in event_stream:
            if "chunk" in event:
                data = event["chunk"]["bytes"]
                final_answer = data.decode("utf8")
                # logger.info(f"Final answer ->\n{final_answer}")
                end_event_received = True
                # End event indicates that the request finished successfully
            elif "trace" in event:
                # logger.info(json.dumps(event['trace'], indent=2))
                with open(
                    "trace_files/full_trace_"
                    + trace_filename_prefix
                    + "_"
                    + str(turn_number)
                    + ".log",
                    "a",
                ) as agent_trace_fp:
                    agent_trace_fp.writelines(json.dumps(event["trace"], indent=2))

                logger.debug(
                    f"entering if loop>>>> {type(event['trace'])} and keys ::: {event['trace']['trace'].keys()}"
                )
                # only save the last trace output for clear display
                if "preProcessingTrace" in event["trace"]["trace"]:
                    logger.debug("saving pre-processing log")
                    with open(
                        "trace_files/preProcessingTrace_"
                        + trace_filename_prefix
                        + "_"
                        + str(turn_number)
                        + ".log",
                        "w",
                    ) as agent_trace_fp:
                        agent_trace_fp.writelines(
                            json.dumps(
                                event["trace"]["trace"]["preProcessingTrace"], indent=2
                            )
                        )

                elif (
                    "orchestrationTrace" in event["trace"]["trace"]
                    and "observation" in event["trace"]["trace"]["orchestrationTrace"]
                    and "knowledgeBaseLookupOutput"
                    in event["trace"]["trace"]["orchestrationTrace"]["observation"]
                ):
                    logger.debug("saving knowledgeBaseLookupOutput log")
                    with open(
                        "trace_files/knowledgeBaseLookupOutput_"
                        + trace_filename_prefix
                        + "_"
                        + str(turn_number)
                        + ".log",
                        "w",
                    ) as agent_trace_fp:
                        agent_trace_fp.writelines(
                            json.dumps(
                                event["trace"]["trace"]["orchestrationTrace"][
                                    "observation"
                                ]["knowledgeBaseLookupOutput"],
                                indent=2,
                            )
                        )

                elif (
                    "orchestrationTrace" in event["trace"]["trace"]
                    and "observation" in event["trace"]["trace"]["orchestrationTrace"]
                    and "actionGroupInvocationOutput"
                    in event["trace"]["trace"]["orchestrationTrace"]["observation"]
                ):
                    logger.debug("saving actionGroupInvocationOutput log")
                    with open(
                        "trace_files/actionGroupInvocationOutput_"
                        + trace_filename_prefix
                        + "_"
                        + str(turn_number)
                        + ".log",
                        "w",
                    ) as agent_trace_fp:
                        agent_trace_fp.writelines(
                            json.dumps(
                                event["trace"]["trace"]["orchestrationTrace"][
                                    "observation"
                                ]["actionGroupInvocationOutput"],
                                indent=2,
                            )
                        )

                elif "orchestrationTrace" in event["trace"]["trace"]:
                    logger.debug("saving orchestrationTrace log")
                    with open(
                        "trace_files/orchestrationTrace_"
                        + trace_filename_prefix
                        + "_"
                        + str(turn_number)
                        + ".log",
                        "w",
                    ) as agent_trace_fp:
                        agent_trace_fp.writelines(
                            json.dumps(
                                event["trace"]["trace"]["orchestrationTrace"], indent=2
                            )
                        )
            else:
                raise Exception("unexpected event.", event)

    except Exception as e:
        raise Exception("unexpected event.", e)

    return final_answer
