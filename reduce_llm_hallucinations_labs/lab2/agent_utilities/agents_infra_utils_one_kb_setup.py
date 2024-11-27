import json
import logging
import os
import pprint
import random
import time
import uuid
import zipfile
from io import BytesIO

import boto3
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

### NOTE: change the logging level to DEBUG if infrasetup fails to get more trace on the issue
logging.basicConfig(
    format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    level=logging.ERROR,
)
logger = logging.getLogger(__name__)

prefix_infra = None
prefix_iam = None
suffix = None

sts_client = boto3.client("sts")
iam_client = boto3.client("iam")
s3_client = boto3.client("s3")
lambda_client = boto3.client("lambda")

session = boto3.session.Session()
region = session.region_name
account_id = sts_client.get_caller_identity()["Account"]
region, account_id
# getting boto3 clients for required AWS services

from botocore.config import Config

timeout_config = Config(read_timeout=1800)

bedrock_agent_client = boto3.client("bedrock-agent")
bedrock_agent_runtime_client = boto3.client(
    "bedrock-agent-runtime", config=timeout_config
)
open_search_serverless_client = boto3.client("opensearchserverless")

# getting boto3 clients for required AWS services
sts_client = boto3.client("sts")
iam_client = boto3.client("iam")
s3_client = boto3.client("s3")
lambda_client = boto3.client("lambda")
sns_client = boto3.client("sns", region_name=region)


agent_functions = [
    {
        "name": "detect_measure_hallucination",
        "description": "detect and measure hallucination from knowledge base response",
        "parameters": {
            "question": {
                "description": "user question on Amazon Bedrock",
                "required": True,
                "type": "string",
            },
            "kbResponse": {
                "description": "knowledge base retrieved response for the user question on Amazon Bedrock",
                "required": True,
                "type": "string",
            },
        },
    }
]


def create_topic(name):
    """
    Creates a SNS notification topic.
    """
    try:
        topic = sns_client.create_topic(Name=name)
        logger.info(f"Created SNS topic {name}.")
    except ClientError:
        logger.exception(f"Could not create SNS topic {name}.")
        raise
    else:
        return topic


def subscribe_to_sns(sns_topic_name):
    this_lambda_name = "LambdaAgentsHallucinationDetection"
    # region, account_id
    # add sns call to customer service queue - separate notebook to see queue
    sns_complete_arn = f"arn:aws:sns:{region}:{account_id}:{sns_topic_name}"
    lambda_complete_arn = (
        f"arn:aws:lambda:{region}:{account_id}:function:{this_lambda_name}"
    )

    subscriptionArn = sns_client.subscribe(
        TopicArn=sns_complete_arn,
        Protocol="lambda",
        Endpoint=lambda_complete_arn,
        ReturnSubscriptionArn=True,
    )["SubscriptionArn"]
    print(
        f"Lambda :: {this_lambda_name} is now subscribed to SNS :: {sns_complete_arn} and subscriptionArn is {subscriptionArn}"
    )


def delete_topic(topic_name):
    """
    Delete a SNS topic.
    """
    topic_arn = f"arn:aws:sns:{region}:{account_id}:{topic_name}"
    try:
        response = sns_client.delete_topic(TopicArn=topic_arn)
    except ClientError:
        logger.exception(f"Could not delete a SNS topic.")
        raise
    else:
        return response


def generate_prefix_for_agent_infra():
    random_uuid = str(uuid.uuid4())
    prefix_infra = "l2" + random_uuid[0:6]
    prefix_iam = "l2" + random_uuid.split("-")[1]

    # logger.info(f"random_uuid :: {random_uuid} prefix_infra :: {prefix_infra} prefix_iam :: {prefix_iam}")
    return prefix_infra, prefix_iam


def setup_agent_infrastructure(
    schema_filename,
    kb_db_file_uri,
    lambda_code_uri,
    sns_topic_name,
    gt_file_name,
    use_existing_kb,
    existing_kb_id,
):

    # prefix and suffix names
    prefix_infra, prefix_iam = generate_prefix_for_agent_infra()
    suffix = f"{account_id}"  # {region}-

    agent_name = f"{prefix_infra}-reduce-hallucinations-in-genai-apps"
    agent_alias_name = f"{prefix_infra}-workshop-alias"
    bucket_name = f"{agent_name}-{suffix}"
    bucket_arn = f"arn:aws:s3:::{bucket_name}"
    schema_key = f"{agent_name}-schema.json"  # file in repo
    schema_name = schema_filename
    schema_arn = f"arn:aws:s3:::{bucket_name}/{schema_key}"
    bedrock_agent_bedrock_allow_policy_name = f"{prefix_iam}-bedrock-allow-{suffix}"
    bedrock_agent_s3_allow_policy_name = f"{prefix_iam}-s3-allow-{suffix}"
    bedrock_agent_kb_allow_policy_name = f"{prefix_iam}-kb-allow-{suffix}"
    lambda_role_name = "AmazonBedrockLambdaExecutionRoleForAgentsAIAssistant05"  # f'{agent_name}-lambda-role-{suffix}'#'AmazonBedrockLambdaExecutionRoleForAgentsAIAssistant03'
    agent_role_name = "AmazonBedrockExecutionRoleForAgentsAIAssistant05"  # f'AmazonBedrockExecutionRoleForAgents_{prefix_iam}'#      #'
    lambda_code_path = lambda_code_uri  # file in repo
    lambda_name = "LambdaAgentsHallucinationDetection"  # f'{agent_name}-{suffix}'##

    ## KB with DB
    kb_db_tag = "kbdb"

    kb_db_name = f"{prefix_infra}-{kb_db_tag}-{suffix}"
    kb_db_data_source_name = f"{prefix_infra}-{kb_db_tag}-docs-{suffix}"
    kb_db_files_path = kb_db_file_uri  # file path keep as-is
    kb_db_key = f"{kb_db_tag}_{prefix_infra}"
    kb_db_role_name = agent_role_name  # f'AmazonBedrockExecutionRoleForKnowledgeBase_{prefix_infra}_{kb_db_tag}_icakb'   #
    kb_db_bedrock_allow_policy_name = "icaKbdbAgentsBedrockAllow01"  # f"ica-{kb_db_tag}-{prefix_infra}-bedrock-allow-{suffix}"
    kb_db_aoss_allow_policy_name = f"ica-{kb_db_tag}-{prefix_infra}-aoss-allow-{suffix}"
    kb_db_s3_allow_policy_name = f"ica-{kb_db_tag}-{prefix_infra}-s3-allow-{suffix}"
    kb_db_collection_name = f"{prefix_iam}-{kb_db_tag}-{suffix}"
    # Select Amazon titan as the embedding model
    kb_db_embedding_model_arn = (
        f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"
    )
    kb_db_vector_index_name = f"bedrock-knowledge-base-{prefix_infra}-{kb_db_tag}-index"
    kb_db_metadataField = f"bedrock-knowledge-base-{prefix_infra}-{kb_db_tag}-metadata"
    kb_db_textField = f"bedrock-knowledge-base-{prefix_infra}-{kb_db_tag}-text"
    kb_db_vectorField = f"bedrock-knowledge-base-{prefix_infra}-{kb_db_tag}-vector"
    gt_file_name = gt_file_name

    # create sns topic name
    create_topic(sns_topic_name)

    # ## Knowledge Base 1 : DB

    # ### Create S3 bucket and upload API Schema and Knowledge Base files
    #
    # Agents require an API Schema stored on s3. Let's create an S3 bucket to store the file and upload the necessary files to the newly created bucket

    # Create S3 bucket for Open API schema
    logger.info(f"region :: {region} ")
    s3bucket = None
    if region.lower() == "us-east-1":
        s3bucket = s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3bucket = s3_client.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
        )

    # Upload Open API schema and ground truth file to this s3 bucket
    # s3_client.upload_file(schema_name, bucket_name, schema_key)
    s3_client.upload_file(gt_file_name, bucket_name, gt_file_name)

    # Upload Knowledge Base files to this s3 bucket
    # the .pdf file is used by lambda to execute queries, it is NOT used in Knowledge base creation
    for f in os.listdir(kb_db_files_path):
        if f.endswith(".pdf"):
            s3_client.upload_file(
                kb_db_files_path + "/" + f, bucket_name, kb_db_key + "/" + f
            )

    """
    # Create IAM Role for the Lambda function
    try:
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "bedrock:InvokeModel",
                    "Principal": {
                        "Service": "lambda.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }

        assume_role_policy_document_json = json.dumps(assume_role_policy_document)

        lambda_iam_role = iam_client.create_role(
            RoleName=lambda_role_name,
            AssumeRolePolicyDocument=assume_role_policy_document_json
        )

        # Pause to make sure role is created
        time.sleep(20) # nosemgrep: arbitrary-sleep
    except:
        lambda_iam_role = iam_client.get_role(RoleName=lambda_role_name)

    iam_client.attach_role_policy(
        RoleName=lambda_role_name,
        PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
    )
    iam_client.attach_role_policy(
        RoleName=lambda_role_name,
        PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
    )
    """

    agent_bedrock_policy = None
    agent_s3_schema_policy = None
    kb_db_bedrock_policy = None
    kb_aws_bedrock_policy = None
    kb_db_aoss_policy = None
    kb_db_s3_policy = None
    agent_kb_schema_policy = None
    knowledge_base_db_arn = None
    kb_db_opensearch_collection_response = None

    lambda_iam_role = iam_client.get_role(RoleName=lambda_role_name)
    agent_role = iam_client.get_role(RoleName=agent_role_name)

    kb_db_role = iam_client.get_role(RoleName=kb_db_role_name)

    for policy in iam_client.list_policies()["Policies"]:
        # print(policy)
        if bedrock_agent_bedrock_allow_policy_name in policy["PolicyName"]:
            agent_bedrock_policy = policy["Arn"]

        if bedrock_agent_s3_allow_policy_name in policy["PolicyName"]:
            agent_s3_schema_policy = policy["Arn"]

        # ---
        if bedrock_agent_bedrock_allow_policy_name in policy["PolicyName"]:
            kb_db_bedrock_policy = policy["Arn"]

        if bedrock_agent_bedrock_allow_policy_name in policy["PolicyName"]:
            kb_aws_bedrock_policy = policy["Arn"]

        # ---

        if kb_db_s3_allow_policy_name in policy["PolicyName"]:
            kb_db_s3_policy = policy["Arn"]

        if kb_db_aoss_allow_policy_name in policy["PolicyName"]:
            kb_db_aoss_policy = policy["Arn"]

    print(f"agent_bedrock_policy :: {agent_bedrock_policy}")
    print(f"agent_s3_schema_policy :: {agent_s3_schema_policy}")
    print(f"kb_db_bedrock_policy :: {kb_db_bedrock_policy}")
    print(f"kb_aws_bedrock_policy :: {kb_aws_bedrock_policy}")
    print(f"kb_db_s3_policy :: {kb_db_s3_policy}")

    # Delete Lambda function if exists
    for lambdas in lambda_client.list_functions()["Functions"]:
        if lambda_name in lambdas["FunctionName"]:
            lambda_client.delete_function(FunctionName=lambda_name)

    # Package up the lambda function code
    s = BytesIO()
    z = zipfile.ZipFile(s, "w")
    z.write(lambda_code_path)
    z.close()
    zip_content = s.getvalue()

    # Create Lambda Function
    lambda_function = lambda_client.create_function(
        FunctionName=lambda_name,
        Runtime="python3.12",
        Timeout=900,
        MemorySize=1024,
        EphemeralStorage={"Size": 1024},
        Role=lambda_iam_role["Role"]["Arn"],
        Code={"ZipFile": zip_content},
        Handler="lambda_hallucination_detection.lambda_handler",
        Environment={
            "Variables": {
                "BUCKET_NAME": bucket_name,
                "KB_PREFIX": kb_db_key,
                "SNS_TOPIC_NAME": sns_topic_name,
            }
        },
    )

    # subscribe the lambda to the sns topic
    subscribe_to_sns(sns_topic_name)
    if not use_existing_kb or existing_kb_id is None:

        # ### <a name="5">Create Knowledge Base 1 fAsk questions to get answers from the latest Amazon Bedrock User Guide </a>
        # (<a href="#0">Go to top</a>)
        #
        # We will now create the knowledge base used by the agent to gather the outstanding documents requirements. We will use [Amazon OpenSearch Serverless](https://aws.amazon.com/opensearch-service/) as the vector databse and index the files stored on the previously created S3 bucket

        # #### Create Knowledge Base Role
        # Let's first create IAM policies to allow our Knowledge Base to access Bedrock Titan Embedding Foundation model, Amazon OpenSearch Serverless and the S3 bucket with the Knowledge Base Files.
        #
        # Once the policies are ready, we will create the Knowledge Base role

        # Create IAM policies for KB to invoke embedding model

        """
        bedrock_kb_db_allow_fm_model_policy_statement = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AmazonBedrockAgentBedrockFoundationModelPolicy",
                    "Effect": "Allow",
                    "Action": "bedrock:InvokeModel",
                    "Resource": [
                        kb_db_embedding_model_arn
                    ]
                }
            ]
        }

        kb_db_bedrock_policy_json = json.dumps(bedrock_kb_db_allow_fm_model_policy_statement)

        kb_db_bedrock_policy = iam_client.create_policy(
            PolicyName=kb_db_bedrock_allow_policy_name,
            PolicyDocument=kb_db_bedrock_policy_json
        )


        # Create IAM policies for KB to access OpenSearch Serverless
        bedrock_kb_db_allow_aoss_policy_statement = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "aoss:APIAccessAll",
                    "Resource": [
                        f"arn:aws:aoss:{region}:{account_id}:collection/*"
                    ]
                }
            ]
        }


        kb_db_aoss_policy_json = json.dumps(bedrock_kb_db_allow_aoss_policy_statement)

        kb_db_aoss_policy = iam_client.create_policy(
            PolicyName=kb_db_aoss_allow_policy_name,
            PolicyDocument=kb_db_aoss_policy_json
        )


        kb_db_s3_allow_policy_statement = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowKBAccessDocuments",
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}/*",
                        f"arn:aws:s3:::{bucket_name}"
                    ],
                    "Condition": {
                        "StringEquals": {
                            "aws:ResourceAccount": f"{account_id}"
                        }
                    }
                }
            ]
        }


        kb_db_s3_json = json.dumps(kb_db_s3_allow_policy_statement)
        kb_db_s3_policy = iam_client.create_policy(
            PolicyName=kb_db_s3_allow_policy_name,
            PolicyDocument=kb_db_s3_json
        )


        # Create IAM Role for the agent and attach IAM policies
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [{
                  "Effect": "Allow",
                  "Principal": {
                    "Service": "bedrock.amazonaws.com"
                  },
                  "Action": "sts:AssumeRole"
            }]
        }

        assume_role_policy_document_json = json.dumps(assume_role_policy_document)
        kb_db_role = iam_client.create_role(
            RoleName=kb_db_role_name,
            AssumeRolePolicyDocument=assume_role_policy_document_json
        )

        # Pause to make sure role is created
        time.sleep(10) # nosemgrep: arbitrary-sleep

        iam_client.attach_role_policy(
            RoleName=kb_db_role_name,
            PolicyArn=kb_db_bedrock_policy['Policy']['Arn']
        )

        iam_client.attach_role_policy(
            RoleName=kb_db_role_name,
            PolicyArn=kb_db_aoss_policy['Policy']['Arn']
        )

        iam_client.attach_role_policy(
            RoleName=kb_db_role_name,
            PolicyArn=kb_db_s3_policy['Policy']['Arn']
        )
        """
        kb_db_role_arn = kb_db_role["Role"]["Arn"]
        logger.debug(f"kb_db_role_arn :: {kb_db_role_arn} ")

        # #### Create Vector Data Base
        #
        # First of all we have to create a vector store. In this section we will use *Amazon OpenSerach serverless.*
        #
        # Amazon OpenSearch Serverless is a serverless option in Amazon OpenSearch Service (AOSS). As a developer, you can use OpenSearch Serverless to run petabyte-scale workloads without configuring, managing, and scaling OpenSearch clusters. You get the same interactive millisecond response times as OpenSearch Service with the simplicity of a serverless environment. Pay only for what you use by automatically scaling resources to provide the right amount of capacity for your application—without impacting data ingestion.

        # Create OpenSearch Collection
        security_policy_json = {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{kb_db_collection_name}"],
                }
            ],
            "AWSOwnedKey": True,
        }
        security_policy = open_search_serverless_client.create_security_policy(
            description="security policy of aoss collection",
            name=kb_db_collection_name,
            policy=json.dumps(security_policy_json),
            type="encryption",
        )

        kb_db_network_policy_json = [
            {
                "Rules": [
                    {
                        "Resource": [f"collection/{kb_db_collection_name}"],
                        "ResourceType": "dashboard",
                    },
                    {
                        "Resource": [f"collection/{kb_db_collection_name}"],
                        "ResourceType": "collection",
                    },
                ],
                "AllowFromPublic": True,
            }
        ]

        kb_db_network_policy = open_search_serverless_client.create_security_policy(
            description="network policy of aoss collection",
            name=kb_db_collection_name,
            policy=json.dumps(kb_db_network_policy_json),
            type="network",
        )

        response = sts_client.get_caller_identity()
        current_role = response["Arn"]
        logger.debug(f"current_role :: {current_role} ")

        kb_db_data_policy_json = [
            {
                "Rules": [
                    {
                        "Resource": [f"collection/{kb_db_collection_name}"],
                        "Permission": [
                            "aoss:DescribeCollectionItems",
                            "aoss:CreateCollectionItems",
                            "aoss:UpdateCollectionItems",
                            "aoss:DeleteCollectionItems",
                        ],
                        "ResourceType": "collection",
                    },
                    {
                        "Resource": [f"index/{kb_db_collection_name}/*"],
                        "Permission": [
                            "aoss:CreateIndex",
                            "aoss:DeleteIndex",
                            "aoss:UpdateIndex",
                            "aoss:DescribeIndex",
                            "aoss:ReadDocument",
                            "aoss:WriteDocument",
                        ],
                        "ResourceType": "index",
                    },
                ],
                "Principal": [
                    kb_db_role_arn,
                    f"arn:aws:sts::{account_id}:assumed-role/Admin/*",
                    current_role,
                ],
                "Description": "",
            }
        ]

        kb_db_data_policy = open_search_serverless_client.create_access_policy(
            description="data access policy for aoss collection",
            name=kb_db_collection_name,
            policy=json.dumps(kb_db_data_policy_json),
            type="data",
        )

        kb_db_opensearch_collection_response = open_search_serverless_client.create_collection(
            description="OpenSearch collection for Amazon Bedrock Latest User guide Knowledge Base",
            name=kb_db_collection_name,
            standbyReplicas="DISABLED",
            type="VECTORSEARCH",
        )

        logger.debug(
            f"kb_db_opensearch_collection_response :: {kb_db_opensearch_collection_response} "
        )

        kb_db_collection_arn = kb_db_opensearch_collection_response[
            "createCollectionDetail"
        ]["arn"]
        logger.debug(f"kb_db_collection_arn :: {kb_db_collection_arn} ")

        # wait for collection creation
        response = open_search_serverless_client.batch_get_collection(
            names=[kb_db_collection_name]
        )
        # Periodically check collection status
        while (response["collectionDetails"][0]["status"]) == "CREATING":
            print("Creating collection...")
            time.sleep(30)  # nosemgrep: arbitrary-sleep
            response = open_search_serverless_client.batch_get_collection(
                names=[kb_db_collection_name]
            )
        print("\nCollection successfully created:")
        # print(response["collectionDetails"])
        # Extract the collection endpoint from the response
        host = response["collectionDetails"][0]["collectionEndpoint"]
        final_host = host.replace("https://", "")
        logger.debug(f"final_host :: {final_host} ")

        # #### Create OpenSearch Index
        #
        # Let's now create a vector index to index our data

        credentials = boto3.Session().get_credentials()
        service = "aoss"
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            service,
            session_token=credentials.token,
        )

        # Build the OpenSearch client
        open_search_client = OpenSearch(
            hosts=[{"host": final_host, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
        )
        # It can take up to a minute for data access rules to be enforced
        time.sleep(45)  # nosemgrep: arbitrary-sleep
        index_body = {
            "settings": {
                "index.knn": True,
                "number_of_shards": 1,
                "knn.algo_param.ef_search": 512,
                "number_of_replicas": 0,
            },
            "mappings": {"properties": {}},
        }

        index_body["mappings"]["properties"][kb_db_vectorField] = {
            "type": "knn_vector",
            "dimension": 1024,
            "method": {
                "name": "hnsw",
                "engine": "faiss",
                "space_type": "innerproduct",
                "parameters": {"ef_construction": 512, "m": 16},
            },
        }

        index_body["mappings"]["properties"][kb_db_textField] = {"type": "text"}

        index_body["mappings"]["properties"][kb_db_metadataField] = {"type": "text"}

        # Create index
        response = open_search_client.indices.create(
            kb_db_vector_index_name, body=index_body
        )
        print("\nCreating index:")
        logger.info(f"response :: {response} ")

        kb_db_storage_configuration = {
            "opensearchServerlessConfiguration": {
                "collectionArn": kb_db_collection_arn,
                "fieldMapping": {
                    "metadataField": kb_db_metadataField,
                    "textField": kb_db_textField,
                    "vectorField": kb_db_vectorField,
                },
                "vectorIndexName": kb_db_vector_index_name,
            },
            "type": "OPENSEARCH_SERVERLESS",
        }

        # Creating the knowledge base
        try:
            # ensure the index is created and available
            time.sleep(45)  # nosemgrep: arbitrary-sleep
            kb_db_obj = bedrock_agent_client.create_knowledge_base(
                name=kb_db_name,
                description="This Knowledge Base contains information to provide accurate answers for questions asked from the Amazon Bedrock Latest User guide",
                roleArn=kb_db_role_arn,
                knowledgeBaseConfiguration={
                    "type": "VECTOR",  # Corrected type
                    "vectorKnowledgeBaseConfiguration": {
                        "embeddingModelArn": kb_db_embedding_model_arn
                    },
                },
                storageConfiguration=kb_db_storage_configuration,
            )

            # Pretty print the response
            # pprint.pprint(kb_db_obj)

        except Exception as e:
            print(f"Error occurred: {e}")

        # allow time for KB creation and for it to be active
        time.sleep(60)  # nosemgrep: arbitrary-sleep

        # #### Create a data source that you can attach to the recently created Knowledge Base
        #
        # Let's create a data source for our Knowledge Base. Then we will ingest our data and convert it into embeddings.

        # Define the S3 configuration for your data source
        s3_configuration = {"bucketArn": bucket_arn, "inclusionPrefixes": [kb_db_key]}

        # Define the data source configuration
        kb_db_data_source_configuration = {
            "s3Configuration": s3_configuration,
            "type": "S3",
        }

        knowledge_base_db_id = kb_db_obj["knowledgeBase"]["knowledgeBaseId"]
        knowledge_base_db_arn = kb_db_obj["knowledgeBase"]["knowledgeBaseArn"]

        kb_db_chunking_strategy_configuration = {
            "chunkingStrategy": "FIXED_SIZE",
            "fixedSizeChunkingConfiguration": {
                "maxTokens": 1024,
                "overlapPercentage": 50,
            },
        }

        # Create the data source
        try:
            # ensure that the KB is created and available
            time.sleep(45)  # nosemgrep: arbitrary-sleep
            kb_db_data_source_response = bedrock_agent_client.create_data_source(
                knowledgeBaseId=knowledge_base_db_id,
                name=kb_db_data_source_name,
                description="Datasource for the latest Amazon Bedrock User Guide",
                dataSourceConfiguration=kb_db_data_source_configuration,
                vectorIngestionConfiguration={
                    "chunkingConfiguration": kb_db_chunking_strategy_configuration
                },
            )

            # Pretty print the response
            # pprint.pprint(kb_db_data_source_response)

        except Exception as e:
            print(f"Error occurred: {e}")

        # #### Start ingestion job
        # Once the Knowledge Base and Data Source are created, we can start the ingestion job.
        # During the ingestion job, Knowledge Base will fetch the documents in the data source, pre-process it to extract text, chunk it based on the chunking size provided, create embeddings of each chunk and then write it to the vector database, in this case Amazon OpenSource Serverless.

        # Start an ingestion job
        kb_db_data_source_id = kb_db_data_source_response["dataSource"]["dataSourceId"]
        start_job_response = bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=knowledge_base_db_id, dataSourceId=kb_db_data_source_id
        )
        # allow the knowledge base to ready
        time.sleep(60)  # nosemgrep: arbitrary-sleep
        response = bedrock_agent_client.get_knowledge_base(
            knowledgeBaseId=knowledge_base_db_id
        )
        print(f"Knowledge base status :: {response['knowledgeBase']['status']}")

    ## end of using existing KB
    else:
        knowledge_base_db_id = existing_kb_id
        print(f"Using existing_kb_id :: {knowledge_base_db_id}")
        kbid_response = bedrock_agent_client.get_knowledge_base(
            knowledgeBaseId=knowledge_base_db_id
        )
        knowledge_base_db_arn = kbid_response["knowledgeBase"]["knowledgeBaseArn"]

    # Create IAM policies for agent
    """
    bedrock_agent_bedrock_allow_policy_statement = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AmazonBedrockAgentBedrockFoundationModelPolicy",
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": [
                    f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
                ]
            }
        ]
    }

    bedrock_policy_json = json.dumps(bedrock_agent_bedrock_allow_policy_statement)

    agent_bedrock_policy = iam_client.create_policy(
        PolicyName=bedrock_agent_bedrock_allow_policy_name,
        PolicyDocument=bedrock_policy_json
    )


    bedrock_agent_s3_allow_policy_statement = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowAgentAccessOpenAPISchema",
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": [
                    schema_arn
                ]
            }
        ]
    }


    bedrock_agent_s3_json = json.dumps(bedrock_agent_s3_allow_policy_statement)
    agent_s3_schema_policy = iam_client.create_policy(
        PolicyName=bedrock_agent_s3_allow_policy_name,
        Description=f"Policy to allow invoke Lambda that was provisioned for it.",
        PolicyDocument=bedrock_agent_s3_json
    )


    # ### Make sure KB retreival IAM policy includes both DB and AWS arns


    bedrock_agent_kb_retrival_policy_statement = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:Retrieve"
                ],
                "Resource": [
                    knowledge_base_db_arn
                ]
            }
        ]
    }
    bedrock_agent_kb_json = json.dumps(bedrock_agent_kb_retrival_policy_statement)
    agent_kb_schema_policy = iam_client.create_policy(
        PolicyName=bedrock_agent_kb_allow_policy_name,
        Description=f"Policy to allow agent to retrieve documents from knowledge base.",
        PolicyDocument=bedrock_agent_kb_json
    )
    # Create IAM Role for the agent and attach IAM policies
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [{
              "Effect": "Allow",
              "Principal": {
                "Service": "bedrock.amazonaws.com"
              },
              "Action": "sts:AssumeRole"
        }]
    }

    assume_role_policy_document_json = json.dumps(assume_role_policy_document)
    agent_role = iam_client.create_role(
        RoleName=agent_role_name,
        AssumeRolePolicyDocument=assume_role_policy_document_json
    )
    
    
    logger.debug(f"agent_name :: {agent_name}  >>>>>  agent_role :: {agent_role}")

    # Pause to make sure role is created
    time.sleep(10) # allow the knowledge base to ready

    iam_client.attach_role_policy(
        RoleName=agent_role_name,
        PolicyArn=agent_bedrock_policy['Policy']['Arn']
    )

    iam_client.attach_role_policy(
        RoleName=agent_role_name,
        PolicyArn=agent_s3_schema_policy['Policy']['Arn']
    )

    iam_client.attach_role_policy(
        RoleName=agent_role_name,
        PolicyArn=agent_kb_schema_policy['Policy']['Arn']
    )
    """

    logger.info(
        f"agent_name :: {agent_name} \n agent_alias_name :: {agent_alias_name} \n bucket_name :: {bucket_name} \n schema_key :: {schema_key} \n knowledge_base_db_id :: {knowledge_base_db_id} "
    )

    infra_response = {
        "agent_name": agent_name,
        "agent_alias_name": agent_alias_name,
        "agent_role": agent_role,
        "bucket_name": bucket_name,
        "schema_key": schema_key,
        "knowledge_base_db_id": knowledge_base_db_id,
        "lambda_name": lambda_name,
        "lambda_function": lambda_function,
        "agent_bedrock_policy": agent_bedrock_policy,
        "agent_s3_schema_policy": agent_s3_schema_policy,
        "agent_role_name": agent_role_name,
        "lambda_role_name": lambda_role_name,
        "kb_db_collection_name": kb_db_collection_name,
        "agent_kb_schema_policy": agent_kb_schema_policy,
        "kb_db_bedrock_policy": kb_db_bedrock_policy,
        "kb_db_aoss_policy": kb_db_aoss_policy,
        "kb_db_s3_policy": kb_db_s3_policy,
        "kb_db_role_name": kb_db_role_name,
        "kb_db_opensearch_collection_response": kb_db_opensearch_collection_response,
        "sns_topic_name": sns_topic_name,
    }

    return infra_response


def cleanup_infrastructure(
    agent_action_group_response,
    lambda_name,
    lambda_function,
    lambda_role_name,
    agent_id,
    agent_alias_id,
    agent_role_name,
    bucket_name,
    schema_key,
    agent_bedrock_policy,
    agent_s3_schema_policy,
    agent_kb_schema_policy,
    kb_db_bedrock_policy,
    kb_db_aoss_policy,
    kb_db_s3_policy,
    kb_db_role_name,
    kb_db_collection_name,
    kb_db_opensearch_collection_response,
    knowledge_base_db_id,
    sns_topic_name,
):

    # The next steps demonstrate how to delete our agent. To delete the agent we need to:
    # 1. update the action group to disable it
    # 2. delete agent action group
    # 3. delete agent alias
    # 4. delete agent
    # 5. delete lambda function
    # 6. empty created s3 bucket
    # 7. delete s3 bucket

    # This is not needed, you can delete agent successfully after deleting alias only
    # Additionaly, you need to disable it first

    action_group_id = agent_action_group_response["agentActionGroup"]["actionGroupId"]
    action_group_name = agent_action_group_response["agentActionGroup"][
        "actionGroupName"
    ]

    response = bedrock_agent_client.update_agent_action_group(
        agentId=agent_id,
        agentVersion="DRAFT",
        actionGroupId=action_group_id,
        actionGroupName=action_group_name,
        actionGroupExecutor={"lambda": lambda_function["FunctionArn"]},
        functionSchema={"functions": agent_functions},
        actionGroupState="DISABLED",
    )

    action_group_deletion = bedrock_agent_client.delete_agent_action_group(
        agentId=agent_id, agentVersion="DRAFT", actionGroupId=action_group_id
    )

    agent_alias_deletion = bedrock_agent_client.delete_agent_alias(
        agentId=agent_id, agentAliasId=agent_alias_id
    )

    agent_deletion = bedrock_agent_client.delete_agent(agentId=agent_id)

    # Delete Lambda function
    lambda_client.delete_function(FunctionName=lambda_name)

    # delete sns topic
    delete_topic(sns_topic_name)

    # Empty and delete S3 Bucket
    print(f"bucket_name ::: {bucket_name}")

    objects = s3_client.list_objects(Bucket=bucket_name)
    if "Contents" in objects:
        for obj in objects["Contents"]:
            s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])

    time.sleep(10)
    s3_client.delete_bucket(Bucket=bucket_name)

    """

    # Delete IAM Roles and policies

    for policy in [
        agent_bedrock_policy, 
        agent_s3_schema_policy, 
        agent_kb_schema_policy,
        kb_db_bedrock_policy,
        kb_db_aoss_policy,
        kb_db_s3_policy
    ]:
        response = iam_client.list_entities_for_policy(
            PolicyArn=policy['Policy']['Arn'],
            EntityFilter='Role'
        )

        for role in response['PolicyRoles']:
            iam_client.detach_role_policy(
                RoleName=role['RoleName'], 
                PolicyArn=policy['Policy']['Arn']
            )

        iam_client.delete_policy(
            PolicyArn=policy['Policy']['Arn']
        )


    iam_client.detach_role_policy(RoleName=lambda_role_name, PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole')
    iam_client.detach_role_policy(RoleName=lambda_role_name, PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess')

    for role_name in [
        agent_role_name, 
        lambda_role_name, 
        kb_db_role_name
    ]:
        try: 
            iam_client.delete_role(
                RoleName=role_name
            )
        except Exception as e:
            print(e)
            print("couldn't delete role", role_name)

    """
    try:

        open_search_serverless_client.delete_collection(
            id=kb_db_opensearch_collection_response["createCollectionDetail"]["id"]
        )

        open_search_serverless_client.delete_access_policy(
            name=kb_db_collection_name, type="data"
        )

        open_search_serverless_client.delete_security_policy(
            name=kb_db_collection_name, type="network"
        )

        open_search_serverless_client.delete_security_policy(
            name=kb_db_collection_name, type="encryption"
        )

        bedrock_agent_client.delete_knowledge_base(knowledgeBaseId=knowledge_base_db_id)

    except Exception as e:
        print(e)

    print(f"Cleanup completed >>>>>>> ")


def setup_agent_after_create(
    bedrock_agent_client,
    agent_id,
    agent_alias_name,
    lambda_function,
    bucket_name,
    schema_key,
    lambda_name,
    knowledge_base_db_id,
    sns_topic_name,
):
    """
    Create agent action group --
    We will now create an agent action group that uses the Lambda function and API schema files created before. The create_agent_action_group function provides this functionality. We will use DRAFT as the agent version since we haven't yet created an agent version or alias. To inform the agent about the action group functionalities, we will provide an action group description containing the functionalities of the action group.
    """
    # Pause to make sure agent is created
    time.sleep(30)  # allow the knowledge base to ready
    # Now, we can configure and create an action group here:
    agent_action_group_response = bedrock_agent_client.create_agent_action_group(
        agentId=agent_id,
        agentVersion="DRAFT",
        actionGroupExecutor={"lambda": lambda_function["FunctionArn"]},
        actionGroupName="HallucinationDetectionActionGroup",
        functionSchema={"functions": agent_functions},
        description="Actions for executing hallucination detection and next steps based on the generated answers to the user question",
    )

    """
    Allowing agent to invoke action group Lambda
    Before using our action group, we need to allow our agent to invoke the Lambda function associated to the action group. This is done via resource-based policy. Let's add the resource-based policy to the lambda function created:
    """

    # Create allow invoke permission on lambda
    response = lambda_client.add_permission(
        FunctionName=lambda_name,
        StatementId="allow_bedrock",
        Action="lambda:InvokeFunction",
        Principal="bedrock.amazonaws.com",
        SourceArn=f"arn:aws:bedrock:{region}:{account_id}:agent/{agent_id}",
    )

    response = lambda_client.add_permission(
        FunctionName=lambda_name,
        StatementId="allow_sns",
        Action="lambda:InvokeFunction",
        Principal="sns.amazonaws.com",
        SourceArn=f"arn:aws:sns:{region}:{account_id}:{sns_topic_name}",
    )

    """
    Associating the agent to a knowledge base
    """
    agent_kb_description = bedrock_agent_client.associate_agent_knowledge_base(
        agentId=agent_id,
        agentVersion="DRAFT",
        description=f"Use the information in the {knowledge_base_db_id} Knowledge Base to answer the questions based on the latest Amazon Bedrock User Guide",
        knowledgeBaseId=knowledge_base_db_id,
    )

    """
    Preparing agent
    Let's create a DRAFT version of the agent that can be used for internal testing.
    """

    agent_prepare = bedrock_agent_client.prepare_agent(agentId=agent_id)

    """
    Create agent alias to deploy agent
    We will now create an alias of the agent that can be used to deploy the agent.
    """

    # Pause to make sure agent is prepared
    time.sleep(60)  # nosemgrep: arbitrary-sleep
    agent_alias = bedrock_agent_client.create_agent_alias(
        agentId=agent_id, agentAliasName=agent_alias_name
    )
    # Pause to make sure agent alias is ready
    time.sleep(60)  # nosemgrep: arbitrary-sleep

    # printing the alias information
    agent_alias_name = agent_alias["agentAlias"]["agentAliasName"]
    agent_alias_id = agent_alias["agentAlias"]["agentAliasId"]
    # print(f"agent_alias_name :: {agent_alias_name} and agent_alias_id :: {agent_alias_id}")

    return agent_alias, agent_action_group_response
