import json
import random
import time

import boto3

suffix = random.randrange(200, 900)
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
iam_client = boto3_session.client("iam")
account_number = boto3.client("sts").get_caller_identity().get("Account")
identity = boto3.client("sts").get_caller_identity()["Arn"]
