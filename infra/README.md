# Cloud Infrastructure for the Workshop

This folder contains [Infrastructure-as-Code](https://aws.amazon.com/what-is/iac/) templates to automate deployment of Cloud resources used for the sample(s), so users can focus on the key concepts being explored without getting too distracted by prerequisite setup.


## Deployable Components

### SageMaker Studio Notebook Environment

[SageMaker-Notebook.yaml](./SageMaker-Notebook.yaml) is a [CloudFormation-deployable](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html) template for a [SageMaker AI Studio environment](https://aws.amazon.com/sagemaker/ai/studio/) that you can use to work through the provided sample notebooks.

If you're attending an AWS-led workshop that provides temporary AWS accounts, this notebook environment will be pre-deployed for you.


### Example Bedrock Knowledge Base

[Bedrock-Knowledge-Base.yaml](./Bedrock-Knowledge-Base.yaml) is a CloudFormation template to automatically set up an example [Amazon Bedrock Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) backed by [Amazon OpenSearchServerless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-overview.html) vector search, and pre-load document(s) to it from Amazon S3.

This Knowledge Base is not necessary for the main workshop, but you'll be asked to deploy it for some of the `other-examples`.
