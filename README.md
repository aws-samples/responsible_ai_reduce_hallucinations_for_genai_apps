## Responsible AI: Detecting and Reducing LLM Hallucination with Human in the Loop for Generative AI Applications

This repository contains notebooks for the AWS Responsible AI __Veracity__ dimension and partnered with __Amazon Machine Learning University__ . Our mission is to make Machine Learning accessible to everyone. We have courses available across many topics of machine learning and believe knowledge of ML can be a key enabler for success. These notebooks represent some real-world use cases to detect and reduce LLM hallucinations with Amazon Bedrock Agents, Amazon Knowledge Bases and RAGAS evaluation framework.

---

__Project : Application Builder Assistant using Bedrock Agents and multiple knowledge bases__

| Title | Studio lab |
| :---: | ---: |
| Chat with LLMs grounded by Amazon Guardrails and querying Amazon Knowledge Bases | lab1/lab1.ipynb|
| Detecting and mitigating hallucinations with Amazon Bedrock Agents and Amazon Knowledge Bases | lab2/lab2.ipynb|
---

__Setup Instructions__

The lab1 notebook will download the bedrock user guide from here:
https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf 

Please execute `lab1.ipynb` before proceeding to `lab2.ipynb`.

Note: Using the CloudFormation template and running the noteobooks end-to-end may create AWS service roles and AWS Managed KMS keys that will not incur cost in your account.

---
__Troubleshooting: CFN template__

Please add the following independent role in `SageMaker_Bedrock_Agents.yaml` if you get the following error:

`SageMaker is not authorized to perform: iam:CreateServiceLinkedRole on resource: arn:aws:iam::<account-id>:role/aws-service-role/observability.aoss.amazonaws.com/AWSServiceRoleForAmazonOpenSearchServerless because no identity-based policy allows the iam:CreateServiceLinkedRole action`


```
AOSSLinkedRole:
    Type: AWS::IAM::ServiceLinkedRole
    Properties:
      AWSServiceName: observability.aoss.amazonaws.com
```

---

## License

The license for this repository depends on the section.  Data set for the course is being provided to you by permission of Amazon and is subject to the terms of the [Amazon License and Access](https://www.amazon.com/gp/help/customer/display.html?nodeId=201909000). You are expressly prohibited from copying, modifying, selling, exporting or using this data set in any way other than for the purpose of completing this course. The lecture slides are released under the CC-BY-SA-4.0 License.  This project is licensed under the Apache-2.0 License. See each section's LICENSE file for details.
