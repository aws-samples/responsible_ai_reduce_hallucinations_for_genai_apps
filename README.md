## Responsible AI: Detecting and Reducing LLM Hallucination with Human in the Loop for Generative AI Applications

These notebooks represent some real-world use cases to detect and reduce LLM hallucinations with [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/), [Amazon Bedrock Knowledge Bases](https://aws.amazon.com/bedrock/knowledge-bases/) and [RAGAS](https://docs.ragas.io/en/stable/) evaluation framework.

---

__Project : Hallucination Detection and Mitigation__

| Title | Studio lab |
| :---: | ---: |
| Chat with LLMs grounded by Amazon Guardrails and querying Amazon Knowledge Bases | lab1/lab1.ipynb|
| Detecting and mitigating hallucinations with Amazon Bedrock Agents and Amazon Knowledge Bases | lab2/lab2.ipynb|
---

__Setup Instructions__

The lab1 notebook will download the bedrock user guide from here: [AWS docs for Amazon Bedrock User Guide](https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf )

Please execute `lab1.ipynb` before proceeding to `lab2.ipynb`.

Note: Using the CloudFormation template and running the noteobooks end-to-end may create AWS service roles and AWS Managed KMS keys that will not incur cost in your account. If you do not run the optional cleanup infrastructure cell for every notebook run, there may be S3 buckets, SNS left behind in the account which would need manual cleanup.

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
