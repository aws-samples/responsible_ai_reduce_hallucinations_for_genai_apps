# Detecting and reducing hallucinations in generative AI applications

This repository demonstrates a range of methods for detecting and intervening to avoid "hallucinations" in LLM-powered applications, including AI Agents with [Strands](https://strandsagents.com/latest/).

The main flow accompanies the **[guided workshop instructions HERE](https://catalog.workshops.aws/workshops/1fa309f2-c771-42d5-87bc-e8f919e7bcc9/en-US)** - where you can find more information about prerequisites and environment setup, the concepts discussed, and a step-by-step walkthrough of the labs.

Some additional samples are also provided in the [other-examples](./other-examples) folder for reference.


## Project Structure

The examples are generally presented via interactive Python notebooks, and the repository contains some other supporting utility code:

```sh
/
├── .env.example           # Template for configuring environment variables
├── .env                   # Your credentials (git-ignored, and sensitive!)
├── hallucination_utils/   # Utility code to help simplify the notebooks
│   ├── bedrock_guardrails/  # Strands + Amazon Bedrock Guardrails integration
│   ├── strands_models/      # Custom Strands model wrappers
│   │   └── parallel.py      # Run multiple parallel models and combine results
│   │   └── with_checks.py   # Integrate custom, model-level output checks
│   ├── types/               # Type definitions to help clarify interfaces
│   ├── mcp.py               # Simplify connecting to AWS credential-using MCPs
│   └── tracing.py           # Connect Strands to Langfuse via OpenTelemetry
├── infra/                 # Cloud infrastructure templates for deployment
│
├── lab*-*.ipynb           # Main Python notebooks for the hands-on exercises
├── ...
├── other-examples/        # Extra examples, techniques and resources
└── pyproject.toml         # Python project dependencies for installation
```

### About the examples

The main suggested flow of exploration is:

- **Lab 0: SageMaker Model Deployment** ([lab0-deploy-sagemaker-ai-endpoint.ipynb](lab0-deploy-sagemaker-ai-endpoint.ipynb))
    - Deploy a self-hosted LLM on Amazon SageMaker AI, for use in later examples
- **Lab 1: Contextual Grounding** ([lab1-contextual-grounding.ipynb](lab1-contextual-grounding.ipynb))
    - Explore how RAG and similar contextual grounding patterns help reduce hallucinations in agents; their limitations; and how [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html) can add further protection with fully-managed API.
- **Lab 2: Response-Level Detection** ([lab2-response-level-detection.ipynb](lab2-response-level-detection.ipynb))
    - Use three hallucination detection methods that *only* depend on the LLM output (no reference data or internal model activations): *Semantic Similarity Analysis*, *Non-Contradiction Probability*, and *Normalized Semantic Negentropy*.
    - 🛑 Note: You'll need to complete lab 0 first, before lab 2
- **Lab 3: Token Probability-Level Detection** ([lab3-token-probability-level-detection.ipynb](lab3-token-probability-level-detection.ipynb))
    - Try more advanced hallucination detection methods that depend on visibility of the per-token 'logprobs' output scores (which are not exposed by some proprietary model providers, but can be more efficient than response-level detection where available)
    - 🛑 Note: You'll need to complete lab 0 first, before lab 3

If you have extra time, you could also explore the **other-examples/**:

- [bedrock-knowledge-base-guardrails](other-examples/bedrock-knowledge-base-guardrails/)
    - Deploy a private RAG pipeline with [Amazon Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html), and add extra protections with [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html)
    - In contrast to main lab 1, this example 1/ uses a private knowledge base instead of the public AWS Knowledge MCP, and 2/ focuses on the core APIs instead of integrating with Strands Agents SDK
- [bedrock-agent-self-reflection](other-examples/bedrock-agent-self-reflection/)
    - Demonstrate a more specialized pattern as discussed in [this AWS ML Blog post](https://aws.amazon.com/blogs/machine-learning/reducing-hallucinations-in-large-language-models-with-custom-intervention-using-amazon-bedrock-agents/): where the model itself decides when to run the hallucination check.
    - In contrast to the main labs, this example uses [Amazon Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-how.html) instead of the Open Source Strands Agents SDK.


## Troubleshooting and support

If you're at a guided event, please do reach out to your friendly facilitators for help! 

We've also provided some tips below for troubleshooting. If you're struggling with questions or issues, you can also open an issue on this repository.


### Import Errors

If you see a `ModuleNotFoundError`:
1. Check your Python version: `python --version` (must be 3.10+)
2. Ensure you're using the correct kernel in Jupyter
3. Reinstall dependencies: `pip install -e .` or `uv sync`
4. Restart the Jupyter kernel


### Traces not showing in Langfuse

If you've configured Langfuse but your traces aren't appearing after running the agents:
1. Try restarting your notebook kernel and trying again
    - OpenTelemetry can only be configured **once** per kernel session, and IDEs like VSCode may only read your `.env` file when the kernel is initially started
2. Try re-entering your credentials by running `set_up_notebook_langfuse(refresh=True)`
3. Verify your `.env` file exists and has the correct credentials
    - Note that files starting with a dot "." are hidden in JupyterLab folder explorer by default, but you could access, move, or copy them via the terminal.
4. Verify your API keys in the Langfuse UI, and consider creating+configuring a new key pair.


### AWS Bedrock Access

Ensure your AWS credentials are configured:

```bash
aws configure
```

And that your account [has access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) to the required Bedrock models.


## Further reading and resources

- [Strands Agents Documentation](https://github.com/strands-agents/sdk-python)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Sentence Transformers](https://www.sbert.net/)


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.


## License

Different sections of this repository are provided under different licenses.

CSV Datasets under the `other-examples` folder are provided to you by permission of Amazon and are subject to the terms of the [Amazon License and Access](https://www.amazon.com/gp/help/customer/display.html?nodeId=201909000). You are expressly prohibited from copying, modifying, selling, exporting or using these data sets in any way other than for the purpose of completing the course.

Otherwise, this the library is licensed under the Apache-2.0 License. See the [LICENSE](./LICENSE) file for details.
