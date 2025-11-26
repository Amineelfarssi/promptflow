"""
Example: Using PromptFlow with Amazon Bedrock

This example demonstrates how to:
1. Set up prompts for an AI assistant
2. Use the Bedrock integration to call Claude
3. Manage prompt versions and aliases
"""

import boto3
from promptflow import PromptRegistry, PromptMetadata, TemplateFormat
from promptflow.integrations.bedrock import BedrockPromptHelper


def setup_prompts():
    """Set up example prompts in the registry."""
    
    # Use local storage for this example
    # In production, use: PromptRegistry.from_s3(bucket="your-bucket")
    registry = PromptRegistry()
    
    # Register a system prompt for the assistant
    registry.register(
        name="assistant-system",
        template="""You are a helpful AI assistant specialized in {domain}.

Your key responsibilities:
- Provide accurate, well-researched information
- Be concise but thorough
- Cite sources when possible
- Admit when you're uncertain

Communication style: {style}""",
        project="assistant",
        description="System prompt for the AI assistant",
        tags=["system", "production"],
        format=TemplateFormat.FSTRING,
        metadata={
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "temperature": 0.7,
            "max_tokens": 2000,
        },
    )
    
    # Register a summarization prompt
    registry.register(
        name="summarizer",
        template="""Please summarize the following {content_type} in a {style} manner.

Focus on:
{focus_areas}

Content to summarize:
---
{content}
---

Provide your summary below:""",
        project="assistant",
        description="Flexible summarization prompt",
        tags=["summarization", "utility"],
        metadata={
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "temperature": 0.3,  # Lower for more consistent summaries
            "max_tokens": 1000,
        },
    )
    
    # Register a code review prompt
    registry.register(
        name="code-reviewer",
        template="""Review the following {language} code for:
- Code quality and best practices
- Potential bugs or issues
- Performance considerations
- Security vulnerabilities

Code to review:
```{language}
{code}
```

Provide your detailed review with specific line references where applicable.""",
        project="assistant",
        description="Code review prompt",
        tags=["code", "review", "development"],
        metadata={
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    )
    
    # Set initial aliases
    registry.set_alias("assistant-system", "prod", 1)
    registry.set_alias("summarizer", "prod", 1)
    registry.set_alias("code-reviewer", "prod", 1)
    
    print("✓ Prompts registered successfully!")
    return registry


def example_summarization(registry: PromptRegistry):
    """Example: Summarize a document using Bedrock."""
    
    print("\n--- Summarization Example ---")
    
    # Initialize Bedrock client
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    helper = BedrockPromptHelper(registry=registry)
    
    # Sample document to summarize
    document = """
    Artificial Intelligence (AI) has transformed numerous industries in recent years.
    From healthcare to finance, AI applications are improving efficiency and enabling
    new capabilities. Machine learning, a subset of AI, allows systems to learn from
    data and improve over time without explicit programming.
    
    Key developments include:
    - Natural Language Processing (NLP) for understanding human language
    - Computer Vision for image and video analysis
    - Reinforcement Learning for decision-making systems
    
    However, challenges remain, including ethical considerations, bias in training data,
    and the need for explainable AI systems.
    """
    
    # Build the request using the helper
    request = helper.build_converse_request(
        name="summarizer",
        project="assistant",
        alias="prod",
        content_type="article",
        style="concise",
        focus_areas="- Main themes\n- Key developments\n- Challenges mentioned",
        content=document.strip(),
    )
    
    print(f"Using model: {request['modelId']}")
    print(f"Inference config: {request.get('inferenceConfig', {})}")
    
    # Make the API call
    try:
        response = bedrock.converse(**request)
        summary = response["output"]["message"]["content"][0]["text"]
        print(f"\nSummary:\n{summary}")
    except Exception as e:
        print(f"Note: Bedrock call skipped (requires AWS credentials): {e}")
        # Show what would be sent
        print(f"\nRendered prompt:\n{request['messages'][0]['content'][0]['text'][:500]}...")


def example_version_management(registry: PromptRegistry):
    """Example: Managing prompt versions."""
    
    print("\n--- Version Management Example ---")
    
    # Update the summarizer prompt
    print("\nUpdating summarizer prompt...")
    registry.update(
        name="summarizer",
        project="assistant",
        template="""Summarize the following {content_type}.

Style: {style}
Length: {length}

Key areas to cover:
{focus_areas}

---
{content}
---

Summary:""",
        change_message="Added length parameter, simplified format",
    )
    
    # View history
    print("\nVersion history:")
    for version in registry.history("summarizer", project="assistant"):
        print(f"  v{version.version}: {version.change_message or 'No message'}")
        print(f"    Created: {version.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"    Variables: {version.variables}")
    
    # Set staging alias to new version
    registry.set_alias("summarizer", "staging", version=2)
    print("\n✓ Set 'staging' alias to version 2")
    
    # Compare versions
    print("\nComparing v1 vs v2:")
    diff = registry.compare("summarizer", 1, 2, project="assistant")
    print(f"  Template changed: {diff['template_changed']}")
    print(f"  Variables added: {diff['variables_added']}")
    print(f"  Variables removed: {diff['variables_removed']}")
    
    # Promote to production
    print("\nPromoting staging to prod...")
    registry.promote("summarizer", from_alias="staging", to_alias="prod", project="assistant")
    print("✓ Version 2 is now in production")


def example_search_and_discover(registry: PromptRegistry):
    """Example: Searching and discovering prompts."""
    
    print("\n--- Search & Discovery Example ---")
    
    # List all prompts in the assistant project
    print("\nPrompts in 'assistant' project:")
    for prompt in registry.list(project="assistant"):
        print(f"  - {prompt.name}")
        print(f"    Description: {prompt.description}")
        print(f"    Tags: {prompt.tags}")
        print(f"    Aliases: {list(prompt.aliases.keys())}")
    
    # Search for code-related prompts
    print("\nSearching for 'code' prompts:")
    results = registry.search("code")
    for prompt in results:
        print(f"  - {prompt.name}: {prompt.description}")
    
    # List all projects
    print(f"\nAll projects: {registry.list_projects()}")


def example_export_import(registry: PromptRegistry):
    """Example: Export and import prompts."""
    
    print("\n--- Export/Import Example ---")
    
    # Export the assistant project
    json_export = registry.export_json("assistant")
    print(f"Exported {len(json_export)} characters of JSON")
    
    # Save to file
    with open("assistant_prompts_backup.json", "w") as f:
        f.write(json_export)
    print("✓ Saved to assistant_prompts_backup.json")
    
    # In another environment, you would import like this:
    # new_registry = PromptRegistry.from_s3(bucket="other-bucket")
    # stats = new_registry.import_json(json_export)
    # print(f"Imported {stats['imported']} prompts")


def main():
    """Run all examples."""
    print("=" * 60)
    print("PromptFlow + Amazon Bedrock Example")
    print("=" * 60)
    
    # Set up prompts
    registry = setup_prompts()
    
    # Run examples
    example_summarization(registry)
    example_version_management(registry)
    example_search_and_discover(registry)
    example_export_import(registry)
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
