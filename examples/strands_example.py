"""
Example: Using PromptFlow with Strands Agents SDK

This example demonstrates how to:
1. Create tools for prompt retrieval
2. Build an agent with managed prompts
3. Use the PromptFlowToolkit for advanced scenarios
"""

from promptflow import PromptRegistry
from promptflow.integrations.strands import (
    PromptFlowToolkit,
    get_prompt_tools,
    set_registry,
)


def setup_prompts():
    """Set up example prompts for the agent."""
    
    registry = PromptRegistry()
    
    # Agent system prompt
    registry.register(
        name="research-agent-system",
        template="""You are a research assistant agent with access to a prompt registry.

Your capabilities:
- Search for and retrieve managed prompts
- Render prompts with user-provided variables
- Help users discover available prompts

When asked to perform a task:
1. Search for relevant prompts in the registry
2. Select the most appropriate prompt
3. Render it with the required variables
4. Use the rendered prompt to complete the task

Be helpful and explain which prompts you're using and why.""",
        project="agents",
        description="System prompt for the research agent",
        tags=["agent", "system"],
    )
    
    # Various utility prompts the agent can use
    registry.register(
        name="summarizer",
        template="Summarize the following {content_type} in {length} words or less:\n\n{text}",
        project="agents",
        description="Flexible summarization prompt",
        tags=["utility", "summarization"],
    )
    
    registry.register(
        name="translator",
        template="Translate the following text to {target_language}:\n\n{text}",
        project="agents",
        description="Translation prompt",
        tags=["utility", "translation"],
    )
    
    registry.register(
        name="analyzer",
        template="""Analyze the following {subject} and provide:
1. Key themes
2. Main arguments
3. Potential issues or concerns
4. Recommendations

{subject_type}:
---
{content}
---""",
        project="agents",
        description="Content analysis prompt",
        tags=["utility", "analysis"],
    )
    
    registry.register(
        name="question-generator",
        template="""Based on the following {topic}, generate {count} thoughtful questions that would:
- Test understanding
- Encourage deeper thinking
- Cover different aspects of the topic

Topic:
{content}

Generate questions:""",
        project="agents",
        description="Generate questions about a topic",
        tags=["utility", "education"],
    )
    
    # Set aliases
    registry.set_alias("research-agent-system", "prod", 1)
    registry.set_alias("summarizer", "prod", 1)
    registry.set_alias("translator", "prod", 1)
    registry.set_alias("analyzer", "prod", 1)
    
    print("✓ Agent prompts registered!")
    return registry


def example_basic_tools():
    """Example: Using basic prompt tools with an agent."""
    
    print("\n--- Basic Tools Example ---")
    print("This shows how to use prompt retrieval tools with Strands.")
    
    registry = setup_prompts()
    set_registry(registry)
    
    # Get the tools
    tools = get_prompt_tools()
    print(f"\nAvailable tools: {[t.__name__ for t in tools]}")
    
    # Simulate what an agent would do
    from promptflow.integrations.strands import get_prompt, render_prompt, list_prompts, search_prompts
    
    # Agent discovers available prompts
    print("\n1. Agent lists available prompts:")
    result = list_prompts(project="agents", limit=10)
    for p in result["prompts"]:
        print(f"   - {p['name']}: {p['description']}")
    
    # Agent searches for summarization
    print("\n2. Agent searches for 'summarize':")
    search_result = search_prompts("summarize")
    print(f"   Found: {[r['name'] for r in search_result['results']]}")
    
    # Agent gets the prompt details
    print("\n3. Agent retrieves summarizer prompt:")
    prompt_info = get_prompt("summarizer", project="agents")
    print(f"   Template: {prompt_info['template'][:50]}...")
    print(f"   Variables needed: {prompt_info['variables']}")
    
    # Agent renders the prompt
    print("\n4. Agent renders the prompt:")
    rendered = render_prompt(
        "summarizer",
        {
            "content_type": "article",
            "length": "100",
            "text": "Artificial intelligence is transforming industries..."
        },
        project="agents"
    )
    print(f"   Rendered:\n   {rendered['rendered'][:200]}...")
    
    print("\n" + "-" * 40)
    print("In a real Strands agent, you would do:")
    print("""
    from strands import Agent
    from promptflow.integrations.strands import get_prompt_tools
    
    agent = Agent(
        system_prompt="You help users with managed prompts",
        tools=get_prompt_tools()
    )
    
    response = agent("Summarize this article using the summarizer prompt...")
    """)


def example_toolkit():
    """Example: Using the PromptFlowToolkit for more control."""
    
    print("\n--- Toolkit Example ---")
    
    registry = setup_prompts()
    toolkit = PromptFlowToolkit(registry=registry, default_project="agents")
    
    # Get system prompt for agent initialization
    system_prompt = toolkit.get_system_prompt(
        "research-agent-system",
        alias="prod"
    )
    print(f"System prompt ({len(system_prompt)} chars):")
    print(f"  {system_prompt[:200]}...")
    
    # Get model configuration
    config = toolkit.get_model_config("summarizer", alias="prod")
    print(f"\nModel config from metadata: {config}")
    
    # Render a prompt directly
    rendered = toolkit.render(
        "analyzer",
        subject="code",
        subject_type="Python function",
        content="def hello(): print('world')"
    )
    print(f"\nRendered analyzer prompt:\n{rendered[:200]}...")
    
    print("\n" + "-" * 40)
    print("Full agent setup with toolkit:")
    print("""
    from strands import Agent
    from promptflow.integrations.strands import PromptFlowToolkit
    
    toolkit = PromptFlowToolkit()
    
    agent = Agent(
        system_prompt=toolkit.get_system_prompt("my-agent-system", alias="prod"),
        tools=toolkit.get_tools()
    )
    """)


def example_agent_workflow():
    """Example: Complete agent workflow simulation."""
    
    print("\n--- Agent Workflow Simulation ---")
    print("Simulating what a Strands agent would do with prompts.\n")
    
    registry = setup_prompts()
    toolkit = PromptFlowToolkit(registry=registry, default_project="agents")
    
    # Simulate user request
    user_request = "I have an article about machine learning. Can you summarize it and generate some discussion questions?"
    print(f"User: {user_request}\n")
    
    # Agent thinks...
    print("Agent thought process:")
    print("1. User wants summarization and question generation")
    print("2. Searching for relevant prompts...")
    
    from promptflow.integrations.strands import search_prompts, set_registry
    set_registry(registry)
    
    # Search for prompts
    summary_prompts = search_prompts("summarize", project="agents")
    question_prompts = search_prompts("question", project="agents")
    
    print(f"   - Found summarizer: {summary_prompts['results'][0]['name']}")
    print(f"   - Found question generator: {question_prompts['results'][0]['name']}")
    
    # Sample article
    article = """
    Machine learning has revolutionized data analysis by enabling computers
    to learn patterns from data without explicit programming. Key techniques
    include supervised learning for labeled data, unsupervised learning for
    pattern discovery, and reinforcement learning for sequential decision making.
    Recent advances in deep learning have enabled breakthrough applications
    in computer vision, natural language processing, and game playing.
    """
    
    # Generate summary
    print("\n3. Rendering summarizer prompt...")
    summary_prompt = toolkit.render(
        "summarizer",
        content_type="article",
        length="50",
        text=article.strip()
    )
    print(f"   Summary prompt ready ({len(summary_prompt)} chars)")
    
    # Generate questions
    print("\n4. Rendering question generator prompt...")
    questions_prompt = toolkit.render(
        "question-generator",
        topic="machine learning",
        count="3",
        content=article.strip()
    )
    print(f"   Questions prompt ready ({len(questions_prompt)} chars)")
    
    print("\n5. Agent would now call Bedrock with these prompts...")
    print("   (In production, the agent automatically handles LLM calls)")
    
    print("\n" + "=" * 40)
    print("FINAL OUTPUT would include:")
    print("- A concise summary of the article")
    print("- 3 discussion questions about machine learning")


def main():
    """Run all Strands examples."""
    print("=" * 60)
    print("PromptFlow + Strands Agents SDK Examples")
    print("=" * 60)
    
    example_basic_tools()
    example_toolkit()
    example_agent_workflow()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo use with a real Strands agent, ensure you have:")
    print("1. pip install strands-agents")
    print("2. AWS credentials configured for Bedrock")
    print("3. Model access enabled in Bedrock console")
    print("=" * 60)


if __name__ == "__main__":
    main()
