"""
Strands Agents SDK integration for PromptFlow.

Provides tools and utilities to use PromptFlow-managed prompts
with Strands Agents. Includes a @tool-compatible prompt retrieval
function that agents can use to dynamically fetch prompts.

Example:
    from strands import Agent
    from promptflow.integrations.strands import get_prompt_tools, PromptFlowToolkit
    
    # Option 1: Use pre-built tools directly
    agent = Agent(
        system_prompt="You are a helpful assistant that uses managed prompts.",
        tools=get_prompt_tools()
    )
    
    # Option 2: Use the toolkit for more control
    toolkit = PromptFlowToolkit()
    agent = Agent(
        system_prompt=toolkit.get_system_prompt("agent-system", alias="prod"),
        tools=toolkit.get_tools()
    )
    
    response = agent("Summarize the following document using the summarizer prompt")
"""

from __future__ import annotations

import json
from typing import Any, Callable

from promptflow.core.models import PromptVersion
from promptflow.core.registry import PromptRegistry


# Global registry instance for tools
_registry: PromptRegistry | None = None


def _get_registry() -> PromptRegistry:
    """Get or create the global registry instance."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry.from_env()
    return _registry


def set_registry(registry: PromptRegistry) -> None:
    """
    Set a custom registry for the Strands tools.
    
    Call this before creating agents to use a custom registry configuration.
    
    Args:
        registry: PromptFlow registry instance
    """
    global _registry
    _registry = registry


# =============================================================================
# Strands @tool compatible functions
# =============================================================================

def get_prompt(
    name: str,
    project: str = "default",
    version: int | None = None,
    alias: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve a prompt template from the PromptFlow registry.
    
    Use this tool to fetch managed prompts by name. You can specify
    a particular version or use an alias like "prod" or "staging".
    
    Args:
        name: The name of the prompt to retrieve
        project: Project name (default: "default")
        version: Specific version number (optional)
        alias: Alias like "prod" or "staging" (optional)
    
    Returns:
        A dict containing the prompt template, variables, and metadata
    """
    registry = _get_registry()
    prompt_version = registry.get(name, project, version, alias)
    
    if not prompt_version:
        return {
            "error": f"Prompt '{name}' not found in project '{project}'",
            "found": False,
        }
    
    return {
        "found": True,
        "name": name,
        "version": prompt_version.version,
        "template": prompt_version.template,
        "variables": prompt_version.variables,
        "format": prompt_version.format.value,
        "metadata": {
            "model": prompt_version.metadata.model,
            "temperature": prompt_version.metadata.temperature,
            "max_tokens": prompt_version.metadata.max_tokens,
        },
    }


def render_prompt(
    name: str,
    variables: dict[str, Any],
    project: str = "default",
    version: int | None = None,
    alias: str | None = None,
) -> dict[str, Any]:
    """
    Render a prompt template with the provided variables.
    
    Use this tool to get a fully rendered prompt ready for use.
    
    Args:
        name: The name of the prompt to render
        variables: Dictionary of variable values to fill in the template
        project: Project name (default: "default")
        version: Specific version number (optional)
        alias: Alias like "prod" or "staging" (optional)
    
    Returns:
        A dict containing the rendered prompt text
    """
    registry = _get_registry()
    
    try:
        rendered = registry.render(name, project, version, alias, **variables)
        return {
            "success": True,
            "rendered": rendered,
            "name": name,
            "variables_used": list(variables.keys()),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def list_prompts(
    project: str | None = None,
    tags: list[str] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    List available prompts in the registry.
    
    Use this tool to discover what prompts are available.
    
    Args:
        project: Filter by project name (optional)
        tags: Filter by tags (optional)
        limit: Maximum number of prompts to return
    
    Returns:
        A dict containing list of prompt summaries
    """
    registry = _get_registry()
    prompts = registry.list(project=project, tags=tags, limit=limit)
    
    return {
        "count": len(prompts),
        "prompts": [
            {
                "name": p.name,
                "project": p.project,
                "description": p.description,
                "tags": p.tags,
                "latest_version": p.latest_version,
                "aliases": list(p.aliases.keys()),
            }
            for p in prompts
        ],
    }


def search_prompts(
    query: str,
    project: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Search for prompts by text.
    
    Use this tool to find prompts matching a search query.
    
    Args:
        query: Search text to look for in names, descriptions, and content
        project: Filter by project name (optional)
        limit: Maximum number of results
    
    Returns:
        A dict containing matching prompts
    """
    registry = _get_registry()
    prompts = registry.search(query, project, limit)
    
    return {
        "query": query,
        "count": len(prompts),
        "results": [
            {
                "name": p.name,
                "project": p.project,
                "description": p.description,
                "tags": p.tags,
            }
            for p in prompts
        ],
    }


def get_prompt_tools() -> list[Callable]:
    """
    Get all PromptFlow tools for use with Strands agents.
    
    Returns:
        List of tool functions ready for Strands Agent
    
    Example:
        from strands import Agent
        from promptflow.integrations.strands import get_prompt_tools
        
        agent = Agent(tools=get_prompt_tools())
    """
    return [
        get_prompt,
        render_prompt,
        list_prompts,
        search_prompts,
    ]


# =============================================================================
# PromptFlowToolkit - Higher-level integration
# =============================================================================

class PromptFlowToolkit:
    """
    A toolkit for integrating PromptFlow with Strands Agents.
    
    Provides both tools for agents to use and helper methods for
    setting up agent configurations with managed prompts.
    
    Example:
        from strands import Agent
        from promptflow.integrations.strands import PromptFlowToolkit
        
        toolkit = PromptFlowToolkit()
        
        # Use managed system prompt
        agent = Agent(
            system_prompt=toolkit.get_system_prompt("my-agent-system", alias="prod"),
            tools=toolkit.get_tools()
        )
    """
    
    def __init__(
        self,
        registry: PromptRegistry | None = None,
        default_project: str = "default",
    ):
        """
        Initialize the toolkit.
        
        Args:
            registry: Custom registry instance
            default_project: Default project for prompt lookups
        """
        self.registry = registry or PromptRegistry.from_env()
        self.default_project = default_project
        
        # Set global registry for tools
        set_registry(self.registry)
    
    def get_tools(self) -> list[Callable]:
        """
        Get all PromptFlow tools for this toolkit's registry.
        
        Returns:
            List of tool functions
        """
        return get_prompt_tools()
    
    def get_system_prompt(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        **variables: Any,
    ) -> str:
        """
        Get a rendered system prompt for agent initialization.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use
            **variables: Template variables
        
        Returns:
            Rendered system prompt string
        """
        return self.registry.render(
            name,
            project or self.default_project,
            version,
            alias,
            **variables,
        )
    
    def get_prompt(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
    ) -> PromptVersion | None:
        """
        Get a prompt version directly.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use
        
        Returns:
            PromptVersion or None
        """
        return self.registry.get(
            name,
            project or self.default_project,
            version,
            alias,
        )
    
    def render(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        **variables: Any,
    ) -> str:
        """
        Render a prompt with variables.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use
            **variables: Template variables
        
        Returns:
            Rendered prompt string
        """
        return self.registry.render(
            name,
            project or self.default_project,
            version,
            alias,
            **variables,
        )
    
    def get_model_config(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
    ) -> dict[str, Any]:
        """
        Get model configuration from prompt metadata.
        
        Useful for configuring Strands model providers with
        settings stored in prompt metadata.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use
        
        Returns:
            Dict with model configuration
        """
        prompt_version = self.get_prompt(name, project, version, alias)
        
        if not prompt_version:
            return {}
        
        config = {}
        metadata = prompt_version.metadata
        
        if metadata.model:
            config["model_id"] = metadata.model
        if metadata.temperature is not None:
            config["temperature"] = metadata.temperature
        if metadata.max_tokens:
            config["max_tokens"] = metadata.max_tokens
        if metadata.stop_sequences:
            config["stop_sequences"] = metadata.stop_sequences
        
        # Include custom metadata
        config.update(metadata.custom)
        
        return config


# =============================================================================
# Convenience functions
# =============================================================================

def create_strands_toolkit(
    bucket: str | None = None,
    prefix: str = "promptflow",
    region: str | None = None,
) -> PromptFlowToolkit:
    """
    Factory function to create a PromptFlowToolkit with S3 storage.
    
    Args:
        bucket: S3 bucket for prompt storage
        prefix: S3 key prefix
        region: AWS region
    
    Returns:
        Configured PromptFlowToolkit
    """
    if bucket:
        registry = PromptRegistry.from_s3(
            bucket=bucket,
            prefix=prefix,
            region=region,
        )
    else:
        registry = PromptRegistry.from_env()
    
    return PromptFlowToolkit(registry=registry)
