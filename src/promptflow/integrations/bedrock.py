"""
AWS Bedrock integration for PromptFlow.

Provides helper functions and utilities to use PromptFlow-managed prompts
directly with Amazon Bedrock's Converse API.

Example:
    from promptflow.integrations.bedrock import BedrockPromptHelper
    import boto3
    
    bedrock = boto3.client("bedrock-runtime")
    helper = BedrockPromptHelper()
    
    # Get formatted messages for Bedrock Converse API
    messages = helper.get_converse_messages(
        "summarizer",
        alias="prod",
        text="Your long document here..."
    )
    
    response = bedrock.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        **helper.get_inference_config("summarizer")
    )
"""

from __future__ import annotations

import json
from typing import Any

from promptflow.core.models import PromptMetadata, PromptType, PromptVersion
from promptflow.core.registry import PromptRegistry


class BedrockPromptHelper:
    """
    Helper class for using PromptFlow prompts with Amazon Bedrock.
    
    Formats prompts for use with Bedrock's Converse API and handles
    inference configuration from prompt metadata.
    """
    
    def __init__(
        self,
        registry: PromptRegistry | None = None,
        default_project: str = "default",
    ):
        """
        Initialize the Bedrock helper.
        
        Args:
            registry: PromptFlow registry instance
            default_project: Default project for prompt lookups
        """
        self.registry = registry or PromptRegistry.from_env()
        self.default_project = default_project
    
    def get_prompt_version(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
    ) -> PromptVersion:
        """
        Get a prompt version from the registry.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version number
            alias: Alias to use (e.g., "prod")
        
        Returns:
            PromptVersion object
        
        Raises:
            ValueError: If prompt not found
        """
        project = project or self.default_project
        prompt_version = self.registry.get(name, project, version, alias)
        
        if not prompt_version:
            raise ValueError(f"Prompt '{name}' not found in project '{project}'")
        
        return prompt_version
    
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
    
    def get_converse_messages(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        role: str = "user",
        **variables: Any,
    ) -> list[dict[str, Any]]:
        """
        Get messages formatted for Bedrock Converse API.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use
            role: Message role ("user" or "assistant")
            **variables: Template variables
        
        Returns:
            List of message dicts for Converse API
        
        Example:
            messages = helper.get_converse_messages(
                "summarizer",
                alias="prod",
                text="Document to summarize..."
            )
            
            response = bedrock.converse(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=messages
            )
        """
        rendered = self.render(name, project, version, alias, **variables)
        
        return [
            {
                "role": role,
                "content": [{"text": rendered}]
            }
        ]
    
    def get_system_prompt(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        **variables: Any,
    ) -> list[dict[str, str]]:
        """
        Get a system prompt formatted for Bedrock Converse API.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use
            **variables: Template variables
        
        Returns:
            System prompt list for Converse API's 'system' parameter
        
        Example:
            system = helper.get_system_prompt("chat-system", alias="prod")
            messages = [{"role": "user", "content": [{"text": "Hello!"}]}]
            
            response = bedrock.converse(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                system=system,
                messages=messages
            )
        """
        rendered = self.render(name, project, version, alias, **variables)
        return [{"text": rendered}]
    
    def get_inference_config(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Get inference configuration from prompt metadata.
        
        Maps PromptFlow metadata to Bedrock's inferenceConfig parameter.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use
            override: Values to override from metadata
        
        Returns:
            Dict suitable for Converse API's 'inferenceConfig' parameter
        
        Example:
            config = helper.get_inference_config("summarizer", alias="prod")
            
            response = bedrock.converse(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=messages,
                inferenceConfig=config
            )
        """
        prompt_version = self.get_prompt_version(name, project, version, alias)
        metadata = prompt_version.metadata
        
        config: dict[str, Any] = {}
        
        if metadata.max_tokens:
            config["maxTokens"] = metadata.max_tokens
        
        if metadata.temperature is not None:
            config["temperature"] = metadata.temperature
        
        if metadata.stop_sequences:
            config["stopSequences"] = metadata.stop_sequences
        
        # Check for topP in custom metadata
        if "top_p" in metadata.custom:
            config["topP"] = metadata.custom["top_p"]
        
        # Apply overrides
        if override:
            config.update(override)
        
        return config
    
    def get_model_id(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        default: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    ) -> str:
        """
        Get model ID from prompt metadata.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use
            default: Default model if not specified in metadata
        
        Returns:
            Bedrock model ID string
        """
        prompt_version = self.get_prompt_version(name, project, version, alias)
        return prompt_version.metadata.model or default
    
    def build_converse_request(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        system_prompt_name: str | None = None,
        system_prompt_alias: str | None = None,
        default_model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        **variables: Any,
    ) -> dict[str, Any]:
        """
        Build a complete Converse API request from prompts.
        
        This is a convenience method that combines messages, system prompt,
        inference config, and model ID into a single dict ready for the API.
        
        Args:
            name: User prompt name
            project: Project name
            version: Specific version
            alias: Alias for user prompt
            system_prompt_name: Optional system prompt name
            system_prompt_alias: Alias for system prompt
            default_model: Default model ID
            **variables: Template variables
        
        Returns:
            Dict with all Converse API parameters
        
        Example:
            request = helper.build_converse_request(
                "summarizer",
                alias="prod",
                system_prompt_name="summarizer-system",
                text="Your document here..."
            )
            
            response = bedrock.converse(**request)
        """
        request: dict[str, Any] = {
            "modelId": self.get_model_id(name, project, version, alias, default_model),
            "messages": self.get_converse_messages(
                name, project, version, alias, **variables
            ),
        }
        
        # Add system prompt if specified
        if system_prompt_name:
            request["system"] = self.get_system_prompt(
                system_prompt_name,
                project,
                version=None,
                alias=system_prompt_alias,
                **variables,
            )
        
        # Add inference config
        config = self.get_inference_config(name, project, version, alias)
        if config:
            request["inferenceConfig"] = config
        
        return request


def create_bedrock_helper(
    bucket: str | None = None,
    prefix: str = "promptflow",
    region: str | None = None,
) -> BedrockPromptHelper:
    """
    Factory function to create a BedrockPromptHelper with S3 storage.
    
    Args:
        bucket: S3 bucket for prompt storage
        prefix: S3 key prefix
        region: AWS region
    
    Returns:
        Configured BedrockPromptHelper
    """
    if bucket:
        registry = PromptRegistry.from_s3(
            bucket=bucket,
            prefix=prefix,
            region=region,
        )
    else:
        registry = PromptRegistry.from_env()
    
    return BedrockPromptHelper(registry=registry)
