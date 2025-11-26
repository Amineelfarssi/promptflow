"""
PromptRegistry - High-level API for prompt management.

This is the main entry point for using PromptFlow. It provides a clean,
intuitive interface for managing prompts while abstracting storage details.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from promptflow.core.models import (
    Prompt,
    PromptCollection,
    PromptExample,
    PromptMetadata,
    PromptType,
    PromptVersion,
    TemplateFormat,
)
from promptflow.storage.base import StorageBackend, PromptNotFoundError
from promptflow.storage.local import LocalStorageBackend
from promptflow.storage.s3 import S3StorageBackend

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _generate_id() -> str:
    """Generate a unique prompt ID."""
    return uuid.uuid4().hex[:12]


class PromptRegistry:
    """
    Main interface for prompt management.
    
    The registry provides a high-level API for:
    - Creating, updating, and deleting prompts
    - Version management with semantic aliases
    - Template rendering with variable substitution
    - Search and discovery
    - Export/import for backup and migration
    
    Example:
        # Local development
        registry = PromptRegistry()
        
        # Production with S3
        registry = PromptRegistry.from_s3(
            bucket="my-prompts",
            prefix="prod"
        )
        
        # Register a new prompt
        prompt = registry.register(
            name="summarizer",
            template="Summarize the following: {text}",
            project="nlp",
            tags=["production", "gpt-4"]
        )
        
        # Get and render
        rendered = registry.render("summarizer", text="Long document...")
    """
    
    def __init__(
        self,
        storage: StorageBackend | None = None,
        default_project: str = "default",
        default_user: str | None = None,
    ):
        """
        Initialize the registry.
        
        Args:
            storage: Storage backend (defaults to local storage)
            default_project: Default project for prompts
            default_user: Default user for audit trails
        """
        self.storage = storage or LocalStorageBackend()
        self.default_project = default_project
        self.default_user = default_user or os.getenv("USER", "unknown")
    
    @classmethod
    def from_s3(
        cls,
        bucket: str,
        prefix: str = "promptflow",
        region: str | None = None,
        endpoint_url: str | None = None,
        **kwargs: Any,
    ) -> PromptRegistry:
        """
        Create a registry backed by S3.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix (folder)
            region: AWS region
            endpoint_url: Custom endpoint (LocalStack, MinIO)
            **kwargs: Additional arguments for PromptRegistry
        """
        storage = S3StorageBackend(
            bucket=bucket,
            prefix=prefix,
            region=region,
            endpoint_url=endpoint_url,
        )
        return cls(storage=storage, **kwargs)
    
    @classmethod
    def from_env(cls, **kwargs: Any) -> PromptRegistry:
        """
        Create a registry from environment variables.
        
        Environment variables:
            PROMPTFLOW_STORAGE: 's3' or 'local' (default: 'local')
            PROMPTFLOW_S3_BUCKET: S3 bucket name
            PROMPTFLOW_S3_PREFIX: S3 key prefix
            PROMPTFLOW_S3_REGION: AWS region
            PROMPTFLOW_S3_ENDPOINT: Custom S3 endpoint
            PROMPTFLOW_LOCAL_PATH: Local storage path
            PROMPTFLOW_DEFAULT_PROJECT: Default project name
        """
        storage_type = os.getenv("PROMPTFLOW_STORAGE", "local")
        
        if storage_type == "s3":
            bucket = os.getenv("PROMPTFLOW_S3_BUCKET")
            if not bucket:
                raise ValueError("PROMPTFLOW_S3_BUCKET environment variable required")
            
            return cls.from_s3(
                bucket=bucket,
                prefix=os.getenv("PROMPTFLOW_S3_PREFIX", "promptflow"),
                region=os.getenv("PROMPTFLOW_S3_REGION"),
                endpoint_url=os.getenv("PROMPTFLOW_S3_ENDPOINT"),
                default_project=os.getenv("PROMPTFLOW_DEFAULT_PROJECT", "default"),
                **kwargs,
            )
        else:
            storage = LocalStorageBackend(
                base_path=os.getenv("PROMPTFLOW_LOCAL_PATH", ".promptflow")
            )
            return cls(
                storage=storage,
                default_project=os.getenv("PROMPTFLOW_DEFAULT_PROJECT", "default"),
                **kwargs,
            )
    
    # =========================================================================
    # Core CRUD Operations
    # =========================================================================
    
    def register(
        self,
        name: str,
        template: str,
        project: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        format: TemplateFormat = TemplateFormat.FSTRING,
        prompt_type: PromptType = PromptType.TEMPLATE,
        metadata: PromptMetadata | dict[str, Any] | None = None,
        examples: list[PromptExample | dict[str, Any]] | None = None,
        aliases: dict[str, int] | None = None,
        created_by: str | None = None,
    ) -> Prompt:
        """
        Register a new prompt.
        
        Args:
            name: Unique name within the project
            template: The prompt template string
            project: Project name (uses default if not specified)
            description: Human-readable description
            tags: List of tags for organization
            format: Template format (fstring, jinja2, mustache)
            prompt_type: Type of prompt (system, user, etc.)
            metadata: Model configuration and custom metadata
            examples: Example input/output pairs
            aliases: Initial aliases (e.g., {"prod": 1})
            created_by: User creating the prompt
        
        Returns:
            The created Prompt object
        
        Raises:
            ValueError: If a prompt with this name already exists in the project
        """
        project = project or self.default_project
        created_by = created_by or self.default_user
        
        # Check for existing prompt with same name
        existing = self.storage.get_prompt_by_name(name, project)
        if existing:
            raise ValueError(f"Prompt '{name}' already exists in project '{project}'")
        
        # Parse metadata if dict
        if isinstance(metadata, dict):
            metadata = PromptMetadata(**metadata)
        
        # Parse examples if dicts
        parsed_examples = []
        if examples:
            for ex in examples:
                if isinstance(ex, dict):
                    parsed_examples.append(PromptExample(**ex))
                else:
                    parsed_examples.append(ex)
        
        # Create the prompt
        prompt = Prompt(
            id=_generate_id(),
            name=name,
            description=description,
            project=project,
            tags=tags or [],
            created_by=created_by,
        )
        
        # Add initial version
        prompt.add_version(
            template=template,
            format=format,
            prompt_type=prompt_type,
            metadata=metadata,
            examples=parsed_examples,
            created_by=created_by,
            change_message="Initial version",
        )
        
        # Set initial aliases
        if aliases:
            for alias, version in aliases.items():
                prompt.set_alias(alias, version)
        
        # Save to storage
        self.storage.save_prompt(prompt)
        
        logger.info(f"Registered new prompt: {name} (id={prompt.id})")
        return prompt
    
    def get(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
    ) -> PromptVersion | None:
        """
        Get a specific version of a prompt.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version number
            alias: Alias to resolve (e.g., "prod")
        
        Returns:
            The PromptVersion or None if not found
        """
        project = project or self.default_project
        prompt = self.storage.get_prompt_by_name(name, project)
        
        if not prompt:
            return None
        
        if alias:
            return prompt.get_by_alias(alias)
        
        return prompt.get_version(version)
    
    def get_prompt(
        self,
        name: str,
        project: str | None = None,
    ) -> Prompt | None:
        """
        Get the full prompt object with all versions.
        
        Args:
            name: Prompt name
            project: Project name
        
        Returns:
            The Prompt object or None if not found
        """
        project = project or self.default_project
        return self.storage.get_prompt_by_name(name, project)
    
    def get_by_id(self, prompt_id: str) -> Prompt | None:
        """Get a prompt by its unique ID."""
        return self.storage.get_prompt(prompt_id)
    
    def update(
        self,
        name: str,
        template: str,
        project: str | None = None,
        format: TemplateFormat | None = None,
        prompt_type: PromptType | None = None,
        metadata: PromptMetadata | dict[str, Any] | None = None,
        examples: list[PromptExample | dict[str, Any]] | None = None,
        updated_by: str | None = None,
        change_message: str | None = None,
    ) -> PromptVersion:
        """
        Update a prompt by creating a new version.
        
        Prompts are immutable - updates create new versions.
        
        Args:
            name: Prompt name
            template: New template string
            project: Project name
            format: Template format (inherits from previous if not specified)
            prompt_type: Prompt type (inherits from previous if not specified)
            metadata: New metadata (merged with previous)
            examples: New examples (replaces previous)
            updated_by: User making the update
            change_message: Description of changes
        
        Returns:
            The new PromptVersion
        """
        project = project or self.default_project
        updated_by = updated_by or self.default_user
        
        prompt = self.storage.get_prompt_by_name(name, project)
        if not prompt:
            raise PromptNotFoundError(f"Prompt '{name}' not found in project '{project}'")
        
        # Get current version for defaults
        current = prompt.get_version()
        
        # Parse and merge metadata
        if isinstance(metadata, dict):
            metadata = PromptMetadata(**metadata)
        if metadata and current:
            metadata = current.metadata.merge(metadata)
        elif not metadata and current:
            metadata = current.metadata
        
        # Parse examples
        parsed_examples = []
        if examples:
            for ex in examples:
                if isinstance(ex, dict):
                    parsed_examples.append(PromptExample(**ex))
                else:
                    parsed_examples.append(ex)
        elif current:
            parsed_examples = current.examples
        
        # Create new version
        new_version = prompt.add_version(
            template=template,
            format=format or (current.format if current else TemplateFormat.FSTRING),
            prompt_type=prompt_type or (current.prompt_type if current else PromptType.TEMPLATE),
            metadata=metadata,
            examples=parsed_examples,
            created_by=updated_by,
            change_message=change_message,
        )
        
        # Save
        self.storage.save_prompt(prompt)
        
        logger.info(f"Updated prompt {name} to version {new_version.version}")
        return new_version
    
    def delete(
        self,
        name: str,
        project: str | None = None,
        hard: bool = False,
    ) -> bool:
        """
        Delete a prompt.
        
        Args:
            name: Prompt name
            project: Project name
            hard: If True, permanently delete. If False, soft delete.
        
        Returns:
            True if prompt was deleted
        """
        project = project or self.default_project
        prompt = self.storage.get_prompt_by_name(name, project)
        
        if not prompt:
            return False
        
        return self.storage.delete_prompt(prompt.id, hard=hard)
    
    # =========================================================================
    # Alias Management
    # =========================================================================
    
    def set_alias(
        self,
        name: str,
        alias: str,
        version: int,
        project: str | None = None,
    ) -> None:
        """
        Set an alias to point to a specific version.
        
        Common aliases: "prod", "staging", "dev", "latest", "stable"
        
        Args:
            name: Prompt name
            alias: Alias name
            version: Version number to point to
            project: Project name
        """
        project = project or self.default_project
        prompt = self.storage.get_prompt_by_name(name, project)
        
        if not prompt:
            raise PromptNotFoundError(f"Prompt '{name}' not found")
        
        prompt.set_alias(alias, version)
        self.storage.save_prompt(prompt)
        
        logger.info(f"Set alias '{alias}' -> v{version} for prompt {name}")
    
    def remove_alias(
        self,
        name: str,
        alias: str,
        project: str | None = None,
    ) -> bool:
        """Remove an alias from a prompt."""
        project = project or self.default_project
        prompt = self.storage.get_prompt_by_name(name, project)
        
        if not prompt:
            return False
        
        result = prompt.remove_alias(alias)
        if result:
            self.storage.save_prompt(prompt)
        
        return result
    
    def promote(
        self,
        name: str,
        from_alias: str,
        to_alias: str,
        project: str | None = None,
    ) -> int:
        """
        Promote a version from one alias to another.
        
        Example: promote("summarizer", "staging", "prod")
        
        Returns the version number that was promoted.
        """
        project = project or self.default_project
        prompt = self.storage.get_prompt_by_name(name, project)
        
        if not prompt:
            raise PromptNotFoundError(f"Prompt '{name}' not found")
        
        version = prompt.aliases.get(from_alias)
        if version is None:
            raise ValueError(f"Alias '{from_alias}' not found")
        
        prompt.set_alias(to_alias, version)
        self.storage.save_prompt(prompt)
        
        logger.info(f"Promoted {name} v{version} from '{from_alias}' to '{to_alias}'")
        return version
    
    # =========================================================================
    # Rendering
    # =========================================================================
    
    def render(
        self,
        name: str,
        project: str | None = None,
        version: int | None = None,
        alias: str | None = None,
        **variables: Any,
    ) -> str:
        """
        Render a prompt template with variables.
        
        Args:
            name: Prompt name
            project: Project name
            version: Specific version
            alias: Alias to use (e.g., "prod")
            **variables: Template variables
        
        Returns:
            Rendered prompt string
        """
        prompt_version = self.get(name, project, version, alias)
        
        if not prompt_version:
            raise PromptNotFoundError(f"Prompt '{name}' not found")
        
        return prompt_version.render(variables)
    
    # =========================================================================
    # Discovery
    # =========================================================================
    
    def list(
        self,
        project: str | None = None,
        tags: list[str] | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Prompt]:
        """
        List prompts with optional filtering.
        
        Args:
            project: Filter by project
            tags: Filter by tags (AND logic)
            include_deleted: Include soft-deleted prompts
            limit: Maximum results
            offset: Pagination offset
        """
        return self.storage.list_prompts(
            project=project,
            tags=tags,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
    
    def search(
        self,
        query: str,
        project: str | None = None,
        limit: int = 20,
    ) -> list[Prompt]:
        """
        Search prompts by text.
        
        Searches in name, description, tags, and template content.
        """
        return self.storage.search(query, project, limit)
    
    def list_projects(self) -> list[str]:
        """List all projects."""
        return self.storage.list_projects()
    
    # =========================================================================
    # Version Comparison
    # =========================================================================
    
    def compare(
        self,
        name: str,
        v1: int,
        v2: int,
        project: str | None = None,
    ) -> dict[str, Any]:
        """
        Compare two versions of a prompt.
        
        Returns a dict with differences including template changes,
        variable changes, and metadata changes.
        """
        project = project or self.default_project
        prompt = self.storage.get_prompt_by_name(name, project)
        
        if not prompt:
            raise PromptNotFoundError(f"Prompt '{name}' not found")
        
        return prompt.compare_versions(v1, v2)
    
    def history(
        self,
        name: str,
        project: str | None = None,
        limit: int = 10,
    ) -> list[PromptVersion]:
        """
        Get version history for a prompt.
        
        Returns versions in reverse chronological order.
        """
        project = project or self.default_project
        prompt = self.storage.get_prompt_by_name(name, project)
        
        if not prompt:
            return []
        
        versions = sorted(prompt.versions, key=lambda v: v.version, reverse=True)
        return versions[:limit]
    
    def rollback(
        self,
        name: str,
        to_version: int,
        project: str | None = None,
        alias: str | None = None,
        rolled_back_by: str | None = None,
    ) -> PromptVersion:
        """
        Rollback to a previous version.
        
        This creates a new version with the same content as the target version.
        If an alias is specified, it will point to the new version.
        
        Args:
            name: Prompt name
            to_version: Version to rollback to
            project: Project name
            alias: Alias to update (e.g., "prod")
            rolled_back_by: User performing rollback
        
        Returns:
            The new PromptVersion
        """
        project = project or self.default_project
        rolled_back_by = rolled_back_by or self.default_user
        
        prompt = self.storage.get_prompt_by_name(name, project)
        if not prompt:
            raise PromptNotFoundError(f"Prompt '{name}' not found")
        
        target = prompt.get_version(to_version)
        if not target:
            raise ValueError(f"Version {to_version} not found")
        
        # Create new version with rolled back content
        new_version = prompt.add_version(
            template=target.template,
            format=target.format,
            prompt_type=target.prompt_type,
            metadata=target.metadata,
            examples=target.examples,
            created_by=rolled_back_by,
            change_message=f"Rollback to version {to_version}",
        )
        
        # Update alias if specified
        if alias:
            prompt.set_alias(alias, new_version.version)
        
        self.storage.save_prompt(prompt)
        
        logger.info(f"Rolled back {name} to v{to_version} (new version: {new_version.version})")
        return new_version
    
    # =========================================================================
    # Export/Import
    # =========================================================================
    
    def export_project(self, project: str) -> PromptCollection:
        """Export all prompts in a project."""
        return self.storage.export_project(project)
    
    def import_collection(
        self,
        collection: PromptCollection,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Import a collection of prompts."""
        return self.storage.import_collection(collection, overwrite)
    
    def export_json(self, project: str) -> str:
        """Export project as JSON string."""
        import json
        collection = self.export_project(project)
        return json.dumps(collection.to_export(), indent=2)
    
    def import_json(self, json_str: str, overwrite: bool = False) -> dict[str, Any]:
        """Import prompts from JSON string."""
        import json
        data = json.loads(json_str)
        collection = PromptCollection.from_import(data)
        return self.import_collection(collection, overwrite)
