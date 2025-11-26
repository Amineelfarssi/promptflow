"""
Abstract storage interface for prompt management.

This module defines the contract that all storage backends must implement.
Enables easy switching between S3, local filesystem, or other backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from promptflow.core.models import Prompt, PromptCollection


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    All storage operations are designed to be atomic and consistent.
    Implementations must handle their own locking/consistency mechanisms.
    """
    
    @abstractmethod
    def save_prompt(self, prompt: Prompt) -> None:
        """
        Save a prompt to storage.
        
        If the prompt already exists (by id), it will be overwritten.
        This is the primary write operation - versioning is handled at
        the Prompt model level.
        """
        pass
    
    @abstractmethod
    def get_prompt(self, prompt_id: str) -> Prompt | None:
        """
        Retrieve a prompt by its unique ID.
        
        Returns None if the prompt doesn't exist.
        """
        pass
    
    @abstractmethod
    def get_prompt_by_name(self, name: str, project: str = "default") -> Prompt | None:
        """
        Retrieve a prompt by name within a project.
        
        Names are unique within a project.
        Returns None if not found.
        """
        pass
    
    @abstractmethod
    def list_prompts(
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
            project: Filter by project name
            tags: Filter by tags (AND logic - must have all)
            include_deleted: Include soft-deleted prompts
            limit: Maximum number of results
            offset: Pagination offset
        """
        pass
    
    @abstractmethod
    def delete_prompt(self, prompt_id: str, hard: bool = False) -> bool:
        """
        Delete a prompt.
        
        Args:
            prompt_id: The prompt's unique ID
            hard: If True, permanently delete. If False, soft delete.
        
        Returns True if the prompt existed and was deleted.
        """
        pass
    
    @abstractmethod
    def list_projects(self) -> list[str]:
        """List all unique project names."""
        pass
    
    @abstractmethod
    def export_project(self, project: str) -> PromptCollection:
        """
        Export all prompts in a project as a collection.
        
        Useful for backup, migration, or sharing.
        """
        pass
    
    @abstractmethod
    def import_collection(
        self,
        collection: PromptCollection,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Import a collection of prompts.
        
        Args:
            collection: The collection to import
            overwrite: If True, overwrite existing prompts with same ID
        
        Returns a summary of the import operation.
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        project: str | None = None,
        limit: int = 20,
    ) -> list[Prompt]:
        """
        Search prompts by text in name, description, or template content.
        
        This is a simple text search - implementations may vary in sophistication.
        """
        pass


class StorageError(Exception):
    """Base exception for storage errors."""
    pass


class PromptNotFoundError(StorageError):
    """Raised when a prompt is not found."""
    pass


class PromptExistsError(StorageError):
    """Raised when trying to create a prompt that already exists."""
    pass


class StorageConnectionError(StorageError):
    """Raised when connection to storage fails."""
    pass
