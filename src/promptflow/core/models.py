"""
Core data models for prompt management.

Inspired by PromptLayer's registry and Cuebit's versioning approach,
but designed for S3-native storage with minimal dependencies.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import xxhash
from pydantic import BaseModel, Field, computed_field, model_validator


class TemplateFormat(str, Enum):
    """Supported template formats."""
    FSTRING = "fstring"      # {variable}
    JINJA2 = "jinja2"        # {{ variable }}
    MUSTACHE = "mustache"    # {{ variable }} with logic


class PromptType(str, Enum):
    """Types of prompts."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TEMPLATE = "template"    # Full template with all parts


class PromptExample(BaseModel):
    """Example input/output pair for a prompt."""
    input_vars: dict[str, Any] = Field(default_factory=dict)
    expected_output: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PromptMetadata(BaseModel):
    """Metadata associated with a prompt."""
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)
    
    def merge(self, other: PromptMetadata) -> PromptMetadata:
        """Merge another metadata object, with other taking precedence."""
        merged = self.model_dump()
        other_dict = other.model_dump(exclude_none=True)
        
        for key, value in other_dict.items():
            if key == "custom":
                merged["custom"] = {**merged.get("custom", {}), **value}
            elif value is not None:
                merged[key] = value
        
        return PromptMetadata(**merged)


class PromptVersion(BaseModel):
    """
    A specific version of a prompt.
    
    Versions are immutable once created. Changes create new versions.
    Content hash ensures version integrity and enables deduplication.
    """
    version: int
    template: str
    format: TemplateFormat = TemplateFormat.FSTRING
    prompt_type: PromptType = PromptType.TEMPLATE
    metadata: PromptMetadata = Field(default_factory=PromptMetadata)
    examples: list[PromptExample] = Field(default_factory=list)
    
    # Audit fields
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    change_message: str | None = None
    parent_version: int | None = None
    
    # Computed on validation
    content_hash: str = ""
    variables: list[str] = Field(default_factory=list)
    
    @model_validator(mode="after")
    def compute_fields(self) -> PromptVersion:
        """Compute hash and extract variables after model creation."""
        # Compute content hash
        content = f"{self.template}|{self.format.value}|{self.prompt_type.value}"
        self.content_hash = xxhash.xxh64(content.encode()).hexdigest()
        
        # Extract variables based on format
        self.variables = self._extract_variables()
        
        return self
    
    def _extract_variables(self) -> list[str]:
        """Extract variable names from template based on format."""
        if self.format == TemplateFormat.FSTRING:
            # Match {var} but not {{escaped}}
            pattern = r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})"
            return list(set(re.findall(pattern, self.template)))
        
        elif self.format in (TemplateFormat.JINJA2, TemplateFormat.MUSTACHE):
            # Match {{ var }} with flexible whitespace
            pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
            return list(set(re.findall(pattern, self.template)))
        
        return []
    
    def render(self, variables: dict[str, Any]) -> str:
        """Render the template with provided variables."""
        if self.format == TemplateFormat.FSTRING:
            try:
                return self.template.format(**variables)
            except KeyError as e:
                raise ValueError(f"Missing required variable: {e}")
        
        elif self.format in (TemplateFormat.JINJA2, TemplateFormat.MUSTACHE):
            from jinja2 import Template, StrictUndefined
            try:
                jinja_template = Template(self.template, undefined=StrictUndefined)
                return jinja_template.render(**variables)
            except Exception as e:
                raise ValueError(f"Template rendering failed: {e}")
        
        return self.template


class Prompt(BaseModel):
    """
    A prompt with full version history.
    
    This is the main entity for prompt management. It contains:
    - Unique identifier and human-readable name
    - Full version history (all versions are preserved)
    - Aliases for semantic versioning (prod, staging, etc.)
    - Organization via project and tags
    """
    id: str
    name: str
    description: str | None = None
    project: str = "default"
    tags: list[str] = Field(default_factory=list)
    
    # Version management
    versions: list[PromptVersion] = Field(default_factory=list)
    aliases: dict[str, int] = Field(default_factory=dict)  # alias -> version number
    
    # Audit
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str | None = None
    
    # Soft delete support
    is_deleted: bool = False
    deleted_at: datetime | None = None
    
    @computed_field
    @property
    def latest_version(self) -> int:
        """Get the latest version number."""
        return max((v.version for v in self.versions), default=0)
    
    @computed_field
    @property
    def version_count(self) -> int:
        """Get total number of versions."""
        return len(self.versions)
    
    def get_version(self, version: int | None = None) -> PromptVersion | None:
        """Get a specific version or the latest if not specified."""
        if not self.versions:
            return None
        
        if version is None:
            version = self.latest_version
        
        for v in self.versions:
            if v.version == version:
                return v
        return None
    
    def get_by_alias(self, alias: str) -> PromptVersion | None:
        """Get version by alias (e.g., 'prod', 'staging')."""
        version_num = self.aliases.get(alias)
        if version_num is not None:
            return self.get_version(version_num)
        return None
    
    def add_version(
        self,
        template: str,
        format: TemplateFormat = TemplateFormat.FSTRING,
        prompt_type: PromptType = PromptType.TEMPLATE,
        metadata: PromptMetadata | None = None,
        examples: list[PromptExample] | None = None,
        created_by: str | None = None,
        change_message: str | None = None,
    ) -> PromptVersion:
        """Add a new version to this prompt."""
        new_version = PromptVersion(
            version=self.latest_version + 1,
            template=template,
            format=format,
            prompt_type=prompt_type,
            metadata=metadata or PromptMetadata(),
            examples=examples or [],
            created_by=created_by,
            change_message=change_message,
            parent_version=self.latest_version if self.versions else None,
        )
        
        self.versions.append(new_version)
        self.updated_at = datetime.now(timezone.utc)
        
        return new_version
    
    def set_alias(self, alias: str, version: int) -> None:
        """Set an alias to point to a specific version."""
        if not any(v.version == version for v in self.versions):
            raise ValueError(f"Version {version} does not exist")
        
        self.aliases[alias] = version
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_alias(self, alias: str) -> bool:
        """Remove an alias. Returns True if alias existed."""
        if alias in self.aliases:
            del self.aliases[alias]
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False
    
    def compare_versions(self, v1: int, v2: int) -> dict[str, Any]:
        """Compare two versions and return differences."""
        version1 = self.get_version(v1)
        version2 = self.get_version(v2)
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        return {
            "v1": v1,
            "v2": v2,
            "template_changed": version1.template != version2.template,
            "format_changed": version1.format != version2.format,
            "metadata_changed": version1.metadata != version2.metadata,
            "variables_added": list(set(version2.variables) - set(version1.variables)),
            "variables_removed": list(set(version1.variables) - set(version2.variables)),
            "v1_template": version1.template,
            "v2_template": version2.template,
        }
    
    def soft_delete(self) -> None:
        """Mark prompt as deleted without removing data."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def restore(self) -> None:
        """Restore a soft-deleted prompt."""
        self.is_deleted = False
        self.deleted_at = None
        self.updated_at = datetime.now(timezone.utc)


class PromptCollection(BaseModel):
    """
    A collection of prompts, typically representing a project or workspace.
    Used for batch operations like export/import.
    """
    name: str
    description: str | None = None
    prompts: list[Prompt] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exported_at: datetime | None = None
    version: str = "1.0"
    
    def to_export(self) -> dict[str, Any]:
        """Prepare collection for export."""
        self.exported_at = datetime.now(timezone.utc)
        return self.model_dump(mode="json")
    
    @classmethod
    def from_import(cls, data: dict[str, Any]) -> PromptCollection:
        """Create collection from imported data."""
        return cls.model_validate(data)
