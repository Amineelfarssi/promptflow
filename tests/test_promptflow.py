"""
Tests for PromptFlow prompt management system.

Uses pytest with moto for S3 mocking.
"""

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from promptflow.core.models import (
    Prompt,
    PromptCollection,
    PromptExample,
    PromptMetadata,
    PromptType,
    PromptVersion,
    TemplateFormat,
)
from promptflow.core.registry import PromptRegistry
from promptflow.storage.local import LocalStorageBackend


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for local storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def local_storage(temp_dir):
    """Create a local storage backend in temp directory."""
    return LocalStorageBackend(base_path=temp_dir)


@pytest.fixture
def registry(local_storage):
    """Create a registry with local storage."""
    return PromptRegistry(storage=local_storage)


# =============================================================================
# Model Tests
# =============================================================================

class TestPromptVersion:
    """Tests for PromptVersion model."""
    
    def test_create_version(self):
        """Test creating a basic prompt version."""
        version = PromptVersion(
            version=1,
            template="Hello, {name}!",
        )
        
        assert version.version == 1
        assert version.template == "Hello, {name}!"
        assert version.format == TemplateFormat.FSTRING
        assert version.variables == ["name"]
        assert version.content_hash  # Should be computed
    
    def test_variable_extraction_fstring(self):
        """Test variable extraction for f-string format."""
        version = PromptVersion(
            version=1,
            template="Summarize {text} in {style} style for {audience}.",
            format=TemplateFormat.FSTRING,
        )
        
        assert sorted(version.variables) == ["audience", "style", "text"]
    
    def test_variable_extraction_jinja2(self):
        """Test variable extraction for Jinja2 format."""
        version = PromptVersion(
            version=1,
            template="Summarize {{ text }} in {{ style }} style.",
            format=TemplateFormat.JINJA2,
        )
        
        assert sorted(version.variables) == ["style", "text"]
    
    def test_render_fstring(self):
        """Test rendering f-string template."""
        version = PromptVersion(
            version=1,
            template="Hello, {name}! You are {age} years old.",
        )
        
        result = version.render({"name": "Alice", "age": 30})
        assert result == "Hello, Alice! You are 30 years old."
    
    def test_render_jinja2(self):
        """Test rendering Jinja2 template."""
        version = PromptVersion(
            version=1,
            template="Hello, {{ name }}! You are {{ age }} years old.",
            format=TemplateFormat.JINJA2,
        )
        
        result = version.render({"name": "Bob", "age": 25})
        assert result == "Hello, Bob! You are 25 years old."
    
    def test_render_missing_variable(self):
        """Test rendering with missing variable raises error."""
        version = PromptVersion(
            version=1,
            template="Hello, {name}!",
        )
        
        with pytest.raises(ValueError):
            version.render({})
    
    def test_content_hash_consistency(self):
        """Test that same content produces same hash."""
        v1 = PromptVersion(version=1, template="Test template")
        v2 = PromptVersion(version=2, template="Test template")
        
        assert v1.content_hash == v2.content_hash
    
    def test_content_hash_different(self):
        """Test that different content produces different hash."""
        v1 = PromptVersion(version=1, template="Template A")
        v2 = PromptVersion(version=1, template="Template B")
        
        assert v1.content_hash != v2.content_hash


class TestPrompt:
    """Tests for Prompt model."""
    
    def test_create_prompt(self):
        """Test creating a basic prompt."""
        prompt = Prompt(
            id="test123",
            name="test-prompt",
            description="A test prompt",
            project="default",
            tags=["test", "example"],
        )
        
        assert prompt.id == "test123"
        assert prompt.name == "test-prompt"
        assert prompt.latest_version == 0
        assert prompt.version_count == 0
    
    def test_add_version(self):
        """Test adding versions to a prompt."""
        prompt = Prompt(id="test", name="test")
        
        v1 = prompt.add_version(
            template="Version 1: {text}",
            created_by="alice",
            change_message="Initial version",
        )
        
        assert v1.version == 1
        assert prompt.latest_version == 1
        assert prompt.version_count == 1
        
        v2 = prompt.add_version(
            template="Version 2: {text}",
            created_by="bob",
            change_message="Updated template",
        )
        
        assert v2.version == 2
        assert v2.parent_version == 1
        assert prompt.latest_version == 2
    
    def test_get_version(self):
        """Test getting specific versions."""
        prompt = Prompt(id="test", name="test")
        prompt.add_version(template="v1")
        prompt.add_version(template="v2")
        prompt.add_version(template="v3")
        
        # Get latest
        assert prompt.get_version().template == "v3"
        
        # Get specific
        assert prompt.get_version(1).template == "v1"
        assert prompt.get_version(2).template == "v2"
        
        # Get non-existent
        assert prompt.get_version(999) is None
    
    def test_aliases(self):
        """Test alias management."""
        prompt = Prompt(id="test", name="test")
        prompt.add_version(template="v1")
        prompt.add_version(template="v2")
        prompt.add_version(template="v3")
        
        prompt.set_alias("prod", 2)
        prompt.set_alias("staging", 3)
        
        assert prompt.get_by_alias("prod").version == 2
        assert prompt.get_by_alias("staging").version == 3
        assert prompt.get_by_alias("nonexistent") is None
    
    def test_alias_invalid_version(self):
        """Test setting alias to non-existent version."""
        prompt = Prompt(id="test", name="test")
        prompt.add_version(template="v1")
        
        with pytest.raises(ValueError):
            prompt.set_alias("prod", 999)
    
    def test_compare_versions(self):
        """Test version comparison."""
        prompt = Prompt(id="test", name="test")
        prompt.add_version(template="Hello {name}")
        prompt.add_version(template="Hi {name}, welcome to {place}!")
        
        diff = prompt.compare_versions(1, 2)
        
        assert diff["template_changed"] is True
        assert "place" in diff["variables_added"]
    
    def test_soft_delete(self):
        """Test soft delete and restore."""
        prompt = Prompt(id="test", name="test")
        
        assert prompt.is_deleted is False
        
        prompt.soft_delete()
        assert prompt.is_deleted is True
        assert prompt.deleted_at is not None
        
        prompt.restore()
        assert prompt.is_deleted is False
        assert prompt.deleted_at is None


# =============================================================================
# Registry Tests
# =============================================================================

class TestPromptRegistry:
    """Tests for PromptRegistry."""
    
    def test_register_prompt(self, registry):
        """Test registering a new prompt."""
        prompt = registry.register(
            name="summarizer",
            template="Summarize: {text}",
            project="nlp",
            description="Text summarization prompt",
            tags=["prod", "nlp"],
        )
        
        assert prompt.name == "summarizer"
        assert prompt.project == "nlp"
        assert prompt.latest_version == 1
    
    def test_register_duplicate_fails(self, registry):
        """Test that registering duplicate name fails."""
        registry.register(name="test", template="v1")
        
        with pytest.raises(ValueError, match="already exists"):
            registry.register(name="test", template="v2")
    
    def test_get_prompt(self, registry):
        """Test getting a prompt."""
        registry.register(name="test", template="Hello {name}")
        
        version = registry.get("test")
        assert version.template == "Hello {name}"
    
    def test_get_specific_version(self, registry):
        """Test getting specific version."""
        registry.register(name="test", template="v1")
        registry.update("test", template="v2")
        
        assert registry.get("test", version=1).template == "v1"
        assert registry.get("test", version=2).template == "v2"
        assert registry.get("test").template == "v2"  # Latest
    
    def test_get_by_alias(self, registry):
        """Test getting by alias."""
        registry.register(name="test", template="v1")
        registry.update("test", template="v2")
        registry.set_alias("test", "prod", 1)
        
        assert registry.get("test", alias="prod").version == 1
        assert registry.get("test").version == 2
    
    def test_update_prompt(self, registry):
        """Test updating a prompt."""
        registry.register(name="test", template="v1", metadata={"model": "gpt-4"})
        
        new_version = registry.update(
            "test",
            template="v2",
            metadata={"temperature": 0.7},
            change_message="Updated template",
        )
        
        assert new_version.version == 2
        assert new_version.template == "v2"
        # Metadata should be merged
        assert new_version.metadata.model == "gpt-4"
        assert new_version.metadata.temperature == 0.7
    
    def test_render_prompt(self, registry):
        """Test rendering a prompt."""
        registry.register(name="greeting", template="Hello, {name}!")
        
        result = registry.render("greeting", name="World")
        assert result == "Hello, World!"
    
    def test_delete_prompt(self, registry):
        """Test deleting a prompt."""
        registry.register(name="test", template="v1")
        
        assert registry.get("test") is not None
        
        registry.delete("test")
        
        # Soft deleted - should not appear in normal get
        assert registry.get("test") is None
    
    def test_list_prompts(self, registry):
        """Test listing prompts."""
        registry.register(name="p1", template="t1", project="proj1", tags=["a"])
        registry.register(name="p2", template="t2", project="proj1", tags=["a", "b"])
        registry.register(name="p3", template="t3", project="proj2", tags=["b"])
        
        # All prompts
        all_prompts = registry.list()
        assert len(all_prompts) == 3
        
        # By project
        proj1 = registry.list(project="proj1")
        assert len(proj1) == 2
        
        # By tags
        tagged_b = registry.list(tags=["b"])
        assert len(tagged_b) == 2
    
    def test_search_prompts(self, registry):
        """Test searching prompts."""
        registry.register(
            name="summarizer",
            template="Summarize the text",
            description="Summarization prompt",
        )
        registry.register(
            name="translator",
            template="Translate to {language}",
            description="Translation prompt",
        )
        
        results = registry.search("summar")
        assert len(results) == 1
        assert results[0].name == "summarizer"
    
    def test_promote_alias(self, registry):
        """Test promoting from one alias to another."""
        registry.register(name="test", template="v1")
        registry.update("test", template="v2")
        registry.set_alias("test", "staging", 2)
        
        promoted_version = registry.promote("test", "staging", "prod")
        
        assert promoted_version == 2
        assert registry.get("test", alias="prod").version == 2
    
    def test_rollback(self, registry):
        """Test rolling back to previous version."""
        registry.register(name="test", template="v1 - original")
        registry.update("test", template="v2 - broken")
        
        new_version = registry.rollback("test", to_version=1)
        
        assert new_version.version == 3
        assert new_version.template == "v1 - original"
        assert "Rollback" in new_version.change_message
    
    def test_history(self, registry):
        """Test getting version history."""
        registry.register(name="test", template="v1")
        registry.update("test", template="v2")
        registry.update("test", template="v3")
        
        history = registry.history("test")
        
        assert len(history) == 3
        assert history[0].version == 3  # Most recent first
        assert history[2].version == 1
    
    def test_export_import(self, registry, temp_dir):
        """Test export and import functionality."""
        # Create prompts
        registry.register(name="p1", template="t1", project="export-test")
        registry.register(name="p2", template="t2", project="export-test")
        
        # Export
        json_str = registry.export_json("export-test")
        
        # Create new registry
        new_storage = LocalStorageBackend(base_path=f"{temp_dir}/new")
        new_registry = PromptRegistry(storage=new_storage)
        
        # Import
        stats = new_registry.import_json(json_str)
        
        assert stats["imported"] == 2
        assert new_registry.get("p1", project="export-test") is not None


# =============================================================================
# Storage Tests
# =============================================================================

class TestLocalStorage:
    """Tests for LocalStorageBackend."""
    
    def test_save_and_get(self, local_storage):
        """Test saving and retrieving a prompt."""
        prompt = Prompt(id="test123", name="test", project="default")
        prompt.add_version(template="Hello")
        
        local_storage.save_prompt(prompt)
        
        retrieved = local_storage.get_prompt("test123")
        assert retrieved is not None
        assert retrieved.name == "test"
    
    def test_get_by_name(self, local_storage):
        """Test getting prompt by name."""
        prompt = Prompt(id="test123", name="my-prompt", project="myproj")
        prompt.add_version(template="Test")
        local_storage.save_prompt(prompt)
        
        retrieved = local_storage.get_prompt_by_name("my-prompt", "myproj")
        assert retrieved is not None
        assert retrieved.id == "test123"
    
    def test_list_projects(self, local_storage):
        """Test listing projects."""
        for i, proj in enumerate(["alpha", "beta", "gamma"]):
            p = Prompt(id=f"p{i}", name=f"prompt{i}", project=proj)
            p.add_version(template=f"t{i}")
            local_storage.save_prompt(p)
        
        projects = local_storage.list_projects()
        assert sorted(projects) == ["alpha", "beta", "gamma"]
    
    def test_delete_soft(self, local_storage):
        """Test soft delete."""
        prompt = Prompt(id="test", name="test")
        prompt.add_version(template="v1")
        local_storage.save_prompt(prompt)
        
        local_storage.delete_prompt("test", hard=False)
        
        # Should still exist but be marked deleted
        retrieved = local_storage.get_prompt("test")
        assert retrieved.is_deleted is True
    
    def test_delete_hard(self, local_storage):
        """Test hard delete."""
        prompt = Prompt(id="test", name="test")
        prompt.add_version(template="v1")
        local_storage.save_prompt(prompt)
        
        local_storage.delete_prompt("test", hard=True)
        
        # Should be gone
        assert local_storage.get_prompt("test") is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestBedrockIntegration:
    """Tests for Bedrock integration."""
    
    def test_get_converse_messages(self, registry):
        """Test getting Bedrock Converse API messages."""
        from promptflow.integrations.bedrock import BedrockPromptHelper
        
        registry.register(
            name="summarizer",
            template="Summarize this text: {text}",
            metadata={"model": "anthropic.claude-3-sonnet", "temperature": 0.7},
        )
        
        helper = BedrockPromptHelper(registry=registry)
        messages = helper.get_converse_messages(
            "summarizer",
            text="Long document here..."
        )
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Long document here" in messages[0]["content"][0]["text"]
    
    def test_get_inference_config(self, registry):
        """Test getting inference config from metadata."""
        from promptflow.integrations.bedrock import BedrockPromptHelper
        
        registry.register(
            name="test",
            template="Test",
            metadata={
                "model": "claude-3",
                "temperature": 0.5,
                "max_tokens": 1000,
            },
        )
        
        helper = BedrockPromptHelper(registry=registry)
        config = helper.get_inference_config("test")
        
        assert config["temperature"] == 0.5
        assert config["maxTokens"] == 1000


class TestStrandsIntegration:
    """Tests for Strands Agents integration."""
    
    def test_get_prompt_tool(self, registry):
        """Test the get_prompt tool function."""
        from promptflow.integrations.strands import get_prompt, set_registry
        
        set_registry(registry)
        
        registry.register(
            name="agent-helper",
            template="Help the user with {task}",
            tags=["agent"],
        )
        
        result = get_prompt("agent-helper")
        
        assert result["found"] is True
        assert result["name"] == "agent-helper"
        assert "task" in result["variables"]
    
    def test_render_prompt_tool(self, registry):
        """Test the render_prompt tool function."""
        from promptflow.integrations.strands import render_prompt, set_registry
        
        set_registry(registry)
        
        registry.register(
            name="greeter",
            template="Hello, {name}! Welcome to {place}.",
        )
        
        result = render_prompt(
            "greeter",
            {"name": "Alice", "place": "Wonderland"}
        )
        
        assert result["success"] is True
        assert "Alice" in result["rendered"]
        assert "Wonderland" in result["rendered"]
    
    def test_toolkit_system_prompt(self, registry):
        """Test getting system prompt through toolkit."""
        from promptflow.integrations.strands import PromptFlowToolkit
        
        registry.register(
            name="agent-system",
            template="You are a helpful {role} assistant.",
        )
        
        toolkit = PromptFlowToolkit(registry=registry)
        system = toolkit.get_system_prompt("agent-system", role="coding")
        
        assert "helpful coding assistant" in system


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
