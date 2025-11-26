"""
Local filesystem storage backend for development and testing.

Mirrors the S3 structure using local files, making it easy to
develop and test without AWS access.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from promptflow.core.models import Prompt, PromptCollection
from promptflow.storage.base import (
    PromptExistsError,
    PromptNotFoundError,
    StorageBackend,
    StorageError,
)

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend.
    
    Useful for:
    - Local development without AWS
    - Testing in CI/CD pipelines
    - Air-gapped environments
    - Quick prototyping
    
    Structure mirrors S3 backend:
        {base_path}/
            index.json
            projects/
                {project}/
                    index.json
                    prompts/
                        {prompt_id}.json
    """
    
    GLOBAL_INDEX = "index.json"
    PROJECT_INDEX = "index.json"
    PROMPTS_DIR = "prompts"
    PROJECTS_DIR = "projects"
    
    def __init__(
        self,
        base_path: str | Path = ".promptflow",
        compress: bool = False,
    ):
        """
        Initialize local storage backend.
        
        Args:
            base_path: Base directory for storing prompts
            compress: Whether to gzip compress stored data
        """
        self.base_path = Path(base_path).expanduser().resolve()
        self.compress = compress
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized local storage at {self.base_path}")
    
    def _path(self, *parts: str) -> Path:
        """Construct a file path from parts."""
        return self.base_path.joinpath(*parts)
    
    def _prompt_path(self, prompt_id: str, project: str) -> Path:
        """Get the path for a prompt file."""
        return self._path(self.PROJECTS_DIR, project, self.PROMPTS_DIR, f"{prompt_id}.json")
    
    def _project_index_path(self, project: str) -> Path:
        """Get the path for a project index."""
        return self._path(self.PROJECTS_DIR, project, self.PROJECT_INDEX)
    
    def _read_json(self, path: Path) -> dict[str, Any] | None:
        """Read and parse a JSON file."""
        if not path.exists():
            return None
        
        try:
            with open(path, "rb") as f:
                data = f.read()
            
            # Handle gzipped content
            if path.suffix == ".gz" or self.compress:
                try:
                    data = gzip.decompress(data)
                except gzip.BadGzipFile:
                    pass  # Not compressed
            
            return json.loads(data.decode("utf-8"))
        
        except Exception as e:
            raise StorageError(f"Failed to read {path}: {e}")
    
    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON data to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        content = json.dumps(data, default=str, indent=2).encode("utf-8")
        
        if self.compress:
            content = gzip.compress(content)
        
        # Atomic write via temp file
        temp_path = path.with_suffix(".tmp")
        try:
            with open(temp_path, "wb") as f:
                f.write(content)
            temp_path.replace(path)
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise StorageError(f"Failed to write {path}: {e}")
    
    def _delete_path(self, path: Path) -> bool:
        """Delete a file. Returns True if it existed."""
        if path.exists():
            path.unlink()
            return True
        return False
    
    def _update_global_index(self, prompt: Prompt, remove: bool = False) -> None:
        """Update the global index."""
        path = self._path(self.GLOBAL_INDEX)
        index = self._read_json(path) or {"prompts": {}, "updated_at": None}
        
        if remove:
            index["prompts"].pop(prompt.id, None)
        else:
            index["prompts"][prompt.id] = {
                "id": prompt.id,
                "name": prompt.name,
                "project": prompt.project,
                "tags": prompt.tags,
                "latest_version": prompt.latest_version,
                "is_deleted": prompt.is_deleted,
                "updated_at": prompt.updated_at.isoformat(),
            }
        
        index["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_json(path, index)
    
    def _update_project_index(self, prompt: Prompt, remove: bool = False) -> None:
        """Update the project-level index."""
        path = self._project_index_path(prompt.project)
        index = self._read_json(path) or {"prompts": {}, "updated_at": None}
        
        if remove:
            index["prompts"].pop(prompt.id, None)
        else:
            index["prompts"][prompt.id] = {
                "id": prompt.id,
                "name": prompt.name,
                "tags": prompt.tags,
                "aliases": list(prompt.aliases.keys()),
                "latest_version": prompt.latest_version,
                "is_deleted": prompt.is_deleted,
                "updated_at": prompt.updated_at.isoformat(),
            }
        
        index["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_json(path, index)
    
    def save_prompt(self, prompt: Prompt) -> None:
        """Save a prompt to local storage."""
        path = self._prompt_path(prompt.id, prompt.project)
        self._write_json(path, prompt.model_dump(mode="json"))
        
        self._update_global_index(prompt)
        self._update_project_index(prompt)
        
        logger.info(f"Saved prompt {prompt.id} (v{prompt.latest_version})")
    
    def get_prompt(self, prompt_id: str) -> Prompt | None:
        """Get a prompt by ID."""
        global_index = self._read_json(self._path(self.GLOBAL_INDEX))
        
        if not global_index or prompt_id not in global_index.get("prompts", {}):
            return None
        
        project = global_index["prompts"][prompt_id]["project"]
        path = self._prompt_path(prompt_id, project)
        data = self._read_json(path)
        
        if not data:
            return None
        
        return Prompt.model_validate(data)
    
    def get_prompt_by_name(self, name: str, project: str = "default") -> Prompt | None:
        """Get a prompt by name within a project."""
        index = self._read_json(self._project_index_path(project))
        
        if not index:
            return None
        
        for prompt_id, meta in index.get("prompts", {}).items():
            if meta["name"] == name and not meta.get("is_deleted", False):
                return self.get_prompt(prompt_id)
        
        return None
    
    def list_prompts(
        self,
        project: str | None = None,
        tags: list[str] | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Prompt]:
        """List prompts with filtering."""
        if project:
            index = self._read_json(self._project_index_path(project))
            prompt_ids = list(index.get("prompts", {}).keys()) if index else []
        else:
            index = self._read_json(self._path(self.GLOBAL_INDEX))
            prompt_ids = list(index.get("prompts", {}).keys()) if index else []
        
        results: list[Prompt] = []
        
        for prompt_id in prompt_ids:
            if len(results) >= offset + limit:
                break
            
            prompt = self.get_prompt(prompt_id)
            if not prompt:
                continue
            
            if not include_deleted and prompt.is_deleted:
                continue
            
            if tags and not all(t in prompt.tags for t in tags):
                continue
            
            results.append(prompt)
        
        return results[offset:offset + limit]
    
    def delete_prompt(self, prompt_id: str, hard: bool = False) -> bool:
        """Delete a prompt."""
        prompt = self.get_prompt(prompt_id)
        
        if not prompt:
            return False
        
        if hard:
            path = self._prompt_path(prompt_id, prompt.project)
            self._delete_path(path)
            self._update_global_index(prompt, remove=True)
            self._update_project_index(prompt, remove=True)
            logger.info(f"Hard deleted prompt {prompt_id}")
        else:
            prompt.soft_delete()
            self.save_prompt(prompt)
            logger.info(f"Soft deleted prompt {prompt_id}")
        
        return True
    
    def list_projects(self) -> list[str]:
        """List all projects."""
        global_index = self._read_json(self._path(self.GLOBAL_INDEX))
        
        if not global_index:
            return []
        
        projects = set()
        for meta in global_index.get("prompts", {}).values():
            projects.add(meta["project"])
        
        return sorted(projects)
    
    def export_project(self, project: str) -> PromptCollection:
        """Export all prompts in a project."""
        prompts = self.list_prompts(project=project, include_deleted=True)
        
        return PromptCollection(
            name=project,
            description=f"Export of project '{project}'",
            prompts=prompts,
        )
    
    def import_collection(
        self,
        collection: PromptCollection,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Import a collection of prompts."""
        stats = {
            "total": len(collection.prompts),
            "imported": 0,
            "skipped": 0,
            "errors": [],
        }
        
        for prompt in collection.prompts:
            try:
                existing = self.get_prompt(prompt.id)
                
                if existing and not overwrite:
                    stats["skipped"] += 1
                    continue
                
                self.save_prompt(prompt)
                stats["imported"] += 1
                
            except Exception as e:
                stats["errors"].append({
                    "prompt_id": prompt.id,
                    "error": str(e),
                })
        
        return stats
    
    def search(
        self,
        query: str,
        project: str | None = None,
        limit: int = 20,
    ) -> list[Prompt]:
        """Simple text search across prompts."""
        query_lower = query.lower()
        prompts = self.list_prompts(project=project, limit=1000)
        
        results = []
        for prompt in prompts:
            score = 0
            
            if query_lower in prompt.name.lower():
                score += 10
            
            if prompt.description and query_lower in prompt.description.lower():
                score += 5
            
            if any(query_lower in tag.lower() for tag in prompt.tags):
                score += 3
            
            latest = prompt.get_version()
            if latest and query_lower in latest.template.lower():
                score += 2
            
            if score > 0:
                results.append((score, prompt))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in results[:limit]]
    
    def clear(self) -> None:
        """Clear all data. Use with caution!"""
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)
        logger.warning("Cleared all local storage data")
