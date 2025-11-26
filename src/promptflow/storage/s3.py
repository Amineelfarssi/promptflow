"""
S3-based storage backend for prompt management.

This implementation stores prompts as JSON files in S3 with the following structure:
    s3://bucket/prefix/
        index.json                    # Global index for fast lookups
        projects/
            {project}/
                index.json            # Project-level index
                prompts/
                    {prompt_id}.json  # Individual prompt files

Features:
- Atomic writes using S3's strong consistency
- Index files for fast listing and searching
- Optimistic locking via ETags for concurrent access
- Gzip compression for storage efficiency
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import boto3
from botocore.exceptions import ClientError

from promptflow.core.models import Prompt, PromptCollection
from promptflow.storage.base import (
    PromptExistsError,
    PromptNotFoundError,
    StorageBackend,
    StorageConnectionError,
    StorageError,
)

logger = logging.getLogger(__name__)


class S3StorageBackend(StorageBackend):
    """
    S3-based storage backend.
    
    Designed for serverless environments where you only have access to S3.
    Works well with Lambda, Fargate, or any compute that can access S3.
    """
    
    GLOBAL_INDEX = "index.json"
    PROJECT_INDEX = "index.json"
    PROMPTS_DIR = "prompts"
    PROJECTS_DIR = "projects"
    
    def __init__(
        self,
        bucket: str,
        prefix: str = "promptflow",
        region: str | None = None,
        endpoint_url: str | None = None,
        compress: bool = True,
        **boto3_kwargs: Any,
    ):
        """
        Initialize S3 storage backend.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all objects (acts like a folder)
            region: AWS region (uses default if not specified)
            endpoint_url: Custom endpoint (for LocalStack, MinIO, etc.)
            compress: Whether to gzip compress stored data
            **boto3_kwargs: Additional arguments for boto3 client
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.compress = compress
        
        session_kwargs = {}
        if region:
            session_kwargs["region_name"] = region
        
        try:
            session = boto3.Session(**session_kwargs)
            client_kwargs = {**boto3_kwargs}
            if endpoint_url:
                client_kwargs["endpoint_url"] = endpoint_url
            
            self.s3 = session.client("s3", **client_kwargs)
            
            # Verify bucket access
            self.s3.head_bucket(Bucket=bucket)
            
        except ClientError as e:
            raise StorageConnectionError(f"Failed to connect to S3 bucket: {e}")
    
    def _key(self, *parts: str) -> str:
        """Construct an S3 key from path parts."""
        return "/".join([self.prefix, *parts])
    
    def _prompt_key(self, prompt_id: str, project: str) -> str:
        """Get the S3 key for a prompt."""
        return self._key(self.PROJECTS_DIR, project, self.PROMPTS_DIR, f"{prompt_id}.json")
    
    def _project_index_key(self, project: str) -> str:
        """Get the S3 key for a project index."""
        return self._key(self.PROJECTS_DIR, project, self.PROJECT_INDEX)
    
    def _read_json(self, key: str) -> dict[str, Any] | None:
        """Read and parse a JSON file from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            body = response["Body"].read()
            
            # Handle gzipped content
            if key.endswith(".gz") or response.get("ContentEncoding") == "gzip":
                body = gzip.decompress(body)
            
            return json.loads(body.decode("utf-8"))
        
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise StorageError(f"Failed to read from S3: {e}")
    
    def _write_json(self, key: str, data: dict[str, Any], etag: str | None = None) -> str:
        """
        Write JSON data to S3.
        
        Returns the new ETag for optimistic locking.
        """
        body = json.dumps(data, default=str, indent=2).encode("utf-8")
        
        put_kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": key,
            "ContentType": "application/json",
        }
        
        if self.compress:
            buffer = BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                gz.write(body)
            body = buffer.getvalue()
            put_kwargs["ContentEncoding"] = "gzip"
        
        put_kwargs["Body"] = body
        
        # Optimistic locking with ETag
        if etag:
            put_kwargs["IfMatch"] = etag
        
        try:
            response = self.s3.put_object(**put_kwargs)
            return response.get("ETag", "")
        
        except ClientError as e:
            if e.response["Error"]["Code"] == "PreconditionFailed":
                raise StorageError("Concurrent modification detected. Please retry.")
            raise StorageError(f"Failed to write to S3: {e}")
    
    def _delete_object(self, key: str) -> bool:
        """Delete an object from S3. Returns True if object existed."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise StorageError(f"Failed to delete from S3: {e}")
    
    def _update_global_index(self, prompt: Prompt, remove: bool = False) -> None:
        """Update the global index with prompt metadata."""
        key = self._key(self.GLOBAL_INDEX)
        index = self._read_json(key) or {"prompts": {}, "updated_at": None}
        
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
        self._write_json(key, index)
    
    def _update_project_index(self, prompt: Prompt, remove: bool = False) -> None:
        """Update the project-level index."""
        key = self._project_index_key(prompt.project)
        index = self._read_json(key) or {"prompts": {}, "updated_at": None}
        
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
        self._write_json(key, index)
    
    def save_prompt(self, prompt: Prompt) -> None:
        """Save a prompt to S3."""
        key = self._prompt_key(prompt.id, prompt.project)
        
        # Save the prompt
        self._write_json(key, prompt.model_dump(mode="json"))
        
        # Update indexes
        self._update_global_index(prompt)
        self._update_project_index(prompt)
        
        logger.info(f"Saved prompt {prompt.id} (v{prompt.latest_version}) to S3")
    
    def get_prompt(self, prompt_id: str) -> Prompt | None:
        """Get a prompt by ID (searches across all projects)."""
        # Check global index first for project info
        global_index = self._read_json(self._key(self.GLOBAL_INDEX))
        
        if not global_index or prompt_id not in global_index.get("prompts", {}):
            return None
        
        prompt_meta = global_index["prompts"][prompt_id]
        project = prompt_meta["project"]
        
        # Get the full prompt
        key = self._prompt_key(prompt_id, project)
        data = self._read_json(key)
        
        if not data:
            return None
        
        return Prompt.model_validate(data)
    
    def get_prompt_by_name(self, name: str, project: str = "default") -> Prompt | None:
        """Get a prompt by name within a project."""
        index = self._read_json(self._project_index_key(project))
        
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
            # Use project index
            index = self._read_json(self._project_index_key(project))
            prompt_ids = list(index.get("prompts", {}).keys()) if index else []
        else:
            # Use global index
            index = self._read_json(self._key(self.GLOBAL_INDEX))
            prompt_ids = list(index.get("prompts", {}).keys()) if index else []
        
        results: list[Prompt] = []
        
        for prompt_id in prompt_ids:
            if len(results) >= offset + limit:
                break
            
            prompt = self.get_prompt(prompt_id)
            if not prompt:
                continue
            
            # Apply filters
            if not include_deleted and prompt.is_deleted:
                continue
            
            if tags and not all(t in prompt.tags for t in tags):
                continue
            
            results.append(prompt)
        
        return results[offset:offset + limit]
    
    def delete_prompt(self, prompt_id: str, hard: bool = False) -> bool:
        """Delete a prompt (soft or hard delete)."""
        prompt = self.get_prompt(prompt_id)
        
        if not prompt:
            return False
        
        if hard:
            # Remove from S3
            key = self._prompt_key(prompt_id, prompt.project)
            self._delete_object(key)
            
            # Remove from indexes
            self._update_global_index(prompt, remove=True)
            self._update_project_index(prompt, remove=True)
            
            logger.info(f"Hard deleted prompt {prompt_id}")
        else:
            # Soft delete
            prompt.soft_delete()
            self.save_prompt(prompt)
            
            logger.info(f"Soft deleted prompt {prompt_id}")
        
        return True
    
    def list_projects(self) -> list[str]:
        """List all projects."""
        global_index = self._read_json(self._key(self.GLOBAL_INDEX))
        
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
        prompts = self.list_prompts(project=project, limit=1000)  # Get more for search
        
        results = []
        for prompt in prompts:
            score = 0
            
            # Search in name
            if query_lower in prompt.name.lower():
                score += 10
            
            # Search in description
            if prompt.description and query_lower in prompt.description.lower():
                score += 5
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in prompt.tags):
                score += 3
            
            # Search in template content (latest version)
            latest = prompt.get_version()
            if latest and query_lower in latest.template.lower():
                score += 2
            
            if score > 0:
                results.append((score, prompt))
        
        # Sort by score and return
        results.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in results[:limit]]
