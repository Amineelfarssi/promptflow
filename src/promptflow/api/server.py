"""
FastAPI REST API for PromptFlow.

Provides a RESTful interface for prompt management, similar to PromptLayer's API.
Can be deployed as a Lambda function, ECS service, or standalone server.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from promptflow.core.models import (
    Prompt,
    PromptExample,
    PromptMetadata,
    PromptType,
    PromptVersion,
    TemplateFormat,
)
from promptflow.core.registry import PromptRegistry
from promptflow.storage.base import PromptNotFoundError

# Initialize API
app = FastAPI(
    title="PromptFlow API",
    description="Prompt management and versioning API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global registry instance
_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    """Get or create the registry instance."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry.from_env()
    return _registry


# =============================================================================
# Request/Response Models
# =============================================================================


class PromptCreate(BaseModel):
    """Request body for creating a prompt."""
    name: str
    template: str
    project: str = "default"
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    format: TemplateFormat = TemplateFormat.FSTRING
    prompt_type: PromptType = PromptType.TEMPLATE
    metadata: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = Field(default_factory=list)
    aliases: dict[str, int] | None = None


class PromptUpdate(BaseModel):
    """Request body for updating a prompt."""
    template: str
    format: TemplateFormat | None = None
    prompt_type: PromptType | None = None
    metadata: dict[str, Any] | None = None
    examples: list[dict[str, Any]] | None = None
    change_message: str | None = None


class AliasSet(BaseModel):
    """Request body for setting an alias."""
    alias: str
    version: int


class RenderRequest(BaseModel):
    """Request body for rendering a prompt."""
    name: str
    project: str = "default"
    version: int | None = None
    alias: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)


class RenderResponse(BaseModel):
    """Response for prompt rendering."""
    rendered: str
    version: int
    variables_used: list[str]


class CompareRequest(BaseModel):
    """Request body for version comparison."""
    name: str
    project: str = "default"
    v1: int
    v2: int


class RollbackRequest(BaseModel):
    """Request body for rollback."""
    to_version: int
    alias: str | None = None


class ImportRequest(BaseModel):
    """Request body for import."""
    data: dict[str, Any]
    overwrite: bool = False


class PromptSummary(BaseModel):
    """Lightweight prompt summary for listings."""
    id: str
    name: str
    project: str
    description: str | None
    tags: list[str]
    latest_version: int
    aliases: dict[str, int]
    is_deleted: bool
    updated_at: str


class VersionSummary(BaseModel):
    """Lightweight version summary."""
    version: int
    content_hash: str
    variables: list[str]
    created_at: str
    created_by: str | None
    change_message: str | None


# =============================================================================
# Health & Info
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def api_info():
    """API information."""
    return {
        "name": "PromptFlow API",
        "version": "1.0.0",
        "storage": "configured",
    }


# =============================================================================
# Projects
# =============================================================================


@app.get("/api/v1/projects", response_model=list[str])
async def list_projects():
    """List all projects."""
    return get_registry().list_projects()


@app.get("/api/v1/projects/{project}/prompts")
async def list_project_prompts(
    project: str = PathParam(..., description="Project name"),
    tags: str | None = Query(None, description="Comma-separated tags"),
    include_deleted: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[PromptSummary]:
    """List prompts in a project."""
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    
    prompts = get_registry().list(
        project=project,
        tags=tag_list,
        include_deleted=include_deleted,
        limit=limit,
        offset=offset,
    )
    
    return [
        PromptSummary(
            id=p.id,
            name=p.name,
            project=p.project,
            description=p.description,
            tags=p.tags,
            latest_version=p.latest_version,
            aliases=p.aliases,
            is_deleted=p.is_deleted,
            updated_at=p.updated_at.isoformat(),
        )
        for p in prompts
    ]


# =============================================================================
# Prompts CRUD
# =============================================================================


@app.get("/api/v1/prompts")
async def list_prompts(
    project: str | None = Query(None, description="Filter by project"),
    tags: str | None = Query(None, description="Comma-separated tags"),
    include_deleted: bool = Query(False),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[PromptSummary]:
    """List all prompts with optional filtering."""
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    
    prompts = get_registry().list(
        project=project,
        tags=tag_list,
        include_deleted=include_deleted,
        limit=limit,
        offset=offset,
    )
    
    return [
        PromptSummary(
            id=p.id,
            name=p.name,
            project=p.project,
            description=p.description,
            tags=p.tags,
            latest_version=p.latest_version,
            aliases=p.aliases,
            is_deleted=p.is_deleted,
            updated_at=p.updated_at.isoformat(),
        )
        for p in prompts
    ]


@app.post("/api/v1/prompts", status_code=201)
async def create_prompt(body: PromptCreate) -> Prompt:
    """Create a new prompt."""
    try:
        return get_registry().register(
            name=body.name,
            template=body.template,
            project=body.project,
            description=body.description,
            tags=body.tags,
            format=body.format,
            prompt_type=body.prompt_type,
            metadata=body.metadata,
            examples=[PromptExample(**ex) for ex in body.examples] if body.examples else None,
            aliases=body.aliases,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/prompts/{prompt_id}")
async def get_prompt_by_id(
    prompt_id: str = PathParam(..., description="Prompt ID"),
) -> Prompt:
    """Get a prompt by ID."""
    prompt = get_registry().get_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@app.get("/api/v1/prompts/by-name/{name}")
async def get_prompt_by_name(
    name: str = PathParam(..., description="Prompt name"),
    project: str = Query("default", description="Project name"),
) -> Prompt:
    """Get a prompt by name."""
    prompt = get_registry().get_prompt(name, project)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt


@app.put("/api/v1/prompts/{prompt_id}")
async def update_prompt(
    body: PromptUpdate,
    prompt_id: str = PathParam(..., description="Prompt ID"),
) -> VersionSummary:
    """Update a prompt (creates new version)."""
    prompt = get_registry().get_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    try:
        new_version = get_registry().update(
            name=prompt.name,
            template=body.template,
            project=prompt.project,
            format=body.format,
            prompt_type=body.prompt_type,
            metadata=body.metadata,
            examples=[PromptExample(**ex) for ex in body.examples] if body.examples else None,
            change_message=body.change_message,
        )
        
        return VersionSummary(
            version=new_version.version,
            content_hash=new_version.content_hash,
            variables=new_version.variables,
            created_at=new_version.created_at.isoformat(),
            created_by=new_version.created_by,
            change_message=new_version.change_message,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/prompts/{prompt_id}")
async def delete_prompt(
    prompt_id: str = PathParam(..., description="Prompt ID"),
    hard: bool = Query(False, description="Permanently delete"),
) -> dict[str, bool]:
    """Delete a prompt."""
    deleted = get_registry().storage.delete_prompt(prompt_id, hard=hard)
    if not deleted:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"deleted": True}


# =============================================================================
# Aliases
# =============================================================================


@app.post("/api/v1/prompts/{prompt_id}/alias")
async def set_prompt_alias(
    body: AliasSet,
    prompt_id: str = PathParam(..., description="Prompt ID"),
) -> dict[str, Any]:
    """Set an alias for a prompt."""
    prompt = get_registry().get_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    try:
        get_registry().set_alias(prompt.name, body.alias, body.version, prompt.project)
        return {"alias": body.alias, "version": body.version}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/prompts/{prompt_id}/alias/{alias}")
async def remove_prompt_alias(
    prompt_id: str = PathParam(..., description="Prompt ID"),
    alias: str = PathParam(..., description="Alias name"),
) -> dict[str, bool]:
    """Remove an alias from a prompt."""
    prompt = get_registry().get_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    removed = get_registry().remove_alias(prompt.name, alias, prompt.project)
    return {"removed": removed}


@app.get("/api/v1/prompts/alias/{alias}")
async def get_by_alias(
    alias: str = PathParam(..., description="Alias name"),
    name: str = Query(..., description="Prompt name"),
    project: str = Query("default", description="Project name"),
) -> PromptVersion:
    """Get a prompt version by alias."""
    version = get_registry().get(name, project, alias=alias)
    if not version:
        raise HTTPException(status_code=404, detail="Alias not found")
    return version


# =============================================================================
# Rendering
# =============================================================================


@app.post("/api/v1/prompts/render")
async def render_prompt(body: RenderRequest) -> RenderResponse:
    """Render a prompt with variables."""
    try:
        version = get_registry().get(
            body.name,
            body.project,
            body.version,
            body.alias,
        )
        
        if not version:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        rendered = version.render(body.variables)
        
        return RenderResponse(
            rendered=rendered,
            version=version.version,
            variables_used=version.variables,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Version History
# =============================================================================


@app.get("/api/v1/prompts/{prompt_id}/history")
async def get_version_history(
    prompt_id: str = PathParam(..., description="Prompt ID"),
    limit: int = Query(10, ge=1, le=100),
) -> list[VersionSummary]:
    """Get version history for a prompt."""
    prompt = get_registry().get_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    versions = get_registry().history(prompt.name, prompt.project, limit)
    
    return [
        VersionSummary(
            version=v.version,
            content_hash=v.content_hash,
            variables=v.variables,
            created_at=v.created_at.isoformat(),
            created_by=v.created_by,
            change_message=v.change_message,
        )
        for v in versions
    ]


@app.get("/api/v1/prompts/{prompt_id}/versions/{version}")
async def get_specific_version(
    prompt_id: str = PathParam(..., description="Prompt ID"),
    version: int = PathParam(..., description="Version number"),
) -> PromptVersion:
    """Get a specific version of a prompt."""
    prompt = get_registry().get_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    v = prompt.get_version(version)
    if not v:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return v


@app.post("/api/v1/prompts/compare")
async def compare_versions(body: CompareRequest) -> dict[str, Any]:
    """Compare two versions of a prompt."""
    try:
        return get_registry().compare(body.name, body.v1, body.v2, body.project)
    except PromptNotFoundError:
        raise HTTPException(status_code=404, detail="Prompt not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/prompts/{prompt_id}/rollback")
async def rollback_version(
    body: RollbackRequest,
    prompt_id: str = PathParam(..., description="Prompt ID"),
) -> VersionSummary:
    """Rollback to a previous version."""
    prompt = get_registry().get_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    
    try:
        new_version = get_registry().rollback(
            name=prompt.name,
            to_version=body.to_version,
            project=prompt.project,
            alias=body.alias,
        )
        
        return VersionSummary(
            version=new_version.version,
            content_hash=new_version.content_hash,
            variables=new_version.variables,
            created_at=new_version.created_at.isoformat(),
            created_by=new_version.created_by,
            change_message=new_version.change_message,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Search
# =============================================================================


@app.get("/api/v1/search")
async def search_prompts(
    q: str = Query(..., description="Search query"),
    project: str | None = Query(None, description="Filter by project"),
    limit: int = Query(20, ge=1, le=100),
) -> list[PromptSummary]:
    """Search for prompts."""
    prompts = get_registry().search(q, project, limit)
    
    return [
        PromptSummary(
            id=p.id,
            name=p.name,
            project=p.project,
            description=p.description,
            tags=p.tags,
            latest_version=p.latest_version,
            aliases=p.aliases,
            is_deleted=p.is_deleted,
            updated_at=p.updated_at.isoformat(),
        )
        for p in prompts
    ]


# =============================================================================
# Export/Import
# =============================================================================


@app.get("/api/v1/export")
async def export_project(
    project: str = Query(..., description="Project to export"),
) -> dict[str, Any]:
    """Export a project."""
    collection = get_registry().export_project(project)
    return collection.to_export()


@app.post("/api/v1/import")
async def import_prompts(body: ImportRequest) -> dict[str, Any]:
    """Import prompts."""
    from promptflow.core.models import PromptCollection
    
    try:
        collection = PromptCollection.from_import(body.data)
        return get_registry().import_collection(collection, body.overwrite)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Run Server
# =============================================================================


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
