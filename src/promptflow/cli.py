"""
Command-line interface for PromptFlow.

Usage:
    promptflow init                     # Initialize local storage
    promptflow register <name> ...      # Register a new prompt
    promptflow get <name>               # Get a prompt
    promptflow list                     # List prompts
    promptflow render <name> ...        # Render a prompt
    promptflow export <project>         # Export project
    promptflow import <file>            # Import prompts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from promptflow.core.models import PromptCollection, TemplateFormat
from promptflow.core.registry import PromptRegistry

app = typer.Typer(
    name="promptflow",
    help="Prompt management and versioning system",
    no_args_is_help=True,
)

console = Console()


def get_registry() -> PromptRegistry:
    """Get registry from environment configuration."""
    return PromptRegistry.from_env()


@app.command()
def init(
    path: str = typer.Option(".promptflow", "--path", "-p", help="Storage path"),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="S3 bucket for remote storage"),
    s3_prefix: str = typer.Option("promptflow", "--s3-prefix", help="S3 key prefix"),
):
    """Initialize PromptFlow storage."""
    if s3_bucket:
        # Test S3 connection
        try:
            registry = PromptRegistry.from_s3(bucket=s3_bucket, prefix=s3_prefix)
            rprint(f"[green]✓[/green] Initialized S3 storage: s3://{s3_bucket}/{s3_prefix}")
        except Exception as e:
            rprint(f"[red]✗[/red] Failed to connect to S3: {e}")
            raise typer.Exit(1)
    else:
        # Local storage
        from promptflow.storage.local import LocalStorageBackend
        storage = LocalStorageBackend(base_path=path)
        rprint(f"[green]✓[/green] Initialized local storage at: {path}")
    
    rprint("\n[dim]Set environment variables to configure storage:[/dim]")
    rprint("  PROMPTFLOW_STORAGE=s3|local")
    rprint("  PROMPTFLOW_S3_BUCKET=your-bucket")
    rprint("  PROMPTFLOW_LOCAL_PATH=.promptflow")


@app.command()
def register(
    name: str = typer.Argument(..., help="Prompt name"),
    template: str = typer.Option(..., "--template", "-t", help="Prompt template"),
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Description"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    format: str = typer.Option("fstring", "--format", "-f", help="Template format: fstring, jinja2"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name"),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="Temperature"),
):
    """Register a new prompt."""
    registry = get_registry()
    
    tag_list = [t.strip() for t in tags.split(",")] if tags else []
    
    try:
        template_format = TemplateFormat(format)
    except ValueError:
        rprint(f"[red]Invalid format: {format}. Use fstring or jinja2[/red]")
        raise typer.Exit(1)
    
    metadata = {}
    if model:
        metadata["model"] = model
    if temperature:
        metadata["temperature"] = temperature
    
    try:
        prompt = registry.register(
            name=name,
            template=template,
            project=project,
            description=description,
            tags=tag_list,
            format=template_format,
            metadata=metadata if metadata else None,
        )
        
        rprint(Panel(
            f"[green]✓ Registered prompt:[/green] {name}\n"
            f"  ID: {prompt.id}\n"
            f"  Project: {project}\n"
            f"  Version: 1\n"
            f"  Variables: {prompt.get_version().variables}",
            title="Success"
        ))
    except ValueError as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def get(
    name: str = typer.Argument(..., help="Prompt name"),
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    version: Optional[int] = typer.Option(None, "--version", "-v", help="Version number"),
    alias: Optional[str] = typer.Option(None, "--alias", "-a", help="Alias to use"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Get a prompt."""
    registry = get_registry()
    
    prompt_version = registry.get(name, project, version, alias)
    
    if not prompt_version:
        rprint(f"[red]Prompt not found: {name}[/red]")
        raise typer.Exit(1)
    
    if json_output:
        print(prompt_version.model_dump_json(indent=2))
    else:
        syntax = Syntax(prompt_version.template, "text", theme="monokai", word_wrap=True)
        
        rprint(Panel(syntax, title=f"{name} v{prompt_version.version}"))
        rprint(f"[dim]Format: {prompt_version.format.value}[/dim]")
        rprint(f"[dim]Variables: {', '.join(prompt_version.variables) or 'none'}[/dim]")
        rprint(f"[dim]Hash: {prompt_version.content_hash}[/dim]")
        
        if prompt_version.metadata.model:
            rprint(f"[dim]Model: {prompt_version.metadata.model}[/dim]")


@app.command("list")
def list_prompts(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Filter by comma-separated tags"),
    include_deleted: bool = typer.Option(False, "--deleted", help="Include deleted prompts"),
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List prompts."""
    registry = get_registry()
    
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    
    prompts = registry.list(
        project=project,
        tags=tag_list,
        include_deleted=include_deleted,
        limit=limit,
    )
    
    if json_output:
        data = [p.model_dump(mode="json") for p in prompts]
        print(json.dumps(data, indent=2))
        return
    
    if not prompts:
        rprint("[dim]No prompts found[/dim]")
        return
    
    table = Table(title="Prompts")
    table.add_column("Name", style="cyan")
    table.add_column("Project", style="magenta")
    table.add_column("Version", justify="right")
    table.add_column("Tags")
    table.add_column("Aliases")
    table.add_column("Updated", style="dim")
    
    for prompt in prompts:
        aliases = ", ".join(f"{a}→v{v}" for a, v in prompt.aliases.items())
        tags_str = ", ".join(prompt.tags[:3])
        if len(prompt.tags) > 3:
            tags_str += "..."
        
        table.add_row(
            prompt.name,
            prompt.project,
            str(prompt.latest_version),
            tags_str,
            aliases or "-",
            prompt.updated_at.strftime("%Y-%m-%d %H:%M"),
        )
    
    console.print(table)


@app.command()
def render(
    name: str = typer.Argument(..., help="Prompt name"),
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    version: Optional[int] = typer.Option(None, "--version", "-v", help="Version"),
    alias: Optional[str] = typer.Option(None, "--alias", "-a", help="Alias"),
    vars: Optional[str] = typer.Option(None, "--vars", help="JSON variables"),
):
    """Render a prompt with variables."""
    registry = get_registry()
    
    variables = {}
    if vars:
        try:
            variables = json.loads(vars)
        except json.JSONDecodeError as e:
            rprint(f"[red]Invalid JSON for --vars: {e}[/red]")
            raise typer.Exit(1)
    
    try:
        rendered = registry.render(name, project, version, alias, **variables)
        rprint(Panel(rendered, title="Rendered Prompt"))
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def update(
    name: str = typer.Argument(..., help="Prompt name"),
    template: str = typer.Option(..., "--template", "-t", help="New template"),
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Change message"),
):
    """Update a prompt (creates new version)."""
    registry = get_registry()
    
    try:
        new_version = registry.update(
            name=name,
            template=template,
            project=project,
            change_message=message,
        )
        
        rprint(f"[green]✓ Updated {name} to version {new_version.version}[/green]")
        rprint(f"[dim]Variables: {new_version.variables}[/dim]")
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("set-alias")
def set_alias(
    name: str = typer.Argument(..., help="Prompt name"),
    alias: str = typer.Argument(..., help="Alias name"),
    version: int = typer.Argument(..., help="Version number"),
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
):
    """Set an alias to point to a version."""
    registry = get_registry()
    
    try:
        registry.set_alias(name, alias, version, project)
        rprint(f"[green]✓ Set alias '{alias}' → v{version} for {name}[/green]")
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def history(
    name: str = typer.Argument(..., help="Prompt name"),
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of versions"),
):
    """Show version history for a prompt."""
    registry = get_registry()
    
    versions = registry.history(name, project, limit)
    
    if not versions:
        rprint(f"[red]Prompt not found: {name}[/red]")
        raise typer.Exit(1)
    
    prompt = registry.get_prompt(name, project)
    alias_map = {v: a for a, v in prompt.aliases.items()} if prompt else {}
    
    table = Table(title=f"Version History: {name}")
    table.add_column("Version", justify="right", style="cyan")
    table.add_column("Alias", style="yellow")
    table.add_column("Created", style="dim")
    table.add_column("By")
    table.add_column("Message")
    table.add_column("Hash", style="dim")
    
    for v in versions:
        alias = alias_map.get(v.version, "")
        table.add_row(
            str(v.version),
            alias,
            v.created_at.strftime("%Y-%m-%d %H:%M"),
            v.created_by or "-",
            (v.change_message or "-")[:40],
            v.content_hash[:8],
        )
    
    console.print(table)


@app.command()
def compare(
    name: str = typer.Argument(..., help="Prompt name"),
    v1: int = typer.Argument(..., help="First version"),
    v2: int = typer.Argument(..., help="Second version"),
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
):
    """Compare two versions of a prompt."""
    registry = get_registry()
    
    try:
        diff = registry.compare(name, v1, v2, project)
        
        rprint(Panel(f"Comparing v{v1} → v{v2}", title=name))
        
        if diff["template_changed"]:
            rprint("\n[yellow]Template changed[/yellow]")
            rprint("[red]- Old:[/red]")
            rprint(Syntax(diff["v1_template"], "text", theme="monokai"))
            rprint("[green]+ New:[/green]")
            rprint(Syntax(diff["v2_template"], "text", theme="monokai"))
        else:
            rprint("[dim]Template unchanged[/dim]")
        
        if diff["variables_added"]:
            rprint(f"[green]+ Variables added: {diff['variables_added']}[/green]")
        if diff["variables_removed"]:
            rprint(f"[red]- Variables removed: {diff['variables_removed']}[/red]")
        
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    project: str = typer.Argument(..., help="Project to export"),
    output: str = typer.Option("export.json", "--output", "-o", help="Output file"),
):
    """Export a project to JSON."""
    registry = get_registry()
    
    try:
        json_str = registry.export_json(project)
        
        Path(output).write_text(json_str)
        rprint(f"[green]✓ Exported project '{project}' to {output}[/green]")
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("import")
def import_prompts(
    file: str = typer.Argument(..., help="JSON file to import"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing prompts"),
):
    """Import prompts from JSON."""
    registry = get_registry()
    
    try:
        json_str = Path(file).read_text()
        stats = registry.import_json(json_str, overwrite)
        
        rprint(f"[green]✓ Imported {stats['imported']}/{stats['total']} prompts[/green]")
        if stats["skipped"]:
            rprint(f"[yellow]Skipped {stats['skipped']} existing prompts[/yellow]")
        if stats["errors"]:
            rprint(f"[red]Errors: {len(stats['errors'])}[/red]")
            for err in stats["errors"]:
                rprint(f"  - {err['prompt_id']}: {err['error']}")
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def projects():
    """List all projects."""
    registry = get_registry()
    
    project_list = registry.list_projects()
    
    if not project_list:
        rprint("[dim]No projects found[/dim]")
        return
    
    for p in project_list:
        prompts = registry.list(project=p)
        rprint(f"[cyan]{p}[/cyan] ({len(prompts)} prompts)")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results"),
):
    """Search for prompts."""
    registry = get_registry()
    
    results = registry.search(query, project, limit)
    
    if not results:
        rprint(f"[dim]No results for '{query}'[/dim]")
        return
    
    rprint(f"Found {len(results)} results for '{query}':\n")
    
    for prompt in results:
        rprint(f"[cyan]{prompt.name}[/cyan] ({prompt.project})")
        if prompt.description:
            rprint(f"  [dim]{prompt.description[:60]}...[/dim]")
        rprint()


if __name__ == "__main__":
    app()
