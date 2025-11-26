"""
PromptFlow Dashboard - A PromptLayer-inspired UI for prompt management.

Run with: streamlit run src/promptflow/dashboard/app.py
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

import streamlit as st

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from promptflow.core.models import (
    Prompt,
    PromptExample,
    PromptMetadata,
    PromptType,
    PromptVersion,
    TemplateFormat,
)
from promptflow.core.registry import PromptRegistry
from promptflow.storage.local import LocalStorageBackend


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="PromptFlow",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-dark: #1e1e2e;
        --bg-card: #2d2d3f;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d3f 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
    }
    
    /* Card-like containers */
    .prompt-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .prompt-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
    }
    
    /* Version badges */
    .version-badge {
        display: inline-block;
        background: #6366f1;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .alias-badge {
        display: inline-block;
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    
    /* Tag pills */
    .tag-pill {
        display: inline-block;
        background: #e2e8f0;
        color: #475569;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }
    
    /* Template preview */
    .template-preview {
        background: #1e293b;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.875rem;
        white-space: pre-wrap;
        overflow-x: auto;
    }
    
    /* Variable highlights */
    .var-highlight {
        background: rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 0.1rem 0.3rem;
        border-radius: 4px;
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.875rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Diff styling */
    .diff-added {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 0.2rem 0;
    }
    
    .diff-removed {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        padding: 0.2rem 0;
        text-decoration: line-through;
    }
    
    /* Button overrides */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "registry" not in st.session_state:
        # Use environment config or default to local
        storage_path = os.getenv("PROMPTFLOW_LOCAL_PATH", ".promptflow_dashboard")
        storage = LocalStorageBackend(base_path=storage_path)
        st.session_state.registry = PromptRegistry(storage=storage)
    
    if "current_project" not in st.session_state:
        st.session_state.current_project = "default"
    
    if "selected_prompt" not in st.session_state:
        st.session_state.selected_prompt = None
    
    if "page" not in st.session_state:
        st.session_state.page = "registry"


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================

def get_registry() -> PromptRegistry:
    """Get the registry from session state."""
    return st.session_state.registry


def highlight_variables(template: str, format_type: TemplateFormat) -> str:
    """Highlight variables in template for display."""
    import re
    
    if format_type == TemplateFormat.FSTRING:
        pattern = r"(\{[a-zA-Z_][a-zA-Z0-9_]*\})"
    else:
        pattern = r"(\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\})"
    
    def replacer(match):
        return f'<span class="var-highlight">{match.group(1)}</span>'
    
    return re.sub(pattern, replacer, template)


def format_datetime(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%b %d, %Y at %H:%M")


def render_tags(tags: list[str]) -> str:
    """Render tags as HTML pills."""
    if not tags:
        return ""
    return "".join(f'<span class="tag-pill">{tag}</span>' for tag in tags)


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <span style="font-size: 3rem;">🔮</span>
            <h1 style="margin: 0.5rem 0 0 0; font-size: 1.5rem; color: #e2e8f0;">PromptFlow</h1>
            <p style="color: #94a3b8; font-size: 0.875rem; margin: 0;">Prompt Management Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        st.markdown("### Navigation")
        
        if st.button("📋 Prompt Registry", use_container_width=True, 
                     type="primary" if st.session_state.page == "registry" else "secondary"):
            st.session_state.page = "registry"
            st.session_state.selected_prompt = None
            st.rerun()
        
        if st.button("➕ Create Prompt", use_container_width=True,
                     type="primary" if st.session_state.page == "create" else "secondary"):
            st.session_state.page = "create"
            st.rerun()
        
        if st.button("🔍 Search", use_container_width=True,
                     type="primary" if st.session_state.page == "search" else "secondary"):
            st.session_state.page = "search"
            st.rerun()
        
        if st.button("📊 Analytics", use_container_width=True,
                     type="primary" if st.session_state.page == "analytics" else "secondary"):
            st.session_state.page = "analytics"
            st.rerun()
        
        if st.button("📤 Export / Import", use_container_width=True,
                     type="primary" if st.session_state.page == "export" else "secondary"):
            st.session_state.page = "export"
            st.rerun()
        
        st.divider()
        
        # Project selector
        st.markdown("### Project")
        projects = get_registry().list_projects()
        if not projects:
            projects = ["default"]
        
        selected_project = st.selectbox(
            "Select Project",
            projects,
            index=projects.index(st.session_state.current_project) if st.session_state.current_project in projects else 0,
            label_visibility="collapsed"
        )
        
        if selected_project != st.session_state.current_project:
            st.session_state.current_project = selected_project
            st.session_state.selected_prompt = None
            st.rerun()
        
        # Quick stats
        st.divider()
        st.markdown("### Quick Stats")
        
        prompts = get_registry().list(project=st.session_state.current_project)
        total_versions = sum(p.version_count for p in prompts)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prompts", len(prompts))
        with col2:
            st.metric("Versions", total_versions)
        
        # Storage info
        st.divider()
        st.markdown("### Storage")
        storage_type = os.getenv("PROMPTFLOW_STORAGE", "local")
        st.caption(f"📁 {storage_type.upper()}")
        if storage_type == "s3":
            bucket = os.getenv("PROMPTFLOW_S3_BUCKET", "N/A")
            st.caption(f"🪣 {bucket}")


# =============================================================================
# Main Content Pages
# =============================================================================

def render_registry_page():
    """Render the prompt registry page."""
    st.markdown('<h1 class="main-header">📋 Prompt Registry</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Manage your prompt templates with version control</p>', unsafe_allow_html=True)
    
    # Check if viewing a specific prompt
    if st.session_state.selected_prompt:
        render_prompt_detail()
        return
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_query = st.text_input("🔍 Filter prompts", placeholder="Search by name...")
    
    with col2:
        tag_filter = st.text_input("🏷️ Filter by tags", placeholder="tag1, tag2...")
    
    with col3:
        include_deleted = st.checkbox("Show deleted", value=False)
    
    # Get prompts
    tags = [t.strip() for t in tag_filter.split(",") if t.strip()] if tag_filter else None
    prompts = get_registry().list(
        project=st.session_state.current_project,
        tags=tags,
        include_deleted=include_deleted
    )
    
    # Filter by search query
    if search_query:
        prompts = [p for p in prompts if search_query.lower() in p.name.lower() 
                   or (p.description and search_query.lower() in p.description.lower())]
    
    # Display prompts
    if not prompts:
        st.info("No prompts found. Create your first prompt to get started!")
        if st.button("➕ Create New Prompt"):
            st.session_state.page = "create"
            st.rerun()
        return
    
    # Prompt cards grid
    for prompt in prompts:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Prompt header
                header_html = f"""
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.25rem; font-weight: 600; color: #1e293b;">{prompt.name}</span>
                    <span class="version-badge">v{prompt.latest_version}</span>
                """
                for alias, version in prompt.aliases.items():
                    header_html += f'<span class="alias-badge">{alias} → v{version}</span>'
                
                if prompt.is_deleted:
                    header_html += '<span style="color: #ef4444; margin-left: 0.5rem;">🗑️ Deleted</span>'
                
                header_html += "</div>"
                st.markdown(header_html, unsafe_allow_html=True)
                
                # Description
                if prompt.description:
                    st.caption(prompt.description)
                
                # Tags
                if prompt.tags:
                    st.markdown(render_tags(prompt.tags), unsafe_allow_html=True)
                
                # Preview latest template
                latest = prompt.get_version()
                if latest:
                    preview = latest.template[:150] + "..." if len(latest.template) > 150 else latest.template
                    st.code(preview, language="text")
                
                # Metadata row
                st.caption(f"📅 Updated {format_datetime(prompt.updated_at)} • {prompt.version_count} versions • Variables: {', '.join(latest.variables) if latest else 'none'}")
            
            with col2:
                if st.button("View", key=f"view_{prompt.id}", use_container_width=True):
                    st.session_state.selected_prompt = prompt.id
                    st.rerun()
                
                if st.button("Edit", key=f"edit_{prompt.id}", use_container_width=True):
                    st.session_state.selected_prompt = prompt.id
                    st.session_state.edit_mode = True
                    st.rerun()
            
            st.divider()


def render_prompt_detail():
    """Render the detailed view of a single prompt."""
    prompt = get_registry().get_by_id(st.session_state.selected_prompt)
    
    if not prompt:
        st.error("Prompt not found")
        if st.button("← Back to Registry"):
            st.session_state.selected_prompt = None
            st.rerun()
        return
    
    # Back button
    if st.button("← Back to Registry"):
        st.session_state.selected_prompt = None
        st.rerun()
    
    # Header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f'<h1 class="main-header">{prompt.name}</h1>', unsafe_allow_html=True)
        if prompt.description:
            st.markdown(f'<p class="sub-header">{prompt.description}</p>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Latest Version", f"v{prompt.latest_version}")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Template", "📜 History", "🏷️ Aliases", "⚙️ Settings", "🧪 Playground"])
    
    with tab1:
        render_template_tab(prompt)
    
    with tab2:
        render_history_tab(prompt)
    
    with tab3:
        render_aliases_tab(prompt)
    
    with tab4:
        render_settings_tab(prompt)
    
    with tab5:
        render_playground_tab(prompt)


def render_template_tab(prompt: Prompt):
    """Render the template editing tab."""
    latest = prompt.get_version()
    
    if not latest:
        st.warning("No versions found")
        return
    
    # Version selector
    col1, col2 = st.columns([2, 3])
    
    with col1:
        versions = [v.version for v in prompt.versions]
        selected_version = st.selectbox(
            "Version",
            versions,
            index=versions.index(prompt.latest_version),
            format_func=lambda v: f"v{v}" + (" (latest)" if v == prompt.latest_version else "")
        )
    
    with col2:
        # Show alias if this version has one
        alias_for_version = [a for a, v in prompt.aliases.items() if v == selected_version]
        if alias_for_version:
            st.info(f"🏷️ This version has alias: **{', '.join(alias_for_version)}**")
    
    # Get selected version
    version = prompt.get_version(selected_version)
    
    # Template display
    st.markdown("### Template")
    
    # Editable or view mode
    edit_mode = st.checkbox("Edit mode", value=False)
    
    if edit_mode:
        new_template = st.text_area(
            "Template content",
            value=version.template,
            height=300,
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            change_message = st.text_input("Change message", placeholder="Describe your changes...")
        
        with col2:
            new_format = st.selectbox(
                "Format",
                [TemplateFormat.FSTRING, TemplateFormat.JINJA2],
                index=0 if version.format == TemplateFormat.FSTRING else 1,
                format_func=lambda f: "f-string {var}" if f == TemplateFormat.FSTRING else "Jinja2 {{ var }}"
            )
        
        with col3:
            if st.button("💾 Save New Version", type="primary"):
                try:
                    get_registry().update(
                        prompt.name,
                        template=new_template,
                        project=prompt.project,
                        format=new_format,
                        change_message=change_message or "Updated via dashboard"
                    )
                    st.success(f"Created version {prompt.latest_version + 1}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        # Display with syntax highlighting
        st.code(version.template, language="text")
    
    # Variables
    st.markdown("### Variables")
    if version.variables:
        cols = st.columns(min(len(version.variables), 4))
        for i, var in enumerate(version.variables):
            with cols[i % 4]:
                st.markdown(f"""
                <div style="background: #f1f5f9; padding: 0.5rem 1rem; border-radius: 8px; text-align: center;">
                    <code style="color: #6366f1;">{{{var}}}</code>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("No variables detected in this template")
    
    # Metadata
    st.markdown("### Metadata")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Format:** {version.format.value}")
        st.markdown(f"**Type:** {version.prompt_type.value}")
    
    with col2:
        st.markdown(f"**Model:** {version.metadata.model or 'Not specified'}")
        st.markdown(f"**Temperature:** {version.metadata.temperature if version.metadata.temperature is not None else 'Not specified'}")
    
    with col3:
        st.markdown(f"**Max Tokens:** {version.metadata.max_tokens or 'Not specified'}")
        st.markdown(f"**Hash:** `{version.content_hash[:12]}...`")


def render_history_tab(prompt: Prompt):
    """Render the version history tab."""
    st.markdown("### Version History")
    
    # Timeline view
    for version in sorted(prompt.versions, key=lambda v: v.version, reverse=True):
        alias_badges = "".join(
            f'<span class="alias-badge">{a}</span>' 
            for a, v in prompt.aliases.items() if v == version.version
        )
        
        with st.expander(f"**v{version.version}** {alias_badges} - {format_datetime(version.created_at)}", expanded=version.version == prompt.latest_version):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Change message:** {version.change_message or 'No message'}")
                st.markdown(f"**Created by:** {version.created_by or 'Unknown'}")
                st.markdown(f"**Variables:** {', '.join(version.variables) or 'None'}")
                st.code(version.template[:500] + ("..." if len(version.template) > 500 else ""), language="text")
            
            with col2:
                # Actions
                if version.version != prompt.latest_version:
                    if st.button("🔄 Rollback", key=f"rollback_{version.version}"):
                        try:
                            get_registry().rollback(
                                prompt.name,
                                to_version=version.version,
                                project=prompt.project
                            )
                            st.success(f"Rolled back to v{version.version}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                if prompt.latest_version > 1 and version.version > 1:
                    if st.button("📊 Compare", key=f"compare_{version.version}"):
                        st.session_state.compare_v1 = version.version - 1
                        st.session_state.compare_v2 = version.version
    
    # Version comparison
    if hasattr(st.session_state, 'compare_v1'):
        st.markdown("### Version Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            v1 = st.selectbox("Compare version", [v.version for v in prompt.versions], 
                              index=[v.version for v in prompt.versions].index(st.session_state.compare_v1))
        
        with col2:
            v2 = st.selectbox("With version", [v.version for v in prompt.versions],
                              index=[v.version for v in prompt.versions].index(st.session_state.compare_v2))
        
        if v1 != v2:
            diff = prompt.compare_versions(v1, v2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Version {v1}**")
                st.code(diff["v1_template"], language="text")
            
            with col2:
                st.markdown(f"**Version {v2}**")
                st.code(diff["v2_template"], language="text")
            
            if diff["variables_added"]:
                st.success(f"➕ Variables added: {', '.join(diff['variables_added'])}")
            if diff["variables_removed"]:
                st.error(f"➖ Variables removed: {', '.join(diff['variables_removed'])}")


def render_aliases_tab(prompt: Prompt):
    """Render the aliases management tab."""
    st.markdown("### Release Labels (Aliases)")
    st.caption("Use aliases to reference specific versions without hardcoding version numbers.")
    
    # Current aliases
    if prompt.aliases:
        st.markdown("#### Current Aliases")
        
        for alias, version in prompt.aliases.items():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background: #10b981; color: white; padding: 0.5rem 1rem; border-radius: 8px; display: inline-block;">
                    {alias}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"→ **v{version}**")
            
            with col3:
                new_version = st.selectbox(
                    "Move to",
                    [v.version for v in prompt.versions],
                    index=[v.version for v in prompt.versions].index(version),
                    key=f"alias_version_{alias}",
                    label_visibility="collapsed"
                )
                
                if new_version != version:
                    if st.button("Update", key=f"update_alias_{alias}"):
                        get_registry().set_alias(prompt.name, alias, new_version, prompt.project)
                        st.success(f"Updated {alias} to v{new_version}")
                        st.rerun()
            
            with col4:
                if st.button("🗑️", key=f"remove_alias_{alias}"):
                    get_registry().remove_alias(prompt.name, alias, prompt.project)
                    st.success(f"Removed alias {alias}")
                    st.rerun()
    else:
        st.info("No aliases defined yet")
    
    # Add new alias
    st.markdown("#### Add New Alias")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        new_alias = st.text_input("Alias name", placeholder="e.g., prod, staging, dev")
    
    with col2:
        target_version = st.selectbox(
            "Target version",
            [v.version for v in prompt.versions],
            format_func=lambda v: f"v{v}"
        )
    
    with col3:
        if st.button("Add Alias", type="primary"):
            if new_alias:
                try:
                    get_registry().set_alias(prompt.name, new_alias, target_version, prompt.project)
                    st.success(f"Added alias {new_alias} → v{target_version}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter an alias name")


def render_settings_tab(prompt: Prompt):
    """Render the prompt settings tab."""
    st.markdown("### Prompt Settings")
    
    latest = prompt.get_version()
    
    # Basic info
    st.markdown("#### Basic Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Prompt ID", value=prompt.id, disabled=True)
        st.text_input("Project", value=prompt.project, disabled=True)
    
    with col2:
        st.text_input("Created", value=format_datetime(prompt.created_at), disabled=True)
        st.text_input("Updated", value=format_datetime(prompt.updated_at), disabled=True)
    
    # Tags
    st.markdown("#### Tags")
    current_tags = ", ".join(prompt.tags) if prompt.tags else ""
    new_tags = st.text_input("Tags (comma-separated)", value=current_tags)
    
    # Model configuration
    st.markdown("#### Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model = st.text_input("Model", value=latest.metadata.model or "" if latest else "")
    
    with col2:
        temperature = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=latest.metadata.temperature if latest and latest.metadata.temperature else 0.7,
            step=0.1
        )
    
    with col3:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1,
            max_value=100000,
            value=latest.metadata.max_tokens if latest and latest.metadata.max_tokens else 1000
        )
    
    # Danger zone
    st.markdown("#### Danger Zone")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not prompt.is_deleted:
            if st.button("🗑️ Soft Delete", type="secondary"):
                get_registry().delete(prompt.name, prompt.project, hard=False)
                st.warning("Prompt soft deleted")
                st.session_state.selected_prompt = None
                st.rerun()
        else:
            if st.button("♻️ Restore", type="primary"):
                p = get_registry().get_by_id(prompt.id)
                if p:
                    p.restore()
                    get_registry().storage.save_prompt(p)
                    st.success("Prompt restored!")
                    st.rerun()
    
    with col2:
        if st.button("🗑️ Hard Delete", type="secondary"):
            if st.checkbox("I understand this cannot be undone"):
                get_registry().delete(prompt.name, prompt.project, hard=True)
                st.error("Prompt permanently deleted")
                st.session_state.selected_prompt = None
                st.rerun()


def render_playground_tab(prompt: Prompt):
    """Render the prompt playground/testing tab."""
    st.markdown("### Prompt Playground")
    st.caption("Test your prompt with sample variables")
    
    latest = prompt.get_version()
    
    if not latest:
        st.warning("No versions available")
        return
    
    # Version selector
    selected_version = st.selectbox(
        "Test version",
        [v.version for v in prompt.versions],
        index=len(prompt.versions) - 1,
        format_func=lambda v: f"v{v}" + (" (latest)" if v == prompt.latest_version else "")
    )
    
    version = prompt.get_version(selected_version)
    
    # Variable inputs
    st.markdown("#### Variables")
    
    variables = {}
    if version.variables:
        cols = st.columns(min(len(version.variables), 2))
        for i, var in enumerate(version.variables):
            with cols[i % 2]:
                variables[var] = st.text_area(
                    f"{var}",
                    placeholder=f"Enter value for {var}...",
                    key=f"var_{var}"
                )
    else:
        st.info("This template has no variables")
    
    # Render button
    if st.button("🚀 Render Prompt", type="primary"):
        try:
            rendered = version.render(variables)
            
            st.markdown("#### Rendered Output")
            st.markdown(f"""
            <div class="template-preview">{rendered}</div>
            """, unsafe_allow_html=True)
            
            # Copy button
            st.code(rendered, language="text")
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(rendered))
            with col2:
                st.metric("Words", len(rendered.split()))
            with col3:
                st.metric("Est. Tokens", len(rendered) // 4)
                
        except ValueError as e:
            st.error(f"Rendering error: {e}")


def render_create_page():
    """Render the create new prompt page."""
    st.markdown('<h1 class="main-header">➕ Create New Prompt</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Create a new prompt template with version control</p>', unsafe_allow_html=True)
    
    # Form
    with st.form("create_prompt_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Prompt Name*", placeholder="e.g., summarizer, translator")
            project = st.text_input("Project", value=st.session_state.current_project)
        
        with col2:
            description = st.text_input("Description", placeholder="Brief description of this prompt")
            tags = st.text_input("Tags", placeholder="tag1, tag2, tag3")
        
        st.markdown("### Template")
        
        template = st.text_area(
            "Prompt Template*",
            height=200,
            placeholder="Enter your prompt template here.\nUse {variable_name} for variables.",
            help="Use {variable} for f-string format or {{ variable }} for Jinja2"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            template_format = st.selectbox(
                "Template Format",
                [TemplateFormat.FSTRING, TemplateFormat.JINJA2],
                format_func=lambda f: "f-string {var}" if f == TemplateFormat.FSTRING else "Jinja2 {{ var }}"
            )
        
        with col2:
            prompt_type = st.selectbox(
                "Prompt Type",
                [PromptType.TEMPLATE, PromptType.SYSTEM, PromptType.USER],
                format_func=lambda t: t.value.capitalize()
            )
        
        with col3:
            model = st.text_input("Model (optional)", placeholder="e.g., claude-3-sonnet")
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        
        with col2:
            max_tokens = st.number_input("Max Tokens", min_value=1, value=1000)
        
        submitted = st.form_submit_button("Create Prompt", type="primary", use_container_width=True)
        
        if submitted:
            if not name or not template:
                st.error("Please fill in required fields (Name and Template)")
            else:
                try:
                    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
                    
                    prompt = get_registry().register(
                        name=name,
                        template=template,
                        project=project,
                        description=description if description else None,
                        tags=tag_list,
                        format=template_format,
                        prompt_type=prompt_type,
                        metadata={
                            "model": model if model else None,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        }
                    )
                    
                    st.success(f"✅ Created prompt '{name}' (ID: {prompt.id})")
                    
                    if st.button("View Prompt"):
                        st.session_state.selected_prompt = prompt.id
                        st.session_state.page = "registry"
                        st.rerun()
                        
                except ValueError as e:
                    st.error(f"Error: {e}")


def render_search_page():
    """Render the search page."""
    st.markdown('<h1 class="main-header">🔍 Search Prompts</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find prompts across all projects</p>', unsafe_allow_html=True)
    
    # Search input
    query = st.text_input("Search", placeholder="Search by name, description, or template content...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_project = st.selectbox(
            "Project",
            ["All Projects"] + get_registry().list_projects(),
            index=0
        )
    
    with col2:
        limit = st.slider("Max Results", 5, 50, 20)
    
    if query:
        project_filter = None if search_project == "All Projects" else search_project
        results = get_registry().search(query, project_filter, limit)
        
        st.markdown(f"### Found {len(results)} results")
        
        for prompt in results:
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{prompt.name}** ({prompt.project})")
                    if prompt.description:
                        st.caption(prompt.description)
                    st.markdown(render_tags(prompt.tags), unsafe_allow_html=True)
                
                with col2:
                    if st.button("View", key=f"search_view_{prompt.id}"):
                        st.session_state.selected_prompt = prompt.id
                        st.session_state.page = "registry"
                        st.rerun()
                
                st.divider()


def render_analytics_page():
    """Render the analytics page."""
    st.markdown('<h1 class="main-header">📊 Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Overview of your prompt registry</p>', unsafe_allow_html=True)
    
    # Get all prompts
    all_prompts = get_registry().list(limit=1000)
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">{}</div>
            <div class="stat-label">Total Prompts</div>
        </div>
        """.format(len(all_prompts)), unsafe_allow_html=True)
    
    with col2:
        total_versions = sum(p.version_count for p in all_prompts)
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
            <div class="stat-number">{}</div>
            <div class="stat-label">Total Versions</div>
        </div>
        """.format(total_versions), unsafe_allow_html=True)
    
    with col3:
        projects = set(p.project for p in all_prompts)
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
            <div class="stat-number">{}</div>
            <div class="stat-label">Projects</div>
        </div>
        """.format(len(projects)), unsafe_allow_html=True)
    
    with col4:
        total_aliases = sum(len(p.aliases) for p in all_prompts)
        st.markdown("""
        <div class="stat-card" style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%);">
            <div class="stat-number">{}</div>
            <div class="stat-label">Active Aliases</div>
        </div>
        """.format(total_aliases), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prompts by project
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Prompts by Project")
        project_counts = {}
        for p in all_prompts:
            project_counts[p.project] = project_counts.get(p.project, 0) + 1
        
        if project_counts:
            import pandas as pd
            df = pd.DataFrame(list(project_counts.items()), columns=["Project", "Count"])
            st.bar_chart(df.set_index("Project"))
    
    with col2:
        st.markdown("### Most Used Tags")
        tag_counts = {}
        for p in all_prompts:
            for tag in p.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        if tag_counts:
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            for tag, count in sorted_tags:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: #f1f5f9; margin-bottom: 0.5rem; border-radius: 4px;">
                    <span>{tag}</span>
                    <span style="font-weight: bold;">{count}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("### Recent Activity")
    recent = sorted(all_prompts, key=lambda p: p.updated_at, reverse=True)[:10]
    
    for prompt in recent:
        st.markdown(f"**{prompt.name}** updated {format_datetime(prompt.updated_at)} • v{prompt.latest_version}")


def render_export_page():
    """Render the export/import page."""
    st.markdown('<h1 class="main-header">📤 Export / Import</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Backup and migrate your prompts</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📤 Export", "📥 Import"])
    
    with tab1:
        st.markdown("### Export Prompts")
        
        projects = get_registry().list_projects()
        
        if not projects:
            st.info("No projects to export")
        else:
            selected_project = st.selectbox("Select project to export", projects)
            
            if st.button("Generate Export", type="primary"):
                try:
                    json_export = get_registry().export_json(selected_project)
                    
                    st.success(f"Export generated for project '{selected_project}'")
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_export,
                        file_name=f"promptflow_export_{selected_project}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    with st.expander("Preview export"):
                        st.json(json.loads(json_export))
                        
                except Exception as e:
                    st.error(f"Export error: {e}")
    
    with tab2:
        st.markdown("### Import Prompts")
        
        uploaded_file = st.file_uploader("Upload JSON export file", type=["json"])
        
        overwrite = st.checkbox("Overwrite existing prompts with same ID")
        
        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                data = json.loads(content)
                
                st.markdown("#### Preview")
                st.json(data)
                
                if st.button("Import Prompts", type="primary"):
                    stats = get_registry().import_json(content, overwrite=overwrite)
                    
                    st.success(f"✅ Imported {stats['imported']}/{stats['total']} prompts")
                    
                    if stats['skipped']:
                        st.info(f"Skipped {stats['skipped']} existing prompts")
                    
                    if stats['errors']:
                        st.warning(f"Errors: {len(stats['errors'])}")
                        for err in stats['errors']:
                            st.error(f"{err['prompt_id']}: {err['error']}")
                            
            except json.JSONDecodeError:
                st.error("Invalid JSON file")
            except Exception as e:
                st.error(f"Import error: {e}")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    render_sidebar()
    
    # Route to appropriate page
    if st.session_state.page == "registry":
        render_registry_page()
    elif st.session_state.page == "create":
        render_create_page()
    elif st.session_state.page == "search":
        render_search_page()
    elif st.session_state.page == "analytics":
        render_analytics_page()
    elif st.session_state.page == "export":
        render_export_page()


if __name__ == "__main__":
    main()
