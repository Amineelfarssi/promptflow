"""
PromptFlow Dashboard - Streamlit UI for prompt management.

Run with:
    streamlit run -m promptflow.dashboard.app
    
Or:
    cd src/promptflow/dashboard && streamlit run app.py
"""

from pathlib import Path

APP_PATH = Path(__file__).parent / "app.py"

def run():
    """Run the Streamlit dashboard."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)])

__all__ = ["run", "APP_PATH"]
