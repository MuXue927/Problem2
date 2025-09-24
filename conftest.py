import sys
from pathlib import Path


def pytest_sessionstart(session):
    """Ensure repository root is on sys.path before collection."""
    repo_root = Path(__file__).parent.resolve()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
