"""ALNSCode package initializer.

This file makes the ALNSCode directory a Python package so it can be
installed/imported with standard tools (e.g., pip install -e .).
"""

__all__ = ["InputDataALNS"]

try:
    from .InputDataALNS import DataALNS  # re-export common class for convenience
except Exception:
    # Import errors are allowed here (e.g., missing optional dependencies or files)
    pass
