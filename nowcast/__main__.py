"""
Entry point for running the package as a module.

This allows running the CLI with `python -m nowcast`.
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
