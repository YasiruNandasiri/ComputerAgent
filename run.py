"""Entry point — run with: uv run run.py [args]

Examples:
  uv run run.py run "take a screenshot"
  uv run run.py chat
  uv run run.py --help
"""

import sys
import os

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from computer_agent.main import app

if __name__ == "__main__":
    app()
