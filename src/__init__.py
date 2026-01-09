"""
Digital Bird Stress Twin Package
Production-grade avian stress behavior modeling system
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Create necessary directories
REQUIRED_DIRS = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "interim",
    PROJECT_ROOT / "data" / "external",
    PROJECT_ROOT / "models" / "checkpoints",
    PROJECT_ROOT / "models" / "registry",
    PROJECT_ROOT / "models" / "exports",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "monitoring" / "reports",
    PROJECT_ROOT / "monitoring" / "evidently_project",
]

for directory in REQUIRED_DIRS:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".gitkeep").touch(exist_ok=True)
