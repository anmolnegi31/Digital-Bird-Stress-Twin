"""
Utility functions and helper classes for the Digital Bird Stress Twin project
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
import sys


class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self._config.copy()


class SpeciesConfig:
    """Species-specific configuration manager"""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "species_config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get_species_info(self, species_key: str) -> Dict[str, Any]:
        """Get information for a specific species"""
        return self._config.get('species', {}).get(species_key, {})
    
    def get_disaster_pattern(self, disaster_type: str) -> Dict[str, Any]:
        """Get disaster pattern information"""
        return self._config.get('disaster_patterns', {}).get(disaster_type, {})


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "500 MB",
    retention: str = "10 days"
) -> None:
    """
    Setup logger with console and file handlers
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        rotation: Log rotation size
        retention: Log retention period
    """
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    logger.info(f"Logger initialized with level: {log_level}")


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.debug(f"Loaded JSON from {filepath}")
    return data


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent.parent


def ensure_dir(directory: Path) -> None:
    """Ensure directory exists"""
    directory.mkdir(parents=True, exist_ok=True)


# Initialize logger on import
setup_logger()
