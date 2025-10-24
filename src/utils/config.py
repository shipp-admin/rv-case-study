"""Configuration Management Utilities"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for analysis project."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._data_config = None
        self._feature_config = None

    def load_data_config(self) -> Dict[str, Any]:
        """Load data configuration."""
        if self._data_config is None:
            config_path = self.config_dir / "data_config.yaml"
            with open(config_path, 'r') as f:
                self._data_config = yaml.safe_load(f)
        return self._data_config

    def load_feature_config(self) -> Dict[str, Any]:
        """Load feature engineering configuration."""
        if self._feature_config is None:
            config_path = self.config_dir / "feature_config.yaml"
            with open(config_path, 'r') as f:
                self._feature_config = yaml.safe_load(f)
        return self._feature_config

    def get(self, config_type: str, *keys) -> Any:
        """Get configuration value by keys.

        Args:
            config_type: 'data' or 'feature'
            *keys: Nested keys to access value

        Returns:
            Configuration value

        Example:
            config.get('data', 'paths', 'raw_data')
        """
        if config_type == 'data':
            config = self.load_data_config()
        elif config_type == 'feature':
            config = self.load_feature_config()
        else:
            raise ValueError(f"Unknown config type: {config_type}")

        value = config
        for key in keys:
            value = value[key]
        return value


# Global config instance
config = Config()
