import logging
import yaml

LOGGER = logging.getLogger(__name__)


class ConfigManager:
    """Load configuration parameters from a YAML file."""

    def __init__(self, path: str = "Parameters.yaml") -> None:
        self.path = path
        LOGGER.info("ConfigManager initialized with path=%s", path)

    def LoadConfig(self) -> dict:
        LOGGER.info("Loading configuration from %s", self.path)
        with open(self.path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
