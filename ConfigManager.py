import logging
import yaml

LOGGER = logging.getLogger(__name__)


class ConfigManager:
    """Load configuration parameters from a YAML file."""

    def __init__(self, path: str = "Parameters.yaml") -> None:
        self.path = path
        self.Config: dict | None = None
        LOGGER.info("ConfigManager initialized with path=%s", path)

    def LoadConfig(self) -> dict:
        LOGGER.info("Loading configuration from %s", self.path)
        with open(self.path, "r", encoding="utf-8") as file:
            self.Config = yaml.safe_load(file) or {}
        return self.Config

    def GetTrainingEndDate(self, default: str = "2023-12-31") -> str:
        """Return the training end date from the loaded configuration."""
        if self.Config is None:
            self.LoadConfig()
        return str(self.Config.get("TrainingEndDate", default))
