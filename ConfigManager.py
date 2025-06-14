import yaml


class ConfigManager:
    """Load configuration parameters from a YAML file."""

    def __init__(self, path: str = "Parameters.yaml") -> None:
        self.path = path

    def LoadConfig(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
