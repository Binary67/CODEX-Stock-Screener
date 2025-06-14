import os
import unittest
import yaml
from ConfigManager import ConfigManager


class TestConfigManager(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "test_params.yaml"
        with open(self.path, "w", encoding="utf-8") as file:
            yaml.dump({"Tickers": ["AAA", "BBB"]}, file)
        self.manager = ConfigManager(self.path)

    def tearDown(self) -> None:
        os.remove(self.path)

    def test_load_config(self):
        config = self.manager.LoadConfig()
        self.assertEqual(config["Tickers"], ["AAA", "BBB"])


if __name__ == "__main__":
    unittest.main()
