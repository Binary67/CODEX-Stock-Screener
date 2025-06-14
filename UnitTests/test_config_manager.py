import os
import unittest
import yaml
from ConfigManager import ConfigManager


class TestConfigManager(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "test_params.yaml"
        with open(self.path, "w", encoding="utf-8") as file:
            yaml.dump({"Tickers": ["AAA", "BBB"], "TrainingEndDate": "2020-12-31"}, file)
        self.manager = ConfigManager(self.path)

    def tearDown(self) -> None:
        os.remove(self.path)

    def test_load_config(self):
        config = self.manager.LoadConfig()
        self.assertEqual(config["Tickers"], ["AAA", "BBB"])

    def test_get_training_end_date(self):
        self.manager.LoadConfig()
        result = self.manager.GetTrainingEndDate()
        self.assertEqual(result, "2020-12-31")


if __name__ == "__main__":
    unittest.main()
