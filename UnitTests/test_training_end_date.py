import os
import unittest
import yaml
import pandas as pd
from ConfigManager import ConfigManager
from MomentumEngine import MomentumEngine


class TestTrainingEndDate(unittest.TestCase):
    def setUp(self) -> None:
        self.path = "test_params.yaml"
        with open(self.path, "w", encoding="utf-8") as file:
            yaml.dump({"TrainingEndDate": "2020-01-04"}, file)
        self.manager = ConfigManager(self.path)
        dates = pd.date_range("2020-01-01", periods=6)
        self.data = pd.DataFrame({
            "AAA": [1, 2, 3, 4, 10, 12],
            "BBB": [1, 2, 3, 4, 5, 6],
            "CCC": [6, 5, 4, 3, 2, 1],
        }, index=dates)

    def tearDown(self) -> None:
        os.remove(self.path)

    def test_momentum_uses_filtered_data(self):
        config = self.manager.LoadConfig()
        end_date = self.manager.GetTrainingEndDate()
        engine = MomentumEngine()
        history = self.data.loc[:end_date]
        ranks_filtered = engine.MomentumRanker(history, [2])
        ranks_full = engine.MomentumRanker(self.data, [2])
        self.assertNotEqual(ranks_filtered.loc["AAA", "Lookback_2"],
                            ranks_full.loc["AAA", "Lookback_2"])
        self.assertEqual(len(history), 4)


if __name__ == "__main__":
    unittest.main()
