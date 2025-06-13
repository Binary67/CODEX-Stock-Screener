import unittest
import pandas as pd
from MomentumEngine import MomentumEngine


class TestMomentumRanker(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = MomentumEngine()
        data = pd.DataFrame({
            'AAA': [1, 2, 3, 4, 5, 6],
            'BBB': [1, 1, 1, 1, 1, 1],
            'CCC': [5, 4, 3, 2, 1, 0.5],
        }, index=pd.date_range('2020-01-01', periods=6))
        self.data = data

    def test_cumulative_return_calculator(self):
        returns = self.engine.CumulativeReturnCalculator(self.data, [1, 5])
        expected_1 = (self.data.iloc[-1] / self.data.iloc[-2] - 1) * 100
        expected_5 = (self.data.iloc[-1] / self.data.iloc[0] - 1) * 100
        self.assertAlmostEqual(returns.loc['AAA', 'Lookback_1'], expected_1['AAA'])
        self.assertAlmostEqual(returns.loc['AAA', 'Lookback_5'], expected_5['AAA'])

    def test_momentum_ranker(self):
        ranks = self.engine.MomentumRanker(self.data, [5])
        self.assertEqual(ranks.loc['AAA', 'Lookback_5'], 1)
        self.assertEqual(ranks.loc['BBB', 'Lookback_5'], 2)
        self.assertEqual(ranks.loc['CCC', 'Lookback_5'], 3)


if __name__ == '__main__':
    unittest.main()
