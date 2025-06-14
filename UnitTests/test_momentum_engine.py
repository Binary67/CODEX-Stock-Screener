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
        self.assertTrue(pd.isna(returns.loc['AAA', 'Lookback_1']))
        pct_return = self.data.iloc[-1] / self.data.iloc[0] - 1
        volatility = self.data.pct_change(fill_method=None).rolling(5).std().iloc[-1]
        expected_5 = (pct_return / volatility) * 100
        self.assertAlmostEqual(returns.loc['AAA', 'Lookback_5'], expected_5['AAA'])

    def test_momentum_ranker(self):
        ranks = self.engine.MomentumRanker(self.data, [5])
        self.assertEqual(ranks.loc['AAA', 'Lookback_5'], 1)
        self.assertEqual(ranks.loc['CCC', 'Lookback_5'], 2)
        self.assertEqual(ranks.loc['BBB', 'Lookback_5'], 3)

    def test_momentum_ranker_multiple(self):
        ranks = self.engine.MomentumRanker(self.data, [1, 5])
        self.assertIn('Lookback_1', ranks.columns)
        self.assertIn('Lookback_5', ranks.columns)
        self.assertEqual(ranks.loc['AAA', 'Lookback_1'], 1)
        self.assertEqual(ranks.loc['BBB', 'Lookback_1'], 2)


if __name__ == '__main__':
    unittest.main()
