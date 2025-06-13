import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
import pandas as pd
from src.MomentumEngine import MomentumEngine


def test_ranking_correctness():
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    prices = pd.DataFrame({
        'AAA': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'BBB': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5],
        'CCC': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    }, index=dates)

    engine = MomentumEngine([1, 3])
    ranked = engine.MomentumRanker(prices)

    expected_order = ['AAA', 'BBB', 'CCC']
    assert list(ranked.index) == expected_order
    assert list(ranked['Rank']) == [1, 2, 3]
