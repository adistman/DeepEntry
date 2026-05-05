from deepentry.metrics import mean_reciprocal_rank, recall_at_k


def test_metrics():
    ranks = [1, 2, 10, 20]
    assert round(mean_reciprocal_rank(ranks), 6) == round((1 + 0.5 + 0.1 + 0.05) / 4, 6)
    assert recall_at_k(ranks, 10) == 0.75
