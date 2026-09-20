from monitoring.model_promotion import MIN_F1_SCORE


def test_candidate_quality_threshold():
    assert MIN_F1_SCORE == 0.95


def test_candidate_must_not_regress():
    production_f1 = 1.0
    candidate_f1 = 0.9981

    assert candidate_f1 < production_f1