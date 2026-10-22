import numpy as np

from openalterego.research.dense_programs import score_programs
from openalterego.research.hyser_fewshot import ridge_query_score, support_query
from openalterego.research.supported_programs import prepare_supported_fold


def test_batched_supported_score_matches_single_program_reference():
    rng = np.random.default_rng(31)
    y1 = np.repeat([1, 2, 3], 20)
    y2 = np.repeat([1, 2, 3], 8)
    x1 = rng.normal(size=(len(y1), 512))
    x2 = rng.normal(size=(len(y2), 512))
    x1[:, :4] += y1[:, None] * 0.5
    x2[:, :4] += y2[:, None] * 0.5
    day1, day2 = (x1, y1), (x2, y2)
    support, query = support_query(y2, 1)
    channels = [0, 1]
    for weight in (0.0, 1.0, 4.0):
        fold = prepare_supported_fold(
            day1, (x2[support], y2[support]), (x2[query], y2[query]),
            features_per_site=2, support_weight=weight)
        batched = float(score_programs(
            [fold], np.asarray([channels]), features_per_site=2)[0])
        reference = ridge_query_score(day1, day2, channels,
            features_per_site=2, alpha=10.0, support=support, query=query,
            support_weight=weight)
        np.testing.assert_allclose(batched, reference)
