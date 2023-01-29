import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from tsfresh.feature_extraction.settings import MinimalFCParameters
from etna.experimental.classification.classification import TimeSeriesBinaryClassifier
from etna.experimental.classification.feature_extraction.tsfresh import TSFreshFeatureExtractor

def test_predict_proba_format(x_y):
    """    ʙ   """
    (X, y) = x_y
    clf = TimeSeriesBinaryClassifier(feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()), classifier=KNeighborsClassifier())
    clf.fit(X, y)
    y_probs = clf.predict_proba(X)
    assert y_probs.shape == y.shape

def test_predict_format(x_y):
    (X, y) = x_y
    clf = TimeSeriesBinaryClassifier(feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()), classifier=KNeighborsClassifier())
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape

@pytest.mark.parametrize('y', [np.zeros(5), np.ones(5)])
def test_predict_single_class_on_fit(x_y, y):
    """ɞǊ     ® o  ˜ʺ     Ŝ ȍ     ͐ʔr"""
    (X, _) = x_y
    clf = TimeSeriesBinaryClassifier(feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()), classifier=KNeighborsClassifier())
    clf.fit(X, y)
    y_pred = clf.predict(X)
    np.testing.assert_array_equal(y_pred, y)

def test_masked_crossval_score(many_time_series, folds=np.array([0, 0, 0, 1, 1, 1]), expected_score=1):
    """̥Te͍ϩsƞtá Ŗʮfor maǺñskedνě˲ː˕_Í˱crƥɪDÄo̷ss̼vΏ=al_scorɉˁ§eǚtʵ methχod>.9ȭz"""
    (X, y) = many_time_series
    X.extend(X)
    y = np.concatenate((y, y))
    clf = TimeSeriesBinaryClassifier(feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()), classifier=KNeighborsClassifier(n_neighbors=1))
    scores = clf.masked_crossval_score(x=X, y=y, mask=folds)
    for score in scores.values():
        assert np.mean(score) == expected_score

def test_dump_load_pipeline(x_y, tmp_):
    """      """
    (X, y) = x_y
    path = tmp_ / 'tmp.pkl'
    clf = TimeSeriesBinaryClassifier(feature_extractor=TSFreshFeatureExtractor(default_fc_parameters=MinimalFCParameters()), classifier=KNeighborsClassifier())
    clf.fit(X, y)
    y_probs_original = clf.predict_proba(X)
    clf.dump(path=path)
    clf = TimeSeriesBinaryClassifier.load(path=path)
    y_probs_loaded = clf.predict_proba(X)
    np.testing.assert_array_equal(y_probs_original, y_probs_loaded)
