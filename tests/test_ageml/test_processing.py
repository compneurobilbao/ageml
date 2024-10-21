import pytest
import numpy as np
import pandas as pd
import ageml.processing as processing

def test_find_correlations():
    # Test a very simple correlation
    X = np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18]])
    Y = np.array([1, 2, 3])
    corrs, order, p_values = processing.find_correlations(X, Y)
    corrs_expected = np.array([1, 1, -1])
    order_expected = np.array([2, 1, 0])
    p_values_expected = np.array([0.0, 0.0, 0.0])

    assert np.allclose(corrs, corrs_expected, rtol=1e-10) is True
    assert np.array_equal(order, order_expected) is True
    assert np.allclose(p_values, p_values_expected, atol=1e-7) is True


@pytest.mark.parametrize(
    "X, Y, exception_msg",
    [
        (
            np.array([[2, 4, np.nan], [4, 8, -12], [6, 12, -18], [8, 16, -24]]),
            np.array([1, 2, 3, 4]),
            "NaN entrie(s) found in X.",
        ),
        (
            np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18], [8, 16, -24]]),
            np.array([1, 2, 3, np.nan]),
            "NaN entrie(s) found in Y.",
        ),
    ],
)

def test_find_correlations_nans(X, Y, exception_msg):
    with pytest.raises(ValueError) as exc_info:
        processing.find_correlations(X, Y)
        assert exc_info.type == ValueError
    assert str(exc_info.value) == exception_msg


def test_features_mutual_info():
    # Test a very simple correlation with  10 samples and 3 features
    X = np.array([[0.3745401188473625, 0.4967141530112327, 0.3736408237384475],
        [0.9507143064099162, 1.6476885381006925, 0.4203169301898962],
        [0.7319939418114051, 1.5230298564080252, 0.5096906931601679],
        [0.5986584841970366, 1.7654272214502887, 0.6308892301278767],
        [0.1560186404424368, 2.1573932080201557, 0.6636521046022546],
        [0.15599452033620265, 2.042242877065379, 0.3835522435823457],
        [0.05808361216819946, 2.3069644747451507, 0.7056900969046166],
        [0.8661761457749352, 2.588435887075369, 0.6991200299110195],
        [0.6011150117432088, 3.3213680975651604, 0.49016643128497884],
        [0.7080725777960455, 4.236709491177892, 0.1544514906036125]])
    Y = np.array([5.75550373512749, 12.839279335258151, 11.8881552158944, 12.7480520608142, 15.878170073704204,
                  13.585769601473384, 14.959185247632686, 17.504653753294827, 19.66407814601714, 23.08686833692636])
    order, _ = processing.features_mutual_info(X, Y)
    order_expected = np.array([1, 2, 0])
    assert np.array_equal(order, order_expected) is True


def test_feature_mutual_info_discrimination():
    # Test a very simple group discrimination with 5 samples and 3 features
    X = np.array( [[0.3745401188473625, 0.4967141530112327, 0.3736408237384475],
        [0.9507143064099162, 1.6476885381006925, 0.4203169301898962],
        [0.7319939418114051, 1.5230298564080252, 0.5096906931601679],
        [0.5986584841970366, 1.7654272214502887, 0.6308892301278767],
        [0.1560186404424368, 2.1573932080201557, 0.6636521046022546],
        [0.15599452033620265, 2.042242877065379, 0.3835522435823457],
        [0.05808361216819946, 2.3069644747451507, 0.7056900969046166],
        [0.8661761457749352, 2.588435887075369, 0.6991200299110195],
        [0.6011150117432088, 3.3213680975651604, 0.49016643128497884],
        [0.7080725777960455, 4.236709491177892, 0.1544514906036125]])
    Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    order, _ = processing.feature_mutual_info_discrimination(X, Y)
    order_expected = np.array([1, 0, 2])
    assert np.array_equal(order, order_expected) is True


def test_covariate_correction():
    # Variables for testing
    X = np.array([[2, 4, -6], [4, 8, -12], [6, 12, -18]])
    Z = np.array([1, 2, 3]).reshape(-1, 1)

    # Check ValueError raies with NaNs
    with pytest.raises(ValueError) as exc_info:
        processing.covariate_correction(X, np.array([1, 2, np.nan]).reshape(-1, 1))
        assert exc_info.type == ValueError
    with pytest.raises(ValueError) as exc_info:
        processing.covariate_correction(np.array([2.0, np.nan]), Z)
        assert exc_info.type == ValueError
    with pytest.raises(ValueError) as exc_info:
        processing.covariate_correction(X, Z, beta=np.array([2.0, np.nan]).reshape(-1, 1))
        assert exc_info.type == ValueError

    # Check ValueError raises with incompatible shapes
    with pytest.raises(ValueError) as exc_info:
        processing.covariate_correction(X, np.array([1, 2]))
        assert exc_info.type == ValueError

    # Test a very simple correlation
    _, beta = processing.covariate_correction(X, Z)
    beta_expected = np.array([2.0, 4.0, -6.0])
    assert np.allclose(beta, beta_expected)


# TODO do again for classification
def test_handler_wrong_task_type():
    with pytest.raises(ValueError) as exc_info:
        processing.CVMetricsHandler(task_type='asdf')
        assert exc_info.type == ValueError
    assert str(exc_info.value) == 'task_type must be either "regression" or "classification"'

@pytest.fixture
def regressionhandler():
    return processing.CVMetricsHandler(task_type='regression')

@pytest.fixture
def sample_metrics():
    return (
        processing.RegressionFoldMetrics(mae=1.0, rmse=1.5, r2=0.8, p=0.05),
        processing.RegressionFoldMetrics(mae=1.2, rmse=1.7, r2=0.75, p=0.06)
    )

def test_add_fold_metrics(regressionhandler, sample_metrics):
    train_fold, test_fold = sample_metrics
    regressionhandler.add_fold_metrics(train_fold, test_fold)
    
    assert len(regressionhandler.train_metrics) == 1
    assert len(regressionhandler.test_metrics) == 1
    assert regressionhandler.train_metrics[0] == train_fold
    assert regressionhandler.test_metrics[0] == test_fold

def test_calculate_summary(regressionhandler, sample_metrics):
    summary = regressionhandler._calculate_summary(list(sample_metrics))

    assert set(summary.keys()) == {'mae', 'rmse', 'r2', 'p'}
    for metric in summary.values():
        assert set(metric.keys()) == {'mean', 'std', 'min', 'max', '95ci'}

    assert pytest.approx(summary['mae']['mean']) == 1.1
    assert pytest.approx(summary['rmse']['mean']) == 1.6

def test_get_summary(regressionhandler, sample_metrics):
    train_fold, test_fold = sample_metrics
    regressionhandler.add_fold_metrics(train_fold, test_fold)
    summary = regressionhandler.get_summary()

    assert set(summary.keys()) == {'train', 'test'}
    for split in summary.values():
        assert set(split.keys()) == {'mae', 'rmse', 'r2', 'p'}
        for metric in split.values():
            assert set(metric.keys()) == {'mean', 'std', 'min', 'max', '95ci'}

def test_get_summary_dataframe(regressionhandler, sample_metrics):
    train_fold, test_fold = sample_metrics
    regressionhandler.add_fold_metrics(train_fold, test_fold)
    df = regressionhandler.get_summary_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {'split', 'metric', 'statistic', 'value'}
    assert len(df) == 40  # 2 splits * 4 metrics * 5 statistics

    assert set(df['split'].unique()) == {'train', 'test'}
    assert set(df['metric'].unique()) == {'mae', 'rmse', 'r2', 'p'}
    assert set(df['statistic'].unique()) == {'mean', 'std', 'min', 'max', '95ci'}