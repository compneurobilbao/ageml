import pytest
import os
import shutil
import pandas as pd

import ageml.modelling as modelling
import ageml.ui as ui
import ageml.utils as utils
import ageml.visualizer as viz
from ageml.datasets import SyntheticData
from .test_modelling import AgeMLTest
from ageml.processing import find_correlations


@pytest.fixture
def dummy_viz():
    return viz.Visualizer(os.path.dirname(__file__))


@pytest.fixture
def np_test_data():
    # We make sure the data has no NaNs because the ui module is supposed to give it clean to the Visualizer.
    return SyntheticData("features").get_data().dropna().to_numpy()


@pytest.fixture
def dummy_ml():
    return AgeMLTest()


def test_visualizer_setup():
    parent_dir = os.path.dirname(__file__)
    output_dir = os.path.join(parent_dir, "test_data")
    # Initialize in any directory
    dummy_visualizer = viz.Visualizer(parent_dir)
    # First assertion
    assert dummy_visualizer.dir == parent_dir
    # Change directory to the one inside the tests
    dummy_visualizer.set_directory(output_dir)
    # Final assertion
    assert dummy_visualizer.dir == output_dir


def test_visualizer_age_distribution(dummy_viz, np_test_data):
    # Plot 'age' distribution (response variable, Y)
    dummy_viz.age_distribution(np_test_data[:, -1])
    # Check file existance
    svg_path = os.path.join(dummy_viz.dir, "figures/age_distribution_.svg")
    assert os.path.exists(svg_path)
    # Cleanup
    shutil.rmtree(os.path.dirname(svg_path))


def test_features_vs_age(dummy_viz, np_test_data):
    # Plot features vs response variable
    X, Y = np_test_data[:, :3], np_test_data[:, -1]
    corr, order, p_values = find_correlations(X, Y)
    dummy_viz.features_vs_age(X, Y, corr, order, p_values, ["X1", "X2", "X3"])
    # Check file existence
    svg_path = os.path.join(dummy_viz.dir, "figures/features_vs_age.svg")
    assert os.path.exists(svg_path)
    # Cleanup
    shutil.rmtree(os.path.dirname(svg_path))


def test_true_vs_pred_age(dummy_viz, np_test_data, dummy_ml):
    # Separate data in X and Y
    X = np_test_data[:, :3]
    Y = np_test_data[:, -1]
    # Fit Age
    Y_pred, Y_corrected = dummy_ml.fit_age(X, Y)
    dummy_viz.true_vs_pred_age(Y, Y_pred)
    # Check file existence
    svg_path = os.path.join(dummy_viz.dir, "figures/true_vs_pred_age.svg")
    assert os.path.exists(svg_path)
    # Cleanup
    shutil.rmtree(os.path.dirname(svg_path))


def test_age_bias_correction(dummy_viz, np_test_data, dummy_ml):
    # Separate data in X and Y
    X = np_test_data[:, :3]
    Y = np_test_data[:, -1]
    # Fit Age
    Y_pred, Y_corrected = dummy_ml.fit_age(X, Y)
    dummy_viz.age_bias_correction(Y, Y_pred, Y_corrected)
    # Check file existence
    svg_path = os.path.join(dummy_viz.dir, "figures/age_bias_correction.svg")
    assert os.path.exists(svg_path)
    # Cleanup
    shutil.rmtree(os.path.dirname(svg_path))


def test_factors_vs_deltas(dummy_viz):
    # Create dummy data
    corrs = [[0.5, 0.6, 0.7, 0.8, 0.9]]
    groups = ["Group 1"]
    labels = ["factor1", "factor2", "factor3", "factor4", "factor5"]
    markers = [['', '*', '', '*', '**']]
    # Plot
    dummy_viz.factors_vs_deltas(corrs, groups, labels, markers)
    # Check file existence
    svg_path = os.path.join(dummy_viz.dir, "figures/factors_vs_deltas.svg")
    assert os.path.exists(svg_path)
    # Cleanup
    shutil.rmtree(os.path.dirname(svg_path))


def test_deltas_by_groups(dummy_viz, np_test_data, dummy_ml):
    # Separate data in X and Y
    X = np_test_data[:, :3]
    Y = np_test_data[:, -1]
    # Fit Age
    _, Y_corrected = dummy_ml.fit_age(X, Y)
    # Compute deltas
    deltas = Y_corrected - Y
    # Create dummy labels
    labels = ["Group 1"]
    # Plot
    dummy_viz.deltas_by_groups(deltas, labels)
    # Check file existence
    svg_path = os.path.join(dummy_viz.dir, "figures/clinical_groups_box_plot.svg")
    assert os.path.exists(svg_path)
    # Cleanup
    shutil.rmtree(os.path.dirname(svg_path))


def test_classification_auc(dummy_viz):
    # Create false labels
    y = [0, 0, 1, 1, 0, 1, 0, 0, 0, 1]
    y_pred = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.8, 0.9, 0.99]
    groups = ["group1", "group2"]
    # Plot
    dummy_viz.classification_auc(y, y_pred, groups)
    # Check file existence
    svg_path = os.path.join(dummy_viz.dir, "figures/roc_curve_%s_vs_%s.svg" % (groups[0], groups[1]))
    assert os.path.exists(svg_path)
    # Cleanup
    shutil.rmtree(os.path.dirname(svg_path))
