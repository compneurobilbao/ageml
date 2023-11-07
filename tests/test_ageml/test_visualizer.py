import pytest
import os
import shutil
import pandas as pd

import ageml.modelling as modelling
import ageml.ui as ui
import ageml.utils as utils
import ageml.visualizer as viz
from ageml.datasets import SyntheticDataset
from .test_modelling import AgeMLTest


@pytest.fixture
def dummy_viz():
    return viz.Visualizer(os.path.dirname(__file__))


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


def test_visualizer_age_distribution(dummy_viz):
    # Get some data
    data = SyntheticDataset().get_np_data()

    # Plot age distribution
    dummy_viz.age_distribution(data[:, -1])
    # Check file existance
    svg_path = os.path.join(dummy_viz.dir,
                            'figures/age_distribution.svg')
    assert os.path.exists(svg_path)
    # Cleanup
    shutil.rmtree(os.path.dirname(svg_path))
