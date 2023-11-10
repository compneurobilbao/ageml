import os
import importlib.resources as pkg_resources
import ageml.datasets.synthetic_test_data as data_generator

if __name__ == "__main__":
    # Generate synthetic features
    data_generator.generate_synthetic_features()
    # Generate synthetic covariates
    data_generator.generate_synthetic_covariates()
    # Generate synthetic factors
    data_generator.generate_synthetic_factors()