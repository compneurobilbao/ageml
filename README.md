# AgeML

_AgeML_ is a Python package for Age Modelling with Machine Learning made easy.

[![Zenodo doi badge](https://img.shields.io/badge/DOI-10.5281/zenodo.10255549-blue.svg)](https://zenodo.org/records/10255550)
[![PyPI version](https://badge.fury.io/py/ageml.svg)](https://badge.fury.io/py/ageml)
[![Lint Test Coverage](https://github.com/compneurobilbao/AgeModelling/actions/workflows/lint_test_coverage.yml/badge.svg?branch=task_5_basic_tests)](https://github.com/compneurobilbao/AgeModelling/actions/workflows/lint_test_coverage.yml)

## Background

Age Modelling consists of trying to predict the chronological age of an organism with a set of features that capture information about its aging status and dynamics. With a rich feature set one can accurately predict the chronological age of a normative (healthy) population using relatively simple machine learning models. When the feature set is sufficiently rich and the model can capture the complexity of the data, performing inference on non-normative populations leads to prediction errors in their chronological age. It has been shown in __CITATIONS__ that the mismatch between the chronological vs. predicted age (known as _age delta_) is a proxy of aberrant aging trajectories, often having high correlations with factors that condition ageing. Furthermore, classification of different clinical groups (e.g.: stable vs progressive) can also be done using the trained models.

## Description

AgeML allows age modelling with a set of simple-to-use CLIs that produce comprehensive figures of the modelling steps and detailed logs for exploring the effectiveness of the trained models.
There are 4 main CLIs:

- __model_age__: It takes a features file (in CSV format) from a set of subjects, which includes their chronological age, and it trains a model for predicting it. Categorical Covariates can be included to train models per category, using a CSV file. Systems, feature groups that correspond to a congruent physiological system (cardiac, muskuloskeletal, gastric, etc.) can also be included to train models per system using a simple .txt file. A CSV file is returned with the predicted ages, and the corresponding _age delta_. Also, a simple correlation analysis is performed to check how the input features correlate with the chronological age. The age distribution of the cohort is plotted too.
- __factor_analysis__: The correlation between the provided factors and the computed _age deltas_ is analyzed, including the strength and significance. If the clinical category of each subject is provided (with a CSV file), this analysis runs for every clinical category.
- __clinical_groups__: To see how the _age delta_ changes across clinical groups, a boxplot is created. The age distribution across the groups is also plotted.
- __clinical_classify__: Using the features file and the ages file (with the _age delta_) classification of two clinical groups is performed. A Receiver Operating Curve (ROC) is plotted for each model used in the classifications.

In the figure below, find a schematic of the described results:
![./basic_functionality_figure](./resources/figs/basic_functionality_fig.svg)

## How to install

_AgeML_ can easily be installed using pip
`pip install ageml`

A Docker image version will be released in the short term.

## How to Contribute

We welcome scientists and developers who want to standardize the procedures of age modelling, share pretrained models or whatever other kind of contribution that can help the project.

The contribution guidelines can be found [here](./docs/CONTRIBUTING.md).

## How to use

### Interactive CLI

### Compact CLI

## Motivation

BrainAge models (Franke et al. 2010, Neuroimage) have had success in exploring the relationship between healthy and pathological ageing of the brain. Furthermore, this type of age modelling can be extended to multiple body systems and modelling of the interactions between them (Tian et al 2023, Nature Medicine).

However, there is no standard for age modelling. There have been works attempting to describe proper procedures, especially for age-bias correction (de Lange and Cole 2020, Neuroimage: Clinical). In this work we developed an Open-Source software that allows anyone to do age modelling following well-established and tested methodologies for any type of clinical data. Age modelling with machine learning made easy.

The objective of AgeML is to standardise procedures, lower the barrier to entry into age modelling and ensure reproducibility. The project is Open-Source to create a welcoming environment and a community to work together to improve and validate existing methodologies. We are actively seeking new developers who want to contribute to growing and expanding the package.

References:

- De Lange, A.-M. G., & Cole, J. H. (2020). Commentary: Correction procedures in brain-age prediction. NeuroImage: Clinical, 26, 102229. <https://doi.org/10.1016/j.nicl.2020.102229>
- Franke, K., Ziegler, G., Klöppel, S., & Gaser, C. (2010). Estimating the age of healthy subjects from T1-weighted MRI scans using kernel methods: Exploring the influence of various parameters. NeuroImage, 50(3), 883–892. <https://doi.org/10.1016/j.neuroimage.2010.01.005>
- Tian, Y. E., Cropley, V., Maier, A. B., Lautenschlager, N. T., Breakspear, M., & Zalesky, A. (2023). Heterogeneous aging across multiple organ systems and prediction of chronic disease and mortality. Nature Medicine, 29(5), 1221–1231. <https://doi.org/10.1038/s41591-023-02296-6>

## License

Licensed under the [Apache 2.0](./LICENSE)
