# `ageml` tutorial

This document shows a typical workflow you can follow with `ageml` for a set of typical inputs.

## Prerequisites

### Installation

You should have `ageml` installed in your system. If you don't, please follow the instructions in the [README.md](../README.md) file, on the [How to install ageml](../README.md#how-to-install-ageml) section.

### Files

You should also have a set of basic input files, as specified in the [input file specification](../docs/input_file_specification.md) file.

We will asume that the files have the following names:

- `features.csv`
- `factors.csv`
- `covariates.csv`, with a covariate column named `biological_gender`
- `clinical_categories.csv`, with two categories, `control` and `patient`
- `systems.txt`, with two systems, `cardiovascular` and `brain`

## 1-`model_age`

To run model age, you can use the following line:

```bash
model_age --features features.csv --ages ages.csv --systems systems.txt --covariates covariates.csv --covar_name biological_gender --clinical clinical_categories.csv --output ./<my_output_dir>
```

This pipeline will produce a file named `ages.csv` in the specified output directory _<my_output_dir>_

## 2- `factor_correlation`

```bash
factor_correlation --factors factors.csv --ages ages.csv --clinical clinical_categories.csv --output ./<my_output_dir>
```

## 3- `clinical_groups`

```bash
clinical_groups --ages ages.csv --clinical clinical_categories.csv --output ./<my_output_dir>
```

## 4- `clinical_classify`

```bash
clinical_classify --ages ages.csv --clinical clinical_categories.csv --groups control patient --output ./<my_output_dir>
```
