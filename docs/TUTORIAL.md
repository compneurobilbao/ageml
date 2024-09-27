# `ageml` tutorial

This document shows a typical workflow you can follow with `ageml` to analyze your data.

You can also tinker around with our [getting started Colab notebook](https://colab.research.google.com/drive/1FtHbIghXLswG8IcOgFZBHJ07wKnLuzbG) if you want a more interactive experience.

## Prerequisites

### Installation

You should have `ageml` installed in your system. If you don't, please follow the instructions in the [README.md](../README.md) file, on the [How to install ageml](../README.md#how-to-install-ageml) section.

### Data

You should also have a set of basic input files, as specified in the [input file specification](../docs/input_file_specification.md) file, or use the toy datasets provided in the `src/ageml/datasets` folder. This tutorial uses the the toy datasets.

- `features.csv`
- `factors.csv`
- `covariates.csv`, with a covariate column named `biological_gender`
- `clinical.csv`, with two categories, `control` and `patient`
- `systems.txt`, with two systems, `cardiovascular` and `brain`

Note: You can generate the toy datasets in a directory of your choice after installing `ageml` by running the following command: `generate_ageml_data -o <your_output_dir>`

## 1. Model Age

To run the model age pipeline run the following command (assuming you are in the toy dataset directory):

```bash
model_age -o <output_dir> -f toy_features.csv --covariates toy_covar.csv --systems toy_systems.txt --clinical toy_clinical.csv --covar_name <covariate_name>
```

This pipeline will:
1. Plot the chronological age distribution of the control group and test if there are statistically significant differences between clinical group age distributions. If a categorical covariate name is provided in the `--covar_name` argument, the control group will be stratified by the covariate and the age distribution will be plotted for each stratum. The `log.txt` file in the specified output directory `output_dir` contains summary statistics of the age distribution of the control group and the results of the statistical tests.
See below the expected output of the `log.txt` file and the age distribution plot:
    ```
    Age distribution of Controls
    [Group: 1]
    Mean age: 64.88
    Std age: 8.57
    Age range: [50,79]
    [Group: 0]
    Mean age: 64.32
    Std age: 8.96
    Age range: [50,79]
    Checking that age distributions are similar using T-test: T-stat (p_value)
    If p_value > 0.05 distributions are considered simiilar and not displayed...
    ```
    <div style="text-align: center;"> <img src="../resources/figs/tutorial/model_age/age_distribution_controls.png" alt="Age distribution of controls stratified by  categorical covariate." height="300"> </div> 

1. Compute and plot the correlation of each feature with the chronological age. If a categorical covariate name is provided in the `--covar_name` argument, the correlation will be computed for each stratum of the covariate. If the `--systems` argument is provided, the correlation will be computed for each system. The `log.txt` file in the specified output directory `output_dir` contains the correlation coefficients and the significance of the correlation for each feature.
See below a summary of the expected outputs:
    ```
    Features by correlation with Age of Controls [System: system_A]
    significance: 0.05 * -> FDR, ** -> bonferroni
    Covariate 1
    1.1 ** x1: -0.57 (3.2e-27)
    2.1 ** x2: -0.52 (1.7e-22)
    3.1 ** x3: -0.44 (9.9e-16)
    4.1 ** x4: -0.31 (6.7e-08)
    Covariate 0
    1.0 ** x2: -0.56 (5.6e-26)
    2.0 ** x1: -0.55 (6.8e-25)
    3.0 ** x3: -0.42 (2e-14)
    4.0 ** x4: -0.30 (8.6e-08)
    ```
    <div style="text-align: center;"> <img src="../resources/figs/tutorial/model_age/features_vs_age_controls_system_A.png" alt="Age distribution of controls stratified by categorical covariate." width="1050"> </div> 

1. Train a model specified with the `-m` argument to predict the chronological age of the control group using the features. The model is trained using a K-fold cross-validation that you can specify with the `--cv` argument and the performance metrics are reported in the log file per fold. The use of hyperparameter tuning, polynomial feature extension, and feature scalers can also be specified with the `-ht`, `-fext`, and `-s` arguments, respectively.
After training the model, an age-bias correction step is performed, where a linear transformation is applied to the data to fit the $y=x$ line. Also, the age deltas are computed and included in a `predicted_ages_*.csv` file, where the deltas per system are included. **This file is necessary to run the rest of the pipelines**.
    <div style="display: flex; justify-content: center;">
        <div style="margin: 10px;">
            <img src="../resources/figs/tutorial/model_age/chronological_vs_pred_age_0_system_A.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
        <div style="margin: 10px;">
            <img src="../resources/figs/tutorial/model_age/age_bias_correction_0_system_A.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
    </div>


## 2. Factor Correlation

To run the factor correlation pipeline run the following command:

```bash
factor_correlation -o <output_dir> -a <output_dir/model_age/predicted_age_*> -f toy_factors.csv --covariates toy_covar.csv --clinical toy_clinical.csv --covcorr_mode cn
```

This pipeline will:
1. Apply covariate correction to the factors, if covariates are provided. We specified to base the covariate correction on the control group, but you can also specify to base it on each clinical group and apply the correction separately to each group. In the [Clinical Classify](#4-clinical_classify) section, we will compare this correction using the option `--covcorr_mode each` to apply the correction based on each clinical group.
2. Compute and plot the correlation between each factor and the age deltas, for every clinical group. If no clinical file is provided, the correlation is computed assuming all subjects are controls. See below the expected output:
    ```
    Correlations between lifestyle factors for g1
    significance: 0.05 * -> FDR, ** -> bonferroni
    System: system_a
    1. ** mocascore: 0.30 (1.5e-05)
    2. ** physicalactivityscore: -0.27 (8.3e-05)
    3. * heavydrinkingscore: 0.16 (0.02)
    System: system_b
    1. ** mocascore: 0.31 (7.1e-06)
    2. ** physicalactivityscore: -0.26 (0.00025)
    3. * heavydrinkingscore: 0.14 (0.046)
    ```
    <div style="text-align: center;"> <img src="../resources/figs/tutorial/factor_correlation/factors_vs_deltas_g1.png" alt="Age distribution of controls stratified by  categorical covariate." height="600"> </div>



## 3. Clinical Groups

To run the clinical groups pipeline run the following command:

```bash
clinical_groups -o <output_dir> -a <output_dir/model_age/predicted_age_*> --clinical toy_clinical.csv --covariates toy_covar.csv --covcorr_mode cn
```

This pipeline will:
1. Plot the age distribution of the clinical groups.
2. Test if there are statistically significant differences between the age distributions of the clinical groups
3. Test if there are statistically significant differences between the distributions of the age deltas, across systems. The log file contains summary statistics of the age distribution of the clinical groups age distributions and the results of the statistical tests.
    ```
    Age distribution of Clinical Groups
    [Group: cn]
    Mean age: 64.60
    Std age: 8.77
    Age range: [50,79]
    [Group: g1]
    Mean age: 71.67
    Std age: 9.31
    Age range: [48,88]
    [Group: g2]
    Mean age: 56.61
    Std age: 9.83
    Age range: [34,80]
    Checking that age distributions are similar using T-test: T-stat (p_value)
    If p_value > 0.05 distributions are considered simiilar and not displayed...
    Age distributions cn and g1 are not similar: -9.70 (4.2e-21)
    -----------------------------------
    -----------------------------------
    Delta distribution for System:system_a
    [Group: cn]
    Mean delta: -0.12
    Std delta: 4.46
    Delta range: [-17, 12]
    [Group: g1]
    Mean delta: -3.12
    Std delta: 4.57
    Delta range: [-17, 6]
    [Group: g2]
    Mean delta: 4.64
    Std delta: 4.66
    Delta range: [-9, 17]
    Checking for statistically significant differences between deltas...
    ===========================================================================================================================
    significance: 0.05 * -> FDR, ** -> bonferroni
    p-val (group1 vs. group2)            Cohen's d              CI (95%):   [low, upp]          Delta difference(mean1 - mean2)
    ---------------------------------------------------------------------------------------------------------------------------
    ** p-val (cn vs. g1): 1.1e-15        Cohen's d: 0.67        CI (95.0%): [2.27  , 3.73  ]    Delta difference: 3.00           
    ** p-val (cn vs. g2): 7.5e-35        Cohen's d: -1.06       CI (95.0%): [-5.51 , -4.02 ]    Delta difference: -4.77          
    ** p-val (g1 vs. g2): 3.2e-48        Cohen's d: -1.68       CI (95.0%): [-8.67 , -6.86 ]    Delta difference: -7.77          
    ```
    <div style="display: flex; justify-content: center;">
        <div style="margin: 10px;">
            <img src="../resources/figs/tutorial/clinical_groups/age_distribution_clinical_groups.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
        <div style="margin: 10px;">
            <img src="../resources/figs/tutorial/clinical_groups/clinical_groups_box_plot_system_a.png" alt="Age distribution of controls stratified by categorical covariate." height="305">
        </div>
    </div>

## 4. `clinical_classify`

To run the clinical classify pipeline run the following command:

```bash
clinical_classify -o <output_dir> -a <output_dir/model_age/predicted_age_*> --clinical toy_clinical.csv --covariates toy_covar.csv --groups <group_A> <group_B> --covcorr_mode cn
```

In the toy dataset, for the groups, you can choose between `cn`, `g1`, and `g2`. This pipeline will:

1. Train a Logistic Regressor to classify between the specified groups using the age deltas. If multiple systems are found in the ages file, a model using the deltas of each system is trained, plus a joint model with the deltas of each system. In the joint model, the coefficient computed for each delta is reported, to have a measure of the importance of the feature importance. Training is done using a K-fold cross-validation that you can specify with the `--cv` argument and the performance metrics are reported in the log file after each model is trained. The threshold used in classification and the confidence interval can be specified with the `--threshold` and `--ci` arguments, respectively.
2. Plot a Receiver-Operating Curve (ROC) for each trained model. See below a summary of the results:
    ```
    Classification between groups g1 and g2 [System: system_a]
    Summary metrics over all CV splits (0.95 CI)
    AUC: 0.881 [0.820-0.941]
    Accuracy: 0.795 [0.715-0.875]
    Sensitivity: 0.798 [0.681-0.914]
    Specificity: 0.795 [0.709-0.881]
    -----------------------------------
    Classification between groups g1 and g2 [System: system_b]
    Summary metrics over all CV splits (0.95 CI)
    AUC: 0.749 [0.681-0.818]
    Accuracy: 0.700 [0.619-0.781]
    Sensitivity: 0.680 [0.558-0.801]
    Specificity: 0.725 [0.663-0.787]
    -----------------------------------
    Classification between groups g1 and g2 [System: all]
    Summary metrics over all CV splits (0.95 CI)
    AUC: 0.910 [0.877-0.943]
    Accuracy: 0.825 [0.780-0.870]
    Sensitivity: 0.826 [0.709-0.943]
    Specificity: 0.828 [0.752-0.905]
    Logistic regressor weigths coef (norm_coef):
    delta_system_a = 0.368 (1.000)
    delta_system_b = 0.224 (0.608)
    ```
    <div style="display: flex; justify-content: center;">
        <div style="margin: 10px;">
            <img src="../resources/figs/tutorial/clinical_classify/cn/roc_curve_g1_vs_g2_system_a.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
        <div style="margin: 10px;">
            <img src="../resources/figs/tutorial/clinical_classify/cn/roc_curve_g1_vs_g2_system_b.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
        <div style="margin: 10px;">
                <img src="../resources/figs/tutorial/clinical_classify/cn/roc_curve_g1_vs_g2_all.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
    </div>


3. Here, we want to demonstrate how the covariate correction can affect the classification results, because in the case of the toy dataset, the covariate "Years of Education" is directly correlated to the age deltas. The classification performance is reduced drastically due to this. We will run the same command but changing the `--covcorr_mode` to `each`:

    ```bash
    clinical_classify -o <output_dir> -a <output_dir/model_age/predicted_age_*> --clinical toy_clinical.csv --covariates toy_covar.csv --groups     <group_A> <group_B> --covcorr_mode each
    ```

    The log and the results are the following:
    ```
    Classification between groups g1 and g2 [System: system_a]
    Applying covariate correction for deltas using each clinical group as reference.
    Summary metrics over all CV splits (0.95 CI)
    AUC: 0.567 [0.448-0.687]
    Accuracy: 0.547 [0.445-0.650]
    Sensitivity: 0.541 [0.450-0.633]
    Specificity: 0.566 [0.370-0.762]
    -----------------------------------
    Classification between groups g1 and g2 [System: system_b]
    Applying covariate correction for deltas using each clinical group as reference.
    Summary metrics over all CV splits (0.95 CI)
    AUC: 0.559 [0.462-0.656]
    Accuracy: 0.540 [0.466-0.614]
    Sensitivity: 0.540 [0.341-0.739]
    Specificity: 0.559 [0.374-0.744]
    -----------------------------------
    Classification between groups g1 and g2 [System: all]
    Applying covariate correction for deltas using each clinical group as reference.
    Summary metrics over all CV splits (0.95 CI)
    AUC: 0.555 [0.503-0.608]
    Accuracy: 0.550 [0.501-0.599]
    Sensitivity: 0.541 [0.439-0.643]
    Specificity: 0.572 [0.426-0.719]
    Logistic regressor weigths coef (norm_coef):
    delta_system_a = 0.050 (1.000)
    delta_system_b = 0.045 (0.895)
    ```

    <div style="display: flex; justify-content: center;">
        <div style="margin: 10px;">
            <img src="../resources/figs/tutorial/clinical_classify/each/roc_curve_g1_vs_g2_system_a.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
        <div style="margin: 10px;">
            <img src="../resources/figs/tutorial/clinical_classify/each/roc_curve_g1_vs_g2_system_b.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
        <div style="margin: 10px;">
                <img src="../resources/figs/tutorial/clinical_classify/each/roc_curve_g1_vs_g2_all.png" alt="Age distribution of controls stratified by categorical covariate." width="400">
        </div>
    </div>

    Thus, we could say that the covariates in our data have good classification power on their own, due to their direct correlation with age deltas. Be aware that this example is only for demonstration purposes of the tool, and reality does not necessarily reflect this phenomenon, so always be cautious when analyzing and interpreting your data.


