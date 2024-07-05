# Input File Type Specification

This document contains the characteristics of the input file types that are supported by the _ageml_ pipelines (`model_age, factor_correlation, clinical_groups, clinical_classify`).

__IMPORTANT__: Not following these specifications may lead to errors and unexpected behavior, so please, make sure your input files comply with them.

## Features File

Specified with the `--features` flag. Contains the age and the features of the subjects.

- __Extension__: CSV. Comma (`,`) separated.
- __Header__: It must contain the variable names, and _the first named column must contain the age of the subjects_. The first column is the row index. (E.g.: `age, BMI, HDL`). The rest of the columns contain the features.
- __Variables__: All variables must be __numeric__. Age can be in any unit.
- __Format__: We use the decimal point (`.`) as the decimal separator.
NOTE: Support for categorical variables is on its way.

Example (units are arbitrary, quantities are not real):

```csv
,age,HDL,LDL,hippocampus_volume,thalamus_volume
0,20,0.5,0.9,142,543
1,21,0.6,1.1,135,636
2,22,0.7,0.89,129,737
3,23,0.8,1.05,128,854
```

## Covariates File

Specified with the `--covariates` flag. Contains the __categorical__ covariates of the subjects. For separation by categories and/or to make covariate corrections.

- __Extension__: CSV. Comma (`,`) separated.
- __Header__: It must contain the covariate names. The first column is row index.
- __Variables__: All variables must be __int__ for categorical variables or __floats__ for continous variables. 

Example:

```csv
,site,biological_gender,smoker,educ_years
0,1,0,1,10.0
1,2,1,1,16.0
2,3,0,0,12.0
3,2,0,10.0
```

## Clinical file

Specified with the `--clinical` flag. Contains the clinical groups to which every subject belongs.

- __Extension__: CSV. Comma (`,`) separated.
- __Header__: It must contain the clinical group names. The first column is the row index. The rest of the columns contain the clinical group names.
- __Variables__: All values must be 0 or 1.

Example (in the context of Alzheimer's disease):

```csv
,CN,MCI,AD
0,1,0,0
1,1,0,0
2,0,1,0
3,0,0,1
```

## Factors File

Specified with the `--factors` flag. Contains the factors for exploring the correlation with the _age delta_.

- __Extension__: CSV. Comma (`,`) separated.
- __Header__: It must contain the factor names. The first column is the row index. The rest of the columns contain the factors.
- __Variables__: All variables must be __numeric__.
- __Format__: We use the decimal point (`.`) as the decimal separator.

Example (units and factors are arbitrary, quantities are not real):

```csv
,func_score,sedentarism_points,neuro_score,MOCA_SCORE,memory_perf,familiar_support,hygiene_habits
0,28.0,0.0,6.0,0.0,21.0,0.702,-0.154
1,30.0,0.0,3.0,1.0,28.0,1.812,2.046
2,30.0,0.0,8.0,0.0,25.0,0.846,0.812
3,26.0,1.0,21.0,20.0,19.0,-0.627,0.643
```

## Systems File

Specified with the `--systems` flag. Contains the systems for which we want to train different models. Each system is a set of columns of the features file. Strictly speaking, systems are variable sets. The naming of the systems is up to the user.

- __Extension__: txt.
- __Format__: In each line, the name of the system, followed by the names of the variables that belong to it, separated by spaces. (E.g.: `system_1_name:var1,var2,var3`). The variable names must be written in the same way as they appear in the header of the features file. Do __not__ include empty lines after the last system.

Example:

```text
cardiovascular:HDL,LDL
brain:hippocampus_volume,thalamus_volume
```
