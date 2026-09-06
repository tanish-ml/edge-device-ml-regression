# Edge Device ML Regression & Rule-Based System

> 📖 **Read the Detailed Report:** For an in-depth understanding of the methodology, architecture, and results, please read the [ML Report](ML_Report.pdf) provided in this repository.

This repository contains a high-precision machine learning pipeline designed to predict two continuous variables (`target01` and `target02`) under distinct constraints. The project involves predicting a bimodal target using advanced feature engineering and ensembling, as well as developing a lightweight, rule-based regression model tailored for resource-constrained edge devices without the use of external ML libraries.

## Project Overview

### Part 1: Training Regression Model for `target01`
The first part of the project focuses on predicting `target01`, a continuous variable with a clearly bimodal distribution and weak linear correlations (highest Pearson correlation ~0.17). Standard regression models optimizing the mean typically fail for such distributions.

**Methodology:**
- **Feature Engineering:** Multiplicative interaction terms were created using the top 6 Mutual Information (MI) features. This was vital in helping the model capture non-linear dependencies.
- **Dimensionality Reduction:** StandardScaler and PCA (26 latent components) were applied strictly within a cross-validation loop to prevent data leakage.
- **Recursive Feature Selection:** A pilot CatBoost model retained only the top 4.2% of features (95.7th percentile).
- **Modeling:** A tuned `CatBoostRegressor` (Depth: 8, Learning Rate: ~0.04) was chosen over XGBoost/LightGBM ensembles due to its superior handling of the feature interactions and bimodal transitions.
- **Results:** Achieved a highly robust **Test $R^2$ of 0.9518** and RMSE of 0.0520 under a 5-fold cross-validation scheme.

### Part 2: Rule-Based Regression for `target02` (Edge Device)
The second part addresses the constraint of deploying a predictive model on edge hardware capable only of basic logical and numerical operations (no standard ML libraries allowed). 

**Methodology:**
- **Feature Discovery:** Using Random Forest feature importance and Mutual Information scoring, the 273 features were reduced to just four dominant drivers: `feat_49` (the primary regime selector), `feat_219`, `feat_111`, and `feat_22` (critical for residual correction).
- **Model Architecture:** A Piecewise Linear/Polynomial Regression (Model Tree) was engineered. A shallow Decision Tree (Depth 3) was used for logical routing (if/else conditions based primarily on `feat_49`), while linear and polynomial regression formulas were applied at the leaves to calculate the continuous output.
- **Results:** This hybrid architecture eliminated the discretization error typical of standard decision trees, achieving an **$R^2$ of 0.985** and RMSE of 0.092. The logic was successfully implemented as a pure-Python script (`framework_7.py`) with minimal computational overhead.

## Repository Structure

- `src/part1_pipeline.py`: The full machine learning pipeline for predicting `target01` using PCA and CatBoost.
- `src/framework_7.py`: The standalone, pure-Python script predicting `target02` via piecewise polynomial equations.
- `src/rule_extractor.py`: Utility script used to extract logical rules and equations for Part 2.
- `notebooks/`: Exploratory Jupyter notebooks demonstrating the data discovery and feature engineering phases.
- `ML_Report.pdf`: The detailed final project report containing full architecture, methodology, and performance analysis.
- `reports/`: Includes comprehensive visual analytics (KDE distributions, residual plots) and draft documentation.
- `data/`: Contains the datasets (`dataset_7.csv`, `target_7.csv`, `EVAL_7.csv`).

## Code Execution Instructions

### Requirements
Ensure you are using a dedicated Python virtual environment. Install dependencies:
```bash
pip install -r requirements.txt
```
*(Note: standard ML libraries like `scikit-learn`, `catboost`, `pandas`, and `numpy` are required for Part 1.)*

### Running Part 1
To train the regression model and generate predictions on the evaluation set:
```bash
python src/part1_pipeline.py
```
*Note: Ensure the dataset files are correctly placed in `src/problem_7/` or updated in the script before execution.*

### Running Part 2 (Edge Implementation)
The edge-device script can be run without any external ML libraries. To generate predictions using the rule-based framework:
```bash
python src/framework_7.py --eval_file_path <path_to_eval_csv>
```

### Rule Extraction
To print the evaluation metrics and extracted rules for Part 2, execute:
```bash
python src/rule_extractor.py
```

## AI Disclosure
This project utilized generative AI tools during development for brainstorming, boilerplate code generation, and documentation refinement. The foundational logic, feature engineering strategies, and model architecture were developed independently.
