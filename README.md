# Evidence-Based Model Selection: When and Why Linear Models Outperform Gradient Boosting Decision Trees

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Abstract

Despite widespread adoption of Gradient Boosting Decision Trees (GBDT), practitioners lack systematic criteria for when linear models provide superior practical value. This knowledge gap impacts model selection in applications requiring computational efficiency, interpretability, and extrapolation capability. We address this through five systematic experiments isolating data characteristics: linearity dominance, feature interactions, extrapolation demands, small-sample scenarios, and interpretability requirements. Our multidimensional evaluation framework integrates predictive performance with computational efficiency and interpretability costs, providing comprehensive empirical comparison of linear regression and GBDT. Linear models significantly outperform GBDT under four critical conditions. For linearity-dominant data, ridge regression achieved positive $R^2$ values while GBDT showed systematic overfitting. In extrapolation tasks, linear models maintained excellent performance ($R^2 = 0.885$), whereas GBDT performed poorly, yielding highly negative $R^2$ values. For small-sample scenarios, Lasso demonstrated strong overfitting resistance (factor: 2.33 vs. XGBoost: 286,563). In real-world classification, logistic regression achieved superior performance with computational advantages ($57\times$ faster training, $>1,000\times$ faster explanation generation). These results establish systematic evidence for conditions favoring linear models, challenging assumptions of GBDT superiority. Our framework provides practitioners with actionable guidelines for evidence-based model selection, with implications for sustainable machine learning, regulatory compliance and resource-constrained deployments.

## Key Findings

-   **Linearity-dominant data**: Ridge regression achieved positive R² values while GBDT methods showed systematic overfitting.
-   **Extrapolation tasks**: Linear models maintained excellent performance (R² = 0.885) while GBDT failed catastrophically (R² = -2.336 to -3.146).
-   **Small-sample scenarios**: Lasso demonstrated extraordinary overfitting resistance (overfitting factor of 2.33 vs. XGBoost's 286,563).
-   **Computational efficiency**: Linear models show up to 57× faster training and >1000× faster explanation generation.
-   **Real-world classification**: Logistic regression achieved superior predictive performance with orders-of-magnitude computational advantages.

## Repository Structure

```
.
├── LICENSE
├── README.md
├── github_setup.sh
└── src/
    ├── ex1_model_comparison_linearity.py
    ├── ex2_low_interaction_comparison.py
    ├── ex3_extrapolation_comparison.py
    ├── ex4_small_sample_comparison.py
    ├── ex5_interpretability_comparison.py
    ├── (*.csv, *.pdf, *.png, *.txt files)
    └── .gitignore
```

## Experimental Framework

This repository contains the complete code and results for five systematic experiments designed to provide an evidence-based comparison between linear models and GBDT.

1.  **Linearity Dominance (`ex1_...`)**: Evaluates performance when underlying relationships are primarily linear.
2.  **Feature Interactions (`ex2_...`)**: Tests behavior with limited feature interaction complexity using the Friedman1 dataset.
3.  **Extrapolation Capability (`ex3_...`)**: Assesses prediction quality beyond the training data distribution.
4.  **Small-Sample Robustness (`ex4_...`)**: Analyzes overfitting resistance in high-dimensional, small-sample scenarios ($p > n$).
5.  **Interpretability & Real-World Performance (`ex5_...`)**: Compares interpretability costs and classification performance on the Statlog (German Credit Data) dataset.

## Reproducibility

To reproduce the results presented in the paper, follow the steps below.

### Prerequisites

-   Python 3.12+
-   Required Python libraries:
    -   `scikit-learn`
    -   `xgboost`
    -   `lightgbm`
    -   `numpy`
    -   `pandas`
    -   `matplotlib`
    -   `japanize-matplotlib`
    -   `seaborn`
    -   `shap`
    -   `ucimlrepo`

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/nshrhm/linear-vs-gbdt.git
    cd linear-vs-gbdt
    ```

2.  Install Python dependencies:
    ```bash
    pip install scikit-learn xgboost lightgbm numpy pandas matplotlib japanize-matplotlib seaborn shap ucimlrepo
    ```

### Running Experiments

All experiments can be run from the root directory of the project. The scripts will generate CSV files with results and PDF/PNG files with visualizations inside the `src/` directory.

```bash
# Experiment 1: Linearity dominance
python src/ex1_model_comparison_linearity.py

# Experiment 2: Feature interactions
python src/ex2_low_interaction_comparison.py

# Experiment 3: Extrapolation capability
python src/ex3_extrapolation_comparison.py

# Experiment 4: Small-sample robustness
python src/ex4_small_sample_comparison.py

# Experiment 5: Interpretability and real-world performance
python src/ex5_interpretability_comparison.py
```

## Citation

If you use this work in your research, please cite the accompanying paper:

```bibtex
@article{shirahama2025evidence,
  title={Evidence-Based Model Selection: When and Why Linear Models Outperform Gradient Boosting Decision Trees},
  author={Shirahama, Naruki},
  journal={arXiv preprint},
  year={2025}
}
```

## Author

-   **Naruki Shirahama**
-   Faculty of Data Science, Shimonoseki City University
-   `nshirahama@ieee.org`

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).
