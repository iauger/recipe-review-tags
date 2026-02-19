# Recipe Rating Prediction - CS 613 Final Project

This project implements machine learning models to predict recipe ratings based on user interactions and recipe metadata. Developed for CS 613, the project focuses on implementing core algorithms (Linear Regression, Logistic Regression, Stochastic Gradient Descent) from scratch and comparing them against `scikit-learn` implementations.

## Execution Pipeline

For ease of inspection and reproducibility, the entire project pipeline—from data loading to model evaluation—is consolidated into a single Jupyter Notebook. This approach allows for an end-to-end walkthrough with supporting visualizations and detailed commentary at every step.

* **Notebook Location:** `notebooks/project_report.ipynb`
* **Pre-trained Assets:** Final models are serialized and stored in the `data/models` directory to avoid redundant training.
* **Reproducibility:** While pre-trained models are provided and all outputs/visualization are already loaded, the notebook is fully functional and supports re-executing the entire pipeline locally if desired. Additionally there is a pdf version of the notebook for review if required. 


## Project Structure

The project is organized into the following directory structure:

```text
PROJECT
├── data/                  # Data storage
│   ├── models/            # Saved model artifacts
│   ├── processed/         # Cleaned and engineered data ready for training
│   └── raw/               # Original immutable data dump
├── notebooks/             # Jupyter notebooks for analysis and reporting
│   ├── project_report.ipynb
│   └── project_report.pdf
├── presentation/          # Final project presentation slides
├── proposal/              # Initial project proposal assets
├── references/            # Academic papers and literature referenced
├── src/                   # Source code
│   ├── data_prep/         # Scripts for cleaning and merging data
│   ├── features/          # Feature engineering logic
│   ├── models/            # Model implementations (Custom & Sklearn)
│   ├── tuning/            # Hyperparameter tuning scripts
│   └── utils/             # Helper functions (text cleaning, etc.)
└── README.md
```

### 1. Data Preparation (`src/data_prep/`)
This module handles the Extract, Transform, Load (ETL) pipeline, converting raw CSVs into model-ready datasets.

* **`load_data.py`**
    * **Data Source:** Automates downloading the `shuyangli94/food-com-recipes-and-user-interactions` dataset via the Kaggle API.
    * **Directory Management:** Automatically generates the required file structure (`data/raw`, `data/processed`).
    * **Ingestion:** Reads `RAW_recipes.csv` and `RAW_interactions.csv` with optimized data types for memory efficiency.

* **`clean_recipes.py`**
    * **Nutrition Parsing:** Expands the single-string `nutrition` list into 7 distinct features: `calories`, `fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, and `carbs`.
    * **Text Normalization:** Parses stringified Python lists (ingredients, steps, tags) and converts them into TF-IDF friendly strings (e.g., joining multi-word ingredients with underscores).
    * **Type Correction:** Standardizes timestamps and converts numeric columns to `float32`.

* **`clean_interactions.py`**
    * **Review Cleaning:** Normalizes user review text using custom text cleaning utilities.
    * **Filtering:** Removes invalid ratings (e.g., 0 ratings used as null placeholders) and ensures valid user/recipe ID linkage.

* **`merge_data.py`**
    * **Dataset merging:** Performs an inner join between recipes and interactions to ensure all data points have complete metadata.
    * **Target Engineering:** Generates a binary classification target `liked` (1 if rating $\ge$ 4, else 0).
    * **Validation:** Enforces data integrity by dropping rows missing essential features like `ingredients_clean` or `calories`.
    * **Splitting:** Includes logic (`clone_data`) to split data into subsets optimized for Vectorization vs. Collaborative Filtering tasks.

* **`sampling.py`**
    * **Class Balancing:** Implements utility functions to handle class imbalance, critical for the rating prediction task.
    * **Strategies:** Includes `undersample_majority_class`, `oversample_minority_class`, and a flexible `ordinal_resample` for multi-class rating targets (1-5 stars).

### 2. Feature Engineering (`src/features/`)
This module handles the transformation of raw text and numeric data into a format suitable for machine learning.

* **`feature_engineering.py`**
    * **Dynamic Text Vectorization:** Implements a dictionary-driven TF-IDF configuration (`TFIDF_COLUMNS`) where `max_features` is dynamically calculated based on a percentage of the unique vocabulary size for each column (e.g., 90% for ingredients, 33% for steps).
    * **Preprocessing Pipeline:** Wraps all transformations in a Scikit-Learn `ColumnTransformer` pipeline:
        * **Text:** Applies `TfidfVectorizer` with specific n-gram ranges (1-2 words) to columns like `ingredients_clean` and `description_clean`.
        * **Numeric:** Applies `MinMaxScaler` to normalize features like `minutes`, `calories`, and `n_steps`.
    * **Leakage Prevention:** The `prepare_features` function ensures transformers are fitted *only* on the training set to prevent data leakage during validation.

### 3. Models (`src/models/`)
This directory contains the core machine learning logic, split between custom implementations (for algorithmic demonstration) and production-ready Scikit-Learn wrappers.

#### Custom Implementations (From Scratch)
* **`gradient_descent.py`**
    * **Base Architecture:** Defines an abstract base class `BaseOptimizer` that enforces a standard interface (`fit`, `predict`, `gradient`) for all custom models.
    * **Optimization Logic:** Implements the core `gradient_descent` loop with learning rate `lr`, max iterations `max_iter`, and tolerance-based convergence checks.
    * **Bias Handling:** Automatically handles the addition of a bias term (intercept) to input matrices.

* **`linear_regression.py`**
    * **Loss Function:** Minimizes Mean Squared Error (MSE): $\frac{1}{2n} \sum (y - \hat{y})^2$.
    * **Gradient:** Computes the gradient analytically as $\frac{1}{n} X^T (Xw - y)$.

* **`logistic_regression.py`**
    * **Activation:** Uses a numerically stable Sigmoid function with `np.clip` to prevent overflow.
    * **Loss Function:** Minimizes Binary Cross-Entropy loss.
    * **Prediction:** Supports both probability estimates (`predict_proba`) and hard class labels (`predict`).

#### Baseline & Evaluation
* **`sklearn_models.py`**
    * **Class Balancing:** Wrappers for `LogisticRegression` and `LinearSVC` utilize `class_weight='balanced'` to automatically adjust weights inversely proportional to class frequencies.
    * **Ensembling:** Includes a `voting_ensemble` that combines Logistic Regression, SVM, and Naive Bayes using soft voting.
    * **Ordinal Regression:** Adapts `mord.LogisticIT` for ordinal tasks where the relative ordering of ratings matters.

* **`train.py`**
    * **Training Orchestration:** Contains `train_classifier` and `train_regressor` functions that manage the full lifecycle: splitting data, resampling, feature transformation, training, and evaluation.
    * **Sampling Integration:** Seamlessly integrates with `src.data_prep.sampling` to apply undersampling, oversampling, or ordinal resampling before training.

* **`evaluate.py`**
    * **Metrics:** Provides specialized reports for different tasks:
        * **Classification:** Accuracy, Precision, Recall, F1, and ROC-AUC.
        * **Ordinal:** Includes Cohen's Kappa (Quadratic weights) to measure agreement.
        * **Regression:** MSE, RMSE, MAE, and $R^2$ Score.

### 4. Hyperparameter Tuning (`src/tuning/`)
Scripts dedicated to optimizing model configurations using Grid Search strategies.

* **`log_gd_tuning.py`**
    * **Custom Model Tuning:** Specifically designed for the custom `LogisticRegressionGD` implementation.
    * **Grid Search:** Iterates through combinations of learning rates (`lrs`), regularization strengths (`lambdas`), and convergence tolerances (`tolerances`) to find the optimal descent parameters.
    * **Progress Tracking:** Uses `tqdm` to visualize the progress of the exhaustive search over parameter combinations.

* **`ordinal_tuning.py`**
    * **Ordinal Strategy:** Tunes the Ordinal Logistic Regression model (`mord.LogisticIT`).
    * **Joint Optimization:** Unlike standard tuning, this script evaluates model hyperparameters (like `alpha`) alongside data sampling strategies (`undersample`, `oversample`, `balanced`) to find the best configuration for the imbalanced rating data.

* **`sklearn_tuning.py`**
    * **Generic Grid Search:** A flexible utility `tune_model` that accepts any Scikit-Learn compatible estimator and a parameter dictionary.
    * **Recursive Generation:** Uses a recursive generator to build parameter combinations from the provided grid.
    * **Analysis:** Includes `results_to_df` to convert the list of metrics into a sorted Pandas DataFrame for easy comparison of experiments.

### 5. Utilities (`src/utils/`)
* **`text_cleaning.py`**
    * **Normalization:** Provides the `normalize_text` function used across the data preparation pipeline.
    * **Logic:** Standardizes text by lowercasing, removing punctuation (preserving hyphens), and collapsing multiple spaces into single whitespace.