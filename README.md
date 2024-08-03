# ImmoEliza - Step 2 : Data cleaning and preprocessing, and machine leaning model for predicting real estate prices

## Overview

This repository contains scripts for cleaning, preprocessing, and modeling real estate data to predict property prices. The project involves two main components:

1. **Data Cleaning**: Script for cleaning and preparing the dataset.
2. **Preprocessing and Modeling**: Script for preparing data for machine learning and building predictive models.

## Files

- **`cleaning.py`**: Contains the DataCleaning class that handles the data cleaning process using multiple methods.
- **`preprocessing_modelling.py`**: Contains functions for data preprocessing and model training. Prints details about the accuracy of the model.

## Usage

1. **Data Cleaning**: Run `cleaning.py` to clean the data and generate a cleaned dataset.
2. **Preprocessing and Modeling**: Run `preprocessing_modelling.py` to preprocess the data and train predictive models.

   Make sure to update the paths to the input data files in the scripts before running them.

## Data Cleaning

The data cleaning process involves various steps to ensure the dataset is in a suitable format for analysis. The key steps include:

- **Removing Unnecessary Values**: Filter out rows with values that are not relevant to the analysis.
- **Filtering Data**: Keep only the rows that meet specific criteria.
- **Merging Data**: Combine datasets based on common columns and remove duplicates.
- **Handling Outliers**: Remove or adjust extreme values to ensure data quality.
- **Correcting Incoherent Values**: Adjust values that do not follow expected patterns or relationships.
- **Replacing Missing Values**: Fill missing values with specified defaults or computed values.
- **Converting Categorical Values**: Transform categorical data into numerical format using predefined mappings.
- **Modifying Columns**: Adjust column values based on conditions from other columns.
- **Executing a Cleaning Process**: Apply a series of predefined cleaning steps to the dataset.

The cleaned dataset is then saved in CSV and Excel formats for use in the next phase.

## Preprocessing and Modeling

The preprocessing and modeling phase prepares the cleaned data for machine learning and builds predictive models. The main steps are:

- **Data Preparation**: The cleaned dataset is split into subsets based on property type (e.g., apartments and houses) and saved for further analysis.
- **Data Splitting**: The dataset is divided into training and testing sets to evaluate model performance.
- **Data Imputation**: Handle missing values using K-Nearest Neighbors (KNN) imputation.
- **Model Training**: XGBoost models are trained separately for apartments and houses with optimized hyperparameters.
- **Model Evaluation**: Assess model performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) metrics.
- **Feature Importance Analysis**: Analyze and visualize feature importance to understand key factors influencing property prices.

## Requirements

To run the scripts, the following Python packages are required:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `openpyxl` (for Excel file handling)

You can install the necessary packages using:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib openpyxl
