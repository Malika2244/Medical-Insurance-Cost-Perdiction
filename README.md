# Medical Insurance Cost Prediction - Big Brains Internship

## Problem Statement
Predict medical insurance charges based on personal attributes like age, BMI, smoking status, gender, children, and region. This helps insurance companies estimate risk and pricing.

## Dataset
- Source: insurance.csv (Kaggle - Medical Cost Personal Datasets)
- Rows: 1338 (after cleaning 1337)
- Columns: age, sex, bmi, children, smoker, region, charges
- No missing values, 1 duplicate removed.

## Data Preprocessing
- Checked dtypes, info(), describe()
- Encoded categorical variables: sex, smoker, region using LabelEncoder
- Cleaned file saved as cleaned_insurance.csv

## Exploratory Data Analysis (EDA)
- Smokers avg charges: $32,050 vs Non-smokers: $8,434 (3.8x higher)
- Correlation: smoking (0.79) and age (0.29) highly correlated with charges
- Visualizations: Boxplot (smoker vs charges), Scatter (age vs charges, bmi vs charges)

## Model Selection & Training
- Features (X): age, sex, bmi, children, smoker, region
- Target (y): charges
- Train-Test Split: 80% train, 20% test, random_state=42
- Models:
    - Linear Regression (Baseline)
    - Random Forest Regressor (Better)

## Model Evaluation - Phase 6
- Linear Regression Metrics:
    - MAE: 4182.35
    - MSE: 35493102.61
    - RMSE: 5957.61
    - R2 Score: 0.8068 (80.68% accuracy)
- Graphs:
    - Actual vs Predicted Charges
    - Residual Plot

## Prediction Application - Phase 7
Built Gradio App where user enters:
age, BMI, smoking status, children, gender, region -> Predicts estimated cost.

Test Case 1: 25yr female, BMI 22, non-smoker, 0 children, northeast => ~$5,000
Test Case 2: 50yr male, BMI 32, smoker, 2 children, southeast => ~$34,000

## Project Structure
