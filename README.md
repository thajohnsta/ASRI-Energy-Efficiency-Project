# ASRI Energy Efficiency Project

This project analyzes residential energy usage using the RECS 2009 public dataset. The goal is to explore how poverty indicators, home age, and climate region relate to household electricity consumption.

## Tools Used

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn

## Methods

The project uses a Random Forest regression model to predict household electricity usage measured by `KWH`. The dataset was cleaned by removing missing values and encoding categorical climate regions.

## Features Used

- Poverty indicator at 100%
- Poverty indicator at 150%
- Year the home was built
- Climate region

## Model Evaluation

The model is evaluated using:

- Mean Absolute Error
- Root Mean Squared Error
- R² Score

## Purpose

This project demonstrates basic data cleaning, feature engineering, regression modeling, visualization, and interpretation of residential energy consumption data.
