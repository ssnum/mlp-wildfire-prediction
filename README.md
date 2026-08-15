# MLP Wildfire Prediction

A PyTorch multilayer perceptron (MLP) that predicts wildfire location (latitude and longitude) from historical wildfire and weather data.

## Description

This project was built as part of the UC Davis COSMOS Data Science program by Team The BackPropagators. It takes historical wildfire records joined with weather data and trains a feedforward neural network to predict where a wildfire is likely to occur. Fires prior to 2015 are used for training, and fires from 2015 are held out for testing.

**Website:** https://ucd25-cosmos-thebackpropagators.github.io/project/

## Overview

Inputs are preprocessed with a `ColumnTransformer`, using `StandardScaler` for numeric features and `OneHotEncoder` for categorical features. The processed features are then passed through a feedforward neural network that regresses directly onto `[latitude, longitude]`.

Two versions of the model are included:

- **`notebooks/model_dev.ipynb`**: baseline MLP with the architecture `Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(2)`, trained with MSE loss and the Adam optimizer.
- **`notebooks/model_dev_with_age.ipynb`**: improved architecture that adds a fire age feature, along with `BatchNorm1d` and `Dropout` layers for regularization: `Linear(256) -> BatchNorm -> ReLU -> Dropout -> Linear(128) -> BatchNorm -> ReLU -> Dropout -> Linear(64) -> ReLU -> Linear(2)`.

## My Contribution

I built the preprocessing pipeline, including feature scaling, encoding, and the train/test split by year, along with both versions of the model: architecture design, training loop, and evaluation.

## Tech Stack

PyTorch, scikit-learn, pandas
