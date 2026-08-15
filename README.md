# MLP Wildfire Prediction

A PyTorch multilayer perceptron (MLP) that predicts wildfire location (latitude/longitude) from historical wildfire and weather data. Built as part of the UC Davis COSMOS Data Science program (Team: The BackPropagators).

**Live project site:** https://ucd25-cosmos-thebackpropagators.github.io/project/

## Overview

The model is trained on wildfire records joined with weather data, using fires prior to 2015 as the training set and 2015 fires as the held-out test set. Inputs are preprocessed with a `ColumnTransformer` (`StandardScaler` for numeric features, `OneHotEncoder` for categorical features), then passed through a feedforward neural network that regresses directly onto `[latitude, longitude]`.

Two versions of the model are included:

- **`notebooks/model_dev.ipynb`** — baseline MLP: `Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(2)`, trained with MSE loss and the Adam optimizer.
- **`notebooks/model_dev_with_age.ipynb`** — improved architecture with an additional "fire age" feature, plus `BatchNorm1d` and `Dropout` layers for regularization:
  `Linear(256) -> BatchNorm -> ReLU -> Dropout -> Linear(128) -> BatchNorm -> ReLU -> Dropout -> Linear(64) -> ReLU -> Linear(2)`

## My contribution

I built the preprocessing pipeline (feature scaling/encoding, train/test split by year) and both versions of the model — architecture design, training loop, and evaluation.

## Tech stack

PyTorch, scikit-learn, pandas
