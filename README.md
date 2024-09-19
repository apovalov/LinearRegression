# Linear Models

## Description

This repository contains implementations of linear models using Python:

- **LinearModel**: Abstract base class for linear models.
- **LinearRegression**: Linear regression class, inheriting from `LinearModel`.
- **LogisticRegression**: Logistic regression class, inheriting from `LinearModel`.

All classes include `fit` and `predict` methods, as well as separate methods for computing loss functions and performance metrics. The following hyperparameters are implemented:

- L1 and L2 regularization
- Learning rate
- Verbose

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
