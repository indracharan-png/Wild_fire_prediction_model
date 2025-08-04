# wild_fire_prediction_model
This project aims to apply machine learning and deep learning techniques to predict the likelihood of wildfires using historical weather and seasonal data.


# Wildfire Risk Prediction with Machine Learning and Deep Learning

## Overview

This repository contains the implementation and evaluation of advanced machine learning and deep learning models aimed at predicting wildfire occurrences using historical weather data. The project integrates robust ensemble methods and deep neural networks, achieving highly accurate and interpretable predictions that support proactive disaster management efforts.

## Problem Statement

Wildfires have significantly increased in frequency and intensity, causing devastating losses to lives, ecosystems, and property, particularly in California. This project addresses the urgent need for effective predictive systems that enable timely intervention and resource allocation.

## Dataset

The project leverages the publicly available [California Weather and Fire Prediction Dataset (1984-2025)](https://scholars.georgiasouthern.edu/en/datasets/california-weather-and-fire-prediction-dataset-19842025-with-engi). Key features include temperature, precipitation, wind speed, humidity, seasonal data, and historical fire occurrences.

## Models Developed

* **Random Forest Classifier**: A robust baseline model offering strong generalization.
* **XGBoost Classifier**: Enhanced gradient boosting method providing superior performance.
* **Baseline Multi-Layer Perceptron (MLP)**: Initial deep learning model to capture complex patterns.
* **Optimized MLP**: Advanced neural network architecture incorporating regularization techniques like dropout and batch normalization, yielding the best performance.

## Performance Metrics

| Model             | ROC-AUC    | Recall  |
| ----------------- | ---------- | ------- |
| Random Forest     | 0.8400     | 60%     |
| XGBoost           | 0.8582     | 65%     |
| Baseline MLP      | 0.8266     | 63%     |
| **Optimized MLP** | **0.8435** | **73%** |

## Fire Risk Scoring System

To enhance interpretability, predictions are converted into actionable risk levels:

* **Low**: ≤ 25%
* **Medium**: 26%-50%
* **High**: 51%-75%
* **Very High**: > 75%

## Project Structure

* `data_utils.py`: Data preprocessing utilities
* `random_forest_model.py`: Random Forest implementation
* `xgboost_model.py`: XGBoost implementation
* `mlp_model.py`: Baseline neural network
* `optimized_mlp_model.py`: Optimized neural network architecture
* `fire_risk_scorer.py`: Converts model probabilities into interpretable risk scores
* `compare_models.py`: Evaluation and comparison across models
* `plot_utils.py`: Visualization tools for performance evaluation

## Requirements

* Python
* scikit-learn
* XGBoost
* PyTorch
* Pandas
* NumPy
* Matplotlib

## Future Work

* Incorporation of LSTM models
* Expansion with additional geographic and human activity data
* Development of a real-time prediction dashboard


