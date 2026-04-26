# G-Research-Crypto-Forecasting

California Housing Sequential Prediction Pipeline
This repository contains a data engineering and machine learning project that adapts the standard California Housing dataset into a sequential time-series problem. It utilizes a dual-model approach featuring a Hidden Markov Model (HMM) for geographical regime detection and a Long Short-Term Memory (LSTM) network for price trend forecasting.

Project Overview
The project demonstrates how to transform static tabular data into a sequence-based problem. By treating consecutive rows as a timeline of real estate transactions, the pipeline identifies latent market patterns across different California neighborhoods and predicts the directional movement of property values.

Technical Architecture
1. Feature Engineering
The pipeline derives new metrics to enhance the predictive power of the models:

Price Change: The percentage change in median house value between consecutive dataset entries.

Rooms per Household: A derived metric to measure density and property type.

Targeted Lagging: The labels are shifted to create a supervised learning task for next-step forecasting.

2. Market Regime Detection (HMM)
A Gaussian Hidden Markov Model is utilized to cluster geographical data (longitude and latitude) along with price volatility. This categorizes the market into distinct "Regimes," allowing the subsequent deep learning model to understand the neighborhood context of each data point.

3. Deep Learning Forecast (LSTM)
A stacked LSTM architecture is implemented using TensorFlow and Keras. The model processes 10-step sequences of housing data, including:

Median Income

Housing Median Age

Derived Room Ratios

HMM-generated Regime states

The inclusion of Batch Normalization and Dropout layers ensures stability and prevents overfitting during training.

Tech Stack
Environment: Google Colab / Python 3.x

Data Processing: Pandas, NumPy, Scikit-Learn

Deep Learning: TensorFlow, Keras

Latent State Modeling: hmmlearn

Visualization: Matplotlib

Implementation Details
The project is designed to be fully reproducible within the Google Colab environment. It leverages the built-in sample data directory, eliminating the need for external file uploads or API keys.

Training Parameters
Sequence Length: 10 steps

Train/Test Split: 80/20

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

Epochs: 5

Batch Size: 32

Performance Metrics
The primary evaluation metric for this pipeline is Directional Accuracy. This measures how often the model correctly predicts whether the price of the next entry will increase or decrease relative to the current one.

Portfolio Summary
This project highlights the ability to:

Apply temporal sequence modeling to non-traditional time-series data.

Integrate unsupervised clustering (HMM) with supervised deep learning (LSTM).

Handle data scaling and sequence generation for recurrent neural networks.
