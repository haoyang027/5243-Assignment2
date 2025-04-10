# File: /bandgap-prediction-project/bandgap-prediction-project/src/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predicted_vs_actual(y_true, y_pred):
    """
    Plots the predicted band gaps against the actual band gaps.

    Parameters:
    y_true (array-like): Actual band gap values.
    y_pred (array-like): Predicted band gap values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue', label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Band Gap (eV)')
    plt.ylabel('Predicted Band Gap (eV)')
    plt.title('Predicted vs. Actual Band Gaps')
    plt.legend()
    plt.grid()
    plt.show()

def plot_loss(history):
    """
    Plots the training and validation loss over epochs.

    Parameters:
    history (History): Keras History object containing training metrics.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()

def plot_feature_importance(importances, feature_names):
    """
    Plots the feature importance for model interpretability.

    Parameters:
    importances (array-like): Importance scores for each feature.
    feature_names (list): Names of the features.
    """
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.ylabel('Importance Score')
    plt.show()