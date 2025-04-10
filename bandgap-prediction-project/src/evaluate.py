# File: /bandgap-prediction-project/bandgap-prediction-project/src/evaluate.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

def evaluate_model(model_path, X_test, y_test):
    """
    Evaluates the performance of a trained model on the test set.

    Parameters:
    - model_path: str, path to the saved model.
    - X_test: np.array, features of the test set.
    - y_test: np.array, true target values of the test set.

    Returns:
    - mse: float, Mean Squared Error of the predictions.
    - r2: float, R^2 score of the predictions.
    - y_pred: np.array, predicted values for the test set.
    """
    # Load the trained model
    model = load_model(model_path)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_pred

def print_evaluation_results(mse, r2):
    """
    Prints the evaluation results.

    Parameters:
    - mse: float, Mean Squared Error.
    - r2: float, R^2 score.
    """
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

def save_predictions(y_pred, output_path):
    """
    Saves the predictions to a CSV file.

    Parameters:
    - y_pred: np.array, predicted values.
    - output_path: str, path to save the predictions.
    """
    predictions_df = pd.DataFrame(y_pred, columns=['Predicted Band Gap'])
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")