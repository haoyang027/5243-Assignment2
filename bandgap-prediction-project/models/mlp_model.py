# File: /bandgap-prediction-project/bandgap-prediction-project/models/mlp_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class MLPModel:
    def __init__(self, input_dim):
        # Initialize the MLP model with the specified input dimension
        self.model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),  # First hidden layer with 64 neurons
            Dense(32, activation='relu'),  # Second hidden layer with 32 neurons
            Dense(1, activation='linear')  # Output layer for regression
        ])
        # Compile the model with Adam optimizer and mean squared error loss
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    def train(self, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32):
        # Train the model on the training data
        history = self.model.fit(X_train, y_train, validation_split=validation_split, 
                                  epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def evaluate(self, X_test, y_test):
        # Evaluate the model on the test data and return predictions and metrics
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return y_pred, mse, r2

    def save_model(self, filepath):
        # Save the trained model to the specified file path
        self.model.save(filepath)

    def load_model(self, filepath):
        # Load a trained model from the specified file path
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)

# Comments:
# - The MLPModel class encapsulates the functionality for building, training, evaluating, and saving/loading a shallow MLP model.
# - The model architecture consists of two hidden layers with ReLU activation, suitable for regression tasks.
# - The Adam optimizer is chosen for its efficiency in training deep learning models.
# - The training method includes options for validation split, epochs, and batch size, allowing flexibility during training.
# - The evaluation method computes performance metrics, providing insights into the model's predictive capabilities.