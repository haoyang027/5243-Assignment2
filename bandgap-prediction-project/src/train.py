import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp_model import MLPModel  # Import the MLPModel class
from src.data_preprocessing import preprocess_data  # Import the preprocess_data function

def train_model(mlp_model, features, target):
    # Load the dataset
    data = pd.read_csv('../data/Bandgap_data.csv')

    # Preprocess the data
    features, target = preprocess_data(data)
    
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"First few rows of features: {features[:5]}")
    print(f"First few rows of target: {target[:5]}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the model
    input_dim = X_train.shape[1]
    mlp_model = MLPModel(input_dim=input_dim)  # Create an instance of the MLPModel class

    # Train the model
    history = mlp_model.train(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32)

    # Save the trained model
    mlp_model.save_model('mlp_bandgap_model.h5')

    # Evaluate the model on the test set
    y_pred, mse, r2 = mlp_model.evaluate(X_test, y_test)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")