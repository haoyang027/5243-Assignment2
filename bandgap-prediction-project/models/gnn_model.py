# File: /bandgap-prediction-project/bandgap-prediction-project/models/gnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class GNNModel:
    def __init__(self, num_node_features, num_classes):
        """
        Initialize the Graph Neural Network model.
        
        Parameters:
        num_node_features (int): Number of features for each node in the graph.
        num_classes (int): Number of output classes (for regression, this is typically 1).
        """
        self.model = self.build_model(num_node_features, num_classes)

    def build_model(self, num_node_features, num_classes):
        """
        Build the architecture of the GNN model.
        
        Parameters:
        num_node_features (int): Number of features for each node.
        num_classes (int): Number of output classes.
        
        Returns:
        model: A compiled Keras model.
        """
        # Input layer for node features
        inputs = layers.Input(shape=(None, num_node_features))  # Variable number of nodes
        
        # Graph convolutional layers
        x = layers.GraphConv(32, activation='relu')(inputs)  # First GNN layer
        x = layers.GraphConv(16, activation='relu')(x)       # Second GNN layer
        
        # Global pooling layer to aggregate node features
        x = layers.GlobalAveragePooling1D()(x)
        
        # Fully connected layers for regression
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output layer for regression
        outputs = layers.Dense(num_classes, activation='linear')(x)
        
        # Create the model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        return model

    def train(self, train_data, train_labels, validation_data, epochs=100, batch_size=32):
        """
        Train the GNN model.
        
        Parameters:
        train_data (np.array): Training data (features).
        train_labels (np.array): Training labels (targets).
        validation_data (tuple): Tuple of validation data and labels.
        epochs (int): Number of epochs for training.
        batch_size (int): Size of the batches for training.
        """
        self.model.fit(train_data, train_labels, 
                       validation_data=validation_data, 
                       epochs=epochs, 
                       batch_size=batch_size, 
                       verbose=1)

    def evaluate(self, test_data, test_labels):
        """
        Evaluate the GNN model on test data.
        
        Parameters:
        test_data (np.array): Test data (features).
        test_labels (np.array): Test labels (targets).
        
        Returns:
        metrics: Evaluation metrics of the model.
        """
        metrics = self.model.evaluate(test_data, test_labels, verbose=0)
        return metrics

    def predict(self, data):
        """
        Make predictions using the trained GNN model.
        
        Parameters:
        data (np.array): Input data for predictions.
        
        Returns:
        np.array: Predicted values.
        """
        return self.model.predict(data)

# Note: This GNN implementation assumes the use of a GraphConv layer, which may require additional libraries such as Spektral or TensorFlow GNN.
# Ensure that the appropriate libraries are installed and imported for graph operations.