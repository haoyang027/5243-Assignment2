# File: /bandgap-prediction-project/bandgap-prediction-project/models/gan_model.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

class GAN:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build the generator and discriminator models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])
        
        # Create the GAN model by combining generator and discriminator
        self.gan = self.build_gan()

    def build_generator(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.input_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.output_dim, activation='linear'))  # Output layer for regression
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.output_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
        return model

    def build_gan(self):
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def train(self, X_train, epochs=10000, batch_size=32):
        for epoch in range(epochs):
            # Train the discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            generated_data = self.generator.predict(noise)

            # Labels for real and fake data
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_data, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            valid_labels = np.ones((batch_size, 1))  # Labels for generator's training
            g_loss = self.gan.train_on_batch(noise, valid_labels)

            # Print the progress
            if epoch % 1000 == 0:
                print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.input_dim))
        generated_samples = self.generator.predict(noise)
        return generated_samples

# Comments:
# - The GAN class is designed to encapsulate the generator and discriminator models.
# - The generator creates synthetic data based on random noise, while the discriminator evaluates the authenticity of the data.
# - The training process involves alternating between training the discriminator and the generator.
# - The generator aims to produce data that is indistinguishable from real data, while the discriminator learns to differentiate between real and fake data.
# - The model architecture uses LeakyReLU activations for better gradient flow and a linear activation for the generator's output to predict continuous values.