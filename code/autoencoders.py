# autoencoder.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from config import *

class Autoencoder:
    def __init__(self, input_dim, hidden_units):
        """
        Initialize the Autoencoder model.
        
        Args:
            input_dim (int): Number of input features.
            hidden_units (int): Number of units in the hidden layer.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.model = self.build_model()
    
    def build_model(self):
        """Build the autoencoder model."""
        inputs = Input(shape=(self.input_dim,))
        encoded = Dense(self.hidden_units, activation='relu')(inputs)
        decoded = Dense(self.input_dim, activation='linear')(encoded)
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=AE_LEARNING_RATE),
                           loss='mse')
        return autoencoder
    
    def train(self, X_train, X_val, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE):
        """
        Train the autoencoder.
        
        Args:
            X_train (np.ndarray): Training features.
            X_val (np.ndarray): Validation features.
        
        Returns:
            Model: Trained encoder model.
        """
        history = self.model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(X_val, X_val), verbose=1)
        # Extract encoder part
        encoder = Model(inputs=self.model.input, outputs=self.model.layers[1].output)
        return encoder
    
    def extract_features(self, X):
        """Extract features using the encoder."""
        return self.model.predict(X)