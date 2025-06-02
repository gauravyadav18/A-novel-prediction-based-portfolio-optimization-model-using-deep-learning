# lstm_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from config import *

class LSTMModel:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim  # Set to num_indices (e.g., 4)
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
            Input(shape=(1, self.input_dim)),
            LSTM(LSTM_HIDDEN_UNITS, 
                 dropout=LSTM_DROPOUT, 
                 recurrent_dropout=LSTM_RECURRENT_DROPOUT),
            Dense(self.output_dim, activation='relu')  # Output size = num_indices
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LSTM_LEARNING_RATE),
                      loss='mse')
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE):
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_reshaped = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        self.model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size,
                       validation_data=(X_val_reshaped, y_val), verbose=1)
    
    def predict(self, X):
        X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        return self.model.predict(X_reshaped)