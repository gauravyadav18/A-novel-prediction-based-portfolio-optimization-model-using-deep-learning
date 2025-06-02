# config.py
# Configuration parameters for the portfolio optimization pipeline

# Data parameters
DATA_PATHS = [
    "VIX.csv", 
    "VTI.csv",
    "DBC.csv",
    "AGG.csv"
]
DATE_COLUMN = "Date"  # Column name for dates
PRICE_COLUMN = "Close"  # Column name for closing prices
TRAIN_START_DATE = "2015-01-01"  # Customize as needed
TRAIN_END_DATE = "2018-12-31"
VALIDATION_END_DATE = "2019-12-31"
TEST_END_DATE = "2020-12-31"
WINDOW_YEARS = 6
TRAIN_YEARS = 4
VALIDATION_YEARS = 1
TEST_YEARS = 1
LAG_DAYS = 60  # Number of lagged days for features

# Autoencoder parameters
AE_HIDDEN_UNITS = 32
AE_EPOCHS = 100
AE_BATCH_SIZE = 100
AE_LEARNING_RATE = 0.001

# LSTM parameters
LSTM_HIDDEN_UNITS = 64
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.1
LSTM_RECURRENT_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 100
LSTM_LEARNING_RATE = 0.002

# Omega model parameters
DELTA = 0.5 # Risk-return preference
WEIGHTS = [1/8, 2/8, 3/8, 1/8, 1/8]  # Coefficients for objective function
SAMPLE_PERIOD = 10  # T^j
NUM_DISTRIBUTIONS = 2  # Number of normal distributions
THRESHOLD = 0.02  # Stock selection threshold