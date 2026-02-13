import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow Version:", tf.__version__)

# ---------------------------------------------------------------------------
# Configuration Parameters (from Paper)
# ---------------------------------------------------------------------------
TICKER = 'BTC-USD'
DATA_PERIOD = "730d" # Fetch slightly more to ensure enough hourly data after potential gaps
DATA_INTERVAL = "1h"
SEQUENCE_LENGTH = 3 # Input: Use previous 3 hours
FORECAST_HORIZON = 1 # Output: Predict next 1 hour 
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume'] # Adjusted for yfinance auto_adjust=True
TARGETS = ['High', 'Low']
TRAIN_SPLIT_RATIO = 0.8

# N-BEATS Hyperparameters
N_BLOCKS = 3
N_NEURONS = 128 # Units per block
N_STACKS = 1 # Paper doesn't explicitly mention stacks, assuming 1 stack of N_BLOCKS
N_LAYERS_PER_BLOCK = 4 # Common N-BEATS setting, not specified in paper
THETA_SIZE = N_NEURONS # Size of the expansion coefficients, can be tuned
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
