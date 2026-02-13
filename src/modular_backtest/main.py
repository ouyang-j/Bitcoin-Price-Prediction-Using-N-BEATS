import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import create_nbeats_model
from backtester import CryptoBacktester

# --- Data Preparation Utilities ---
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

# --- Execution ---
if __name__ == "__main__":
    # 1. Initialize Backtester
    tester = CryptoBacktester(assets=['BTC-USD'], start='2023-01-01')
    
    # 2. Scale Data
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(tester.data[['BTC-USD']])
    
    # 3. Prepare Sequences for N-BEATS
    seq_len = 3 
    X, y = create_sequences(scaled_prices, seq_len)
    
    # 4. Build and Train Model
    # Configuration based on institutional benchmarks
    model = create_nbeats_model(
        seq_len=seq_len, horizon=1, num_features=1, num_targets=1,
        n_blocks=3, n_neurons=128, n_layers_per_block=4, theta_size=8
    )
    model.compile(optimizer='adam', loss='mae')
    
    print("Training N-BEATS model for Bitcoin price prediction...")
    model.fit(X, y, epochs=20, batch_size=64, verbose=1)

    # 5. Define Strategy Signal
    def nbeats_signal_fn(data):
        scaled_input = scaler.transform(data[['BTC-USD']])
        X_full, _ = create_sequences(scaled_input, seq_len)
        preds = model.predict(X_full, verbose=0)
        
        # Simple Logic: Long if predicted price > current price
        signals = np.zeros(len(data))
        for i in range(len(preds)):
            # Compare prediction to the last available price in sequence
            if preds[i] > scaled_input[i + seq_len - 1]:
                signals[i + seq_len] = 1 # Shift forward to execute next period
        
        return pd.DataFrame(signals, index=data.index, columns=data.columns)

    # 6. Run and Evaluate
    tester.run_strategy(nbeats_signal_fn)
    tester.evaluate()
