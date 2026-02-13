data_raw = yf.download(TICKER, period=DATA_PERIOD, interval=DATA_INTERVAL, auto_adjust=True)

# --- DIAGNOSTIC PRINT 1 ---
print("\n--- After yf.download() ---")
print(f"Raw data shape: {data_raw.shape}")
print(f"Raw data columns: {data_raw.columns}")
print("Raw Data Head:\n", data_raw.head())
# --- END DIAGNOSTIC PRINT 1 ---

# Check if data is empty
if data_raw.empty:
    raise ValueError(f"Failed to download data for {TICKER}. DataFrame is empty.")

# Select features (Volume might be NaN sometimes, handle later)
# Check if FEATURES exist before selecting
missing_cols = [col for col in FEATURES if col not in data_raw.columns]
if missing_cols:
     raise KeyError(f"Columns {missing_cols} not found in downloaded data. Available columns: {data_raw.columns}")

data = data_raw[FEATURES].copy() # Use .copy() to avoid SettingWithCopyWarning later

# --- DIAGNOSTIC PRINT 2 ---
print("\n--- After Selecting FEATURES ---")
print(f"Selected data shape: {data.shape}")
print(f"Selected data columns: {data.columns}")
print("Selected Data Head:\n", data.head())
# --- END DIAGNOSTIC PRINT 2 ---

# --- Replace dropna with boolean indexing ---
essential_price_cols = ['Open', 'High', 'Low', 'Close']
# Check if essential columns exist before filtering
missing_essential = [col for col in essential_price_cols if col not in data.columns]
if missing_essential:
    raise KeyError(f"Essential columns {missing_essential} are missing before filtering NaNs. Data columns: {data.columns}")

print(f"\nShape before removing NaNs in essential columns: {data.shape}")
# Keep rows where ALL essential price columns are NOT NaN
data = data[data[essential_price_cols].notna().all(axis=1)]
print(f"Shape after removing NaNs in essential columns: {data.shape}")
# --- End Replace dropna ---


# Fill NaN volume with 0, common practice
if 'Volume' in data.columns:
    data['Volume'] = data['Volume'].fillna(0)
else:
    print("Warning: 'Volume' column not found after NaN filtering.")


print(f"\nFinal data shape for preprocessing: {data.shape}")
if data.empty:
    raise ValueError("DataFrame became empty after removing NaNs. Check downloaded data quality.")
