# Scaling - Fit only on training data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Split data chronologically BEFORE scaling
split_index = int(TRAIN_SPLIT_RATIO * len(data))
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Fit scalers ONLY on training data
scaled_train_features = feature_scaler.fit_transform(train_data[FEATURES])
# Fit target scaler on High/Low columns of training data
target_scaler.fit(train_data[TARGETS])

# Transform train and test data using fitted scalers
scaled_test_features = feature_scaler.transform(test_data[FEATURES])

# Combine scaled features into DataFrames for easier indexing
scaled_train_df = pd.DataFrame(scaled_train_features, columns=FEATURES, index=train_data.index)
scaled_test_df = pd.DataFrame(scaled_test_features, columns=FEATURES, index=test_data.index)

# Also scale the target columns separately for sequence creation
scaled_train_targets_df = pd.DataFrame(target_scaler.transform(train_data[TARGETS]), columns=TARGETS, index=train_data.index)
scaled_test_targets_df = pd.DataFrame(target_scaler.transform(test_data[TARGETS]), columns=TARGETS, index=test_data.index)


# Sequence Creation Function [cite: 132]
def create_sequences(input_features_df, input_targets_df, sequence_length, forecast_horizon):
    X, y = [], []
    indices = []
    # Ensure target indices align with the end of the input sequence
    for i in range(len(input_features_df) - sequence_length - forecast_horizon + 1):
        feature_sequence = input_features_df.iloc[i:(i + sequence_length)].values
        target_values = input_targets_df.iloc[i + sequence_length : i + sequence_length + forecast_horizon].values

        # Ensure the target shape is (forecast_horizon, num_targets) -> (1, 2)
        if target_values.shape == (forecast_horizon, len(TARGETS)):
           X.append(feature_sequence)
           y.append(target_values.flatten()) # Flatten to (2,) for Dense layer output
           indices.append(input_targets_df.index[i + sequence_length]) # Store index of the target time step
        # else: # Debugging shape issues
            # print(f"Skipping index {i} due to shape mismatch: Target shape {target_values.shape}")

    return np.array(X), np.array(y), indices

# Create sequences for training and testing sets
X_train, y_train, train_indices = create_sequences(scaled_train_df, scaled_train_targets_df, SEQUENCE_LENGTH, FORECAST_HORIZON)
X_test, y_test, test_indices = create_sequences(scaled_test_df, scaled_test_targets_df, SEQUENCE_LENGTH, FORECAST_HORIZON)

print(f"X_train shape: {X_train.shape}") # Should be (num_samples, 3, num_features)
print(f"y_train shape: {y_train.shape}") # Should be (num_samples, 2)
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Verify shapes
num_features = len(FEATURES)
num_targets = len(TARGETS)
if X_train.shape[1:] != (SEQUENCE_LENGTH, num_features):
    raise ValueError(f"X_train shape mismatch: expected (None, {SEQUENCE_LENGTH}, {num_features}), got {X_train.shape}")
if y_train.shape[1:] != (num_targets,):
     raise ValueError(f"y_train shape mismatch: expected (None, {num_targets}), got {y_train.shape}")
