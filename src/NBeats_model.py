class NBeatsBlock(layers.Layer):
    def __init__(self, input_size, theta_size, horizon, n_neurons, n_layers, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.theta_size = theta_size
        # Fully connected layers (stack)
        self.hidden = [layers.Dense(n_neurons, activation='relu') for _ in range(n_layers)]
        # Output layers for basis coefficients (thetas)
        self.theta_f = layers.Dense(theta_size, activation='linear', name='theta_f') # Forecast
        self.theta_b = layers.Dense(theta_size, activation='linear', name='theta_b') # Backcast

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta_f = self.theta_f(x)
        theta_b = self.theta_b(x)
        # Simplified basis expansion - direct use of thetas
        # Final projection handled in the main model build
        return theta_b, theta_f

# Build the full N-BEATS model with residual connections (Revised forecast_sum handling)
def create_nbeats_model(seq_len, horizon, num_features, num_targets, n_blocks, n_neurons, n_layers_per_block, theta_size):
    input_layer = layers.Input(shape=(seq_len, num_features), name='model_input')
    # Flatten the input sequence for Dense layers
    x_flat = layers.Flatten()(input_layer)

    residuals = x_flat
    forecast_sum = None # Initialize as None

    for i in range(n_blocks):
        block = NBeatsBlock(input_size=seq_len * num_features, # Block operates on flattened input
                            theta_size=theta_size,
                            horizon=horizon,
                            n_neurons=n_neurons,
                            n_layers=n_layers_per_block,
                            name=f'nbeats_block_{i}')
        # Get backcast and forecast thetas from the block
        theta_b, theta_f = block(residuals)

        # Simplified backcast/forecast generation
        backcast_layer = layers.Dense(seq_len * num_features, name=f'backcast_{i}')
        backcast = backcast_layer(theta_b)

        forecast_layer = layers.Dense(horizon * num_targets, name=f'forecast_{i}')
        forecast = forecast_layer(theta_f) # Shape (batch_size, horizon * num_targets)

        # Double Residual Connection: Subtract backcast from input to block
        residuals = layers.subtract([residuals, backcast], name=f'subtract_{i}')

        # Add the block's forecast to the total forecast
        if forecast_sum is None:
            # Initialize with the first block's forecast
            forecast_sum = forecast
        else:
            # Add subsequent forecasts
            forecast_sum = layers.add([forecast_sum, forecast], name=f'add_{i}')

    # Ensure final output shape is (batch_size, num_targets) for horizon=1
    if horizon == 1:
      # If horizon is 1, forecast_sum already has shape (batch_size, num_targets)
      final_forecast = forecast_sum
    else:
      # If horizon > 1, reshape might be needed
      final_forecast = layers.Reshape((horizon, num_targets), name='final_forecast')(forecast_sum)
      # If you need only the first step of a multi-step forecast:
      # final_forecast = layers.Lambda(lambda x: x[:, 0, :], name='first_step_forecast')(final_forecast)


    model = tf.keras.Model(inputs=input_layer, outputs=final_forecast, name='NBEATS_model')
    return model

# Instantiate the model
model = create_nbeats_model(
    seq_len=SEQUENCE_LENGTH,
    horizon=FORECAST_HORIZON,
    num_features=num_features,
    num_targets=num_targets,
    n_blocks=N_BLOCKS,
    n_neurons=N_NEURONS,
    n_layers_per_block=N_LAYERS_PER_BLOCK,
    theta_size=THETA_SIZE
)

# Compile the model
model.compile(loss='mae', # Mean Absolute Error
              optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=['mae'])

model.summary()
