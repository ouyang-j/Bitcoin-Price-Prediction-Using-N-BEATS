early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10, # Stop if no improvement for 10 epochs
                                                  restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 factor=0.2, # Reduce LR by factor of 5
                                                 patience=5,
                                                 min_lr=LEARNING_RATE / 100)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1, # Use last 10% of training data for validation during training
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss History')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.legend()
plt.show()
