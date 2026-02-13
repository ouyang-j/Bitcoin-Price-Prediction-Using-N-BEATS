# Make predictions on the test set
predictions_scaled = model.predict(X_test)

# Inverse transform predictions and actual values
# Predictions shape: (num_samples, num_targets)
# y_test shape: (num_samples, num_targets)
predictions_original = target_scaler.inverse_transform(predictions_scaled)
y_test_original = target_scaler.inverse_transform(y_test)

# Calculate metrics on the original scale
mae = mean_absolute_error(y_test_original, predictions_original)
r2 = r2_score(y_test_original, predictions_original)

print(f"\nTest Set Evaluation (Original Scale):")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"R-squared (R2 Score):      {r2:.6f}")

# Comparison with paper's results (Note potential ambiguity/typo in paper's reported values)
# Abstract: MAE 0.9998, R2 0.00240 [cite: 7]
# Results : MAE 0.00240, R2 0.9998 [cite: 156, 158]
# Assuming results section values are more likely intended for scaled data
print("\nComparison with Paper (Results Section - likely scaled values):")
print(f"Paper MAE (Scaled?): 0.00240")
print(f"Paper R2  (Scaled?): 0.9998")
print("\nComparison with Paper (Abstract - likely scaled values):")
print(f"Paper MAE (Scaled?): 0.9998")
print(f"Paper R2  (Scaled?): 0.00240")
print("\nNote: The MAE/R2 calculated here are on the *original price scale*,")
print("while the paper's values might be on the *scaled* data (0-1 range).")
print("Direct comparison might be misleading without knowing the paper's exact evaluation scale.")
