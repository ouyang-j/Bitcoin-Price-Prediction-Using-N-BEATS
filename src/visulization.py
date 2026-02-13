# Use the test_indices to align predictions with the original dates if needed
# prediction_dates = test_indices[:len(predictions_original)] # Get corresponding dates

# Plot actual vs predicted High/Low prices for a subset of the test data
plot_points = 500
high_col_index = TARGETS.index('High')
low_col_index = TARGETS.index('Low')

plt.figure(figsize=(15, 7))
plt.plot(y_test_original[:plot_points, high_col_index], label='Actual High', color='blue', linewidth=1)
plt.plot(predictions_original[:plot_points, high_col_index], label='Predicted High', color='orange', linestyle='--', linewidth=1)
plt.title(f'High Price Predictions vs Actual (First {plot_points} Test Points)')
plt.xlabel('Time Steps in Test Set')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 7))
plt.plot(y_test_original[:plot_points, low_col_index], label='Actual Low', color='blue', linewidth=1)
plt.plot(predictions_original[:plot_points, low_col_index], label='Predicted Low', color='orange', linestyle='--', linewidth=1)
plt.title(f'Low Price Predictions vs Actual (First {plot_points} Test Points)')
plt.xlabel('Time Steps in Test Set')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
