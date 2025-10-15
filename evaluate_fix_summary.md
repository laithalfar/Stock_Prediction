# Evaluate.py Fix Summary

## Issues Fixed

### 1. **Main Function Flow (Lines 496-623)**
**Problem**: 
- Incorrect indexing of fold-specific data (accessing `train_results["X_test"]` instead of `train_results["X_test"][f]`)
- Using undefined variable `reconstructed` before defining it
- Incorrect scaler access (accessing single scaler instead of fold-specific scaler)
- Case sensitivity issue with "close" vs "Close" feature name

**Solution**:
```python
# Extract fold-specific data at the start of each loop
X_test_fold = train_results["X_test"][f]
y_test_fold = train_results["y_test"][f]
X_scaler_fold = train_results["X_scaler_list"][f]
y_scaler_fold = train_results["y_scaler_list"][f]
close_series_fold = train_results["close_te_list"][f]
model_fold = train_results["model_list"][f]
```

### 2. **Reconstruction Logic (Lines 533-563)**
**Problem**: 
- Complex, broken reconstruction flow with undefined variables
- References to non-existent DataFrame `df`

**Solution**:
```python
# Simplified inline reconstruction:
# 1. Get predictions and inverse transform
y_pred_returns = y_scaler_fold.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true_returns = y_scaler_fold.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# 2. Reconstruct Close prices from returns
pred_close_next = close_t * (1 + y_pred_returns)
true_close_next = close_series_fold.values[T:]  # Skip first T timesteps

# 3. Ensure alignment
min_len = min(len(pred_close_next), len(true_close_next), len(close_t))
# Truncate all to same length
```

### 3. **Feature Column Reference (Lines 580-593)**
**Problem**: 
- Using `train_results["feature_columns"]` which doesn't exist
- Should be `train_results["feature_columns_X"]`

**Solution**:
```python
psi = evaluate_distribution_shift(
    pd.DataFrame(X_tr_arr, columns=train_results["feature_columns_X"]),
    pd.DataFrame(X_te_arr, columns=train_results["feature_columns_X"]),
    feature="Close",
    use_z=True
)
```

### 4. **Directory Creation (Lines 629-657)**
**Problem**: 
- No directory creation before saving results
- Could fail if directories don't exist

**Solution**:
```python
# Create directories before saving
results_dir = MODEL_DIR / f"results/{MODEL_TYPE}_results"
results_dir.mkdir(parents=True, exist_ok=True)

results_dir_avg = MODEL_DIR / "results"
results_dir_avg.mkdir(parents=True, exist_ok=True)
```

### 5. **Array Initialization (Lines 490-491, 617-623)**
**Problem**: 
- Using `np.concatenate()` on empty lists causes errors

**Solution**:
```python
# Check if first iteration, then concatenate
if len(actual) == 0:
    actual = reconstructed["True_Close_t+1"]
    predicted = reconstructed["Pred_Close_t+1"]
else:
    actual = np.concatenate((actual, reconstructed["True_Close_t+1"]))
    predicted = np.concatenate((predicted, reconstructed["Pred_Close_t+1"]))
```

### 6. **Removed/Deprecated Function**
**Problem**: 
- `reconstruct_close_from_returns()` function was incomplete and referenced undefined variables

**Solution**:
- Replaced with inline logic in `main()` for clarity
- Commented out old function with deprecation note

## Execution Flow (Now Fixed)

```
main()
  ↓
  1. Call train_pipeline() → get all trained models and data
  ↓
  2. For each fold:
     a. Extract fold-specific: model, X_test, y_test, scalers, close_series
     b. Inverse transform X_test to get base Close prices (close_t)
     c. evaluate_model() → get predictions (scaled returns)
     d. Inverse transform predictions → actual returns
     e. Reconstruct Close prices: close_t * (1 + returns)
     f. Compute metrics: MAE, RMSE on both returns and Close prices
     g. Calculate PSI for distribution shift
     h. Store results
     i. Plot training history
  ↓
  3. Plot actual vs predicted across all folds
  ↓
  4. Save fold-level results to CSV
  ↓
  5. Calculate averaged results across folds
  ↓
  6. Save/append to averaged_results.csv
  ↓
  7. Find best fold model and save separately
  ↓
  8. If all 3 models evaluated → compare and pick best
```

## Key Data Structure from train.py

```python
train_results = {
    "model_list": [model_fold1, model_fold2, ...],      # List of trained models
    "history_list": [history1, history2, ...],          # Training histories
    "X_train": [X_train_fold1, X_train_fold2, ...],     # 3D arrays per fold
    "y_train": [y_train_fold1, y_train_fold2, ...],     # 1D arrays per fold
    "X_test": [X_test_fold1, X_test_fold2, ...],        # 3D arrays per fold
    "y_test": [y_test_fold1, y_test_fold2, ...],        # 1D arrays per fold
    "close_te_list": [close_series1, close_series2, ...], # pd.Series per fold
    "X_scaler_list": [scaler1, scaler2, ...],           # StandardScaler per fold
    "y_scaler_list": [scaler1, scaler2, ...],           # StandardScaler per fold
    "folds": [0, 1, 2, ...],                            # Fold indices
    "feature_columns_X": ["Close", "SMA_ratio", ...]    # Feature names
}
```

## Testing Recommendations

1. **Run with single fold first** - Verify fold-specific extraction works
2. **Check Close price reconstruction** - Print shapes and sample values
3. **Verify directory creation** - Ensure all result folders are created
4. **Monitor PSI calculation** - Catch any feature name mismatches
5. **Check final plots** - Ensure concatenation works across all folds
