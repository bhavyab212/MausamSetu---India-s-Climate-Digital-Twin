from src.utils import denormalize_grid, masked_rmse, masked_mae, masked_pearson_r
import numpy as np

class BaselineModels:
    def __init__(self, X_test, Y_test, scalers, mask):
        self.X_test = X_test
        self.Y_test = Y_test
        self.scalers = scalers
        self.mask = mask

    def _evaluate(self, predictions, name):
        if self.X_test is None or self.Y_test is None:
            return None, None, None, None

        # Denormalize first
        preds_denorm_rain = denormalize_grid(predictions[..., 0], self.scalers, is_temp=False)
        preds_denorm_temp = denormalize_grid(predictions[..., 1], self.scalers, is_temp=True)
        
        y_true_rain = denormalize_grid(self.Y_test[..., 0], self.scalers, is_temp=False)
        y_true_temp = denormalize_grid(self.Y_test[..., 1], self.scalers, is_temp=True)

        # Masked metrics for Rainfall
        rmse_r = masked_rmse(y_true_rain, preds_denorm_rain, self.mask)
        mae_r = masked_mae(y_true_rain, preds_denorm_rain, self.mask)
        corr_r = masked_pearson_r(y_true_rain, preds_denorm_rain, self.mask)

        # Masked metrics for Temperature
        rmse_t = masked_rmse(y_true_temp, preds_denorm_temp, self.mask)
        mae_t = masked_mae(y_true_temp, preds_denorm_temp, self.mask)
        corr_t = masked_pearson_r(y_true_temp, preds_denorm_temp, self.mask)

        print(f"--- {name} Baseline ---")
        print(f"Rainfall -> RMSE: {rmse_r:.2f}, MAE: {mae_r:.2f}, Pearson r: {corr_r:.3f}")
        print(f"Temp     -> RMSE: {rmse_t:.2f}, MAE: {mae_t:.2f}, Pearson r: {corr_t:.3f}")

        # Return averaged metrics over both variables for simple summary if needed
        avg_rmse = (rmse_r + rmse_t) / 2
        avg_mae = (mae_r + mae_t) / 2
        avg_corr = (corr_r + corr_t) / 2

        return predictions, avg_rmse, avg_mae, avg_corr

    def evaluate_persistence(self):
        """Next year = Last year in the sequence window."""
        predictions = self.X_test[:, -1, :, :, :]
        return self._evaluate(predictions, "Persistence")

    def evaluate_climatology(self):
        """Next year = Mean over the sequence window."""
        predictions = np.mean(self.X_test, axis=1)
        return self._evaluate(predictions, "Climatology")
