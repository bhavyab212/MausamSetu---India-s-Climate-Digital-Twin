import os
import numpy as np
import tensorflow as tf
from src.model import DigitalTwinModel
from src.unet_model import build_unet

class EnsemblePredictor:
    def __init__(self, seq_length, lat_dim, lon_dim, weights_dir='data'):
        self.seq_length = seq_length
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.member_names = []
        self.weights = None
        self.member_metrics = {}
        
        # Initialize ConvLSTM (from existing DigitalTwinModel, but just using the underlying tf Model)
        from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Add, Lambda
        from tensorflow.keras.models import Model
        
        def build_convlstm():
            inputs = Input(shape=(seq_length, lat_dim, lon_dim, 2))
            last_step = Lambda(lambda x: x[:, -1, :, :, :])(inputs)
            
            x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(inputs)
            x = BatchNormalization()(x)
            x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
            x = BatchNormalization()(x)
            
            residual = Conv2D(filters=2, kernel_size=(1, 1), activation='tanh', padding='same')(x)
            residual = Lambda(lambda x: x * 0.1)(residual)
            outputs = Add()([last_step, residual])
            return Model(inputs=inputs, outputs=outputs)

        self.clstm = build_convlstm()
        self.unet = build_unet(seq_length, lat_dim, lon_dim, 2)
        
        # Load weights (prefer annual model for annual forecast pipeline).
        clstm_daily_path = os.path.join(weights_dir, 'convlstm_daily.weights.h5')
        clstm_path = os.path.join(weights_dir, 'convlstm.weights.h5')
        unet_path = os.path.join(weights_dir, 'unet.weights.h5')
        
        if os.path.exists(clstm_path):
            self.clstm.load_weights(clstm_path)
            self.member_names.append("clstm")
        elif os.path.exists(clstm_daily_path):
            # Fallback when annual weights are unavailable.
            self.clstm.load_weights(clstm_daily_path)
            self.member_names.append("clstm")
        if os.path.exists(unet_path):
            self.unet.load_weights(unet_path)
            self.member_names.append("unet")

        # Trend baseline is always available.
        self.member_names.append("trend")

        if len(self.member_names) <= 1:
            print("Warning: Neural weights not found, using trend-only baseline.")
        else:
            print(f"Ensemble members active: {', '.join(self.member_names)}")

    def _member_predictions(self, X):
        preds = {}
        if "clstm" in self.member_names:
            preds["clstm"] = self.clstm.predict(X, verbose=0)
        if "unet" in self.member_names:
            preds["unet"] = self.unet.predict(X, verbose=0)
        preds["trend"] = X[:, -1, :, :, :]
        return preds

    @staticmethod
    def _masked_rmse(y_true, y_pred, mask=None):
        err = (y_true - y_pred) ** 2
        if mask is None:
            return float(np.sqrt(np.nanmean(err)))
        m = np.expand_dims(mask, axis=(0, -1))
        masked = np.where(m == 1, err, np.nan)
        return float(np.sqrt(np.nanmean(masked)))

    def fit_weights(self, X_val, Y_val, mask=None):
        """Calibrate ensemble weights using inverse-RMSE on a validation split."""
        preds = self._member_predictions(X_val)
        rmses = {}
        for name, pred in preds.items():
            rmse = self._masked_rmse(Y_val, pred, mask=mask)
            if np.isnan(rmse) or rmse <= 0:
                rmse = 1e-6
            rmses[name] = rmse
        inv = np.array([1.0 / rmses[n] for n in preds.keys()], dtype=np.float64)
        w = inv / np.sum(inv)
        self.weights = {name: float(weight) for name, weight in zip(preds.keys(), w)}
        self.member_metrics = {name: {"rmse": float(rmses[name])} for name in preds.keys()}
        return self.weights, self.member_metrics

    def predict(self, X):
        """Returns ensemble mean and standard deviation (uncertainty)"""
        pred_dict = self._member_predictions(X)
        names = list(pred_dict.keys())
        preds = np.stack([pred_dict[n] for n in names], axis=0)

        if self.weights is None:
            # Safe default: equal weights before calibration.
            w = np.ones(len(names), dtype=np.float64) / max(len(names), 1)
        else:
            w = np.array([self.weights.get(n, 0.0) for n in names], dtype=np.float64)
            s = np.sum(w)
            w = w / s if s > 0 else np.ones(len(names), dtype=np.float64) / max(len(names), 1)

        ensemble_mean = np.tensordot(w, preds, axes=(0, 0))
        ensemble_std = np.std(preds, axis=0)
        # Guard against invalid model outputs.
        ensemble_mean = np.nan_to_num(ensemble_mean, nan=0.0, posinf=1.0, neginf=0.0)
        ensemble_std = np.nan_to_num(ensemble_std, nan=0.0, posinf=0.0, neginf=0.0)
        return ensemble_mean, ensemble_std
