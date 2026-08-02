import tensorflow as tf
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, SpatialDropout2D, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from src.utils import denormalize_grid, masked_rmse, masked_mae, masked_pearson_r

class DigitalTwinModel:
    def __init__(self, seq_length, lat_dim, lon_dim, channels=2, binary_mask=None, scalers=None):
        self.seq_length = seq_length
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim
        self.channels = channels
        self.scalers = scalers
        self.mask = binary_mask
        
        # Create a TF constant for the mask to use in the loss function
        if binary_mask is not None:
            # Reshape mask from [lat, lon] to [1, lat, lon, 1] so it broadcasts over batches and channels
            self.mask_tensor = tf.constant(np.expand_dims(np.expand_dims(binary_mask, axis=0), axis=-1), dtype=tf.float32)
        else:
            self.mask_tensor = tf.constant(1.0, dtype=tf.float32)

        self.model = self._build_model()
        
    def masked_mse_loss(self, y_true, y_pred):
        diff = tf.square(y_true - y_pred)
        masked_diff = diff * self.mask_tensor
        return tf.reduce_sum(masked_diff) / (tf.reduce_sum(self.mask_tensor) + tf.keras.backend.epsilon())

    def masked_huber_loss(self, y_true, y_pred, delta=0.08):
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, delta)
        linear = abs_err - quadratic
        huber = 0.5 * tf.square(quadratic) + delta * linear
        masked = huber * self.mask_tensor
        return tf.reduce_sum(masked) / (tf.reduce_sum(self.mask_tensor) + tf.keras.backend.epsilon())

    def _build_model(self):
        from tensorflow.keras.layers import Add, Lambda
        
        inputs = Input(shape=(self.seq_length, self.lat_dim, self.lon_dim, self.channels))
        
        # Extract the last time step to act as a Persistence baseline
        last_step = Lambda(lambda x: x[:, -1, :, :, :], name='persistence_baseline')(inputs)
        
        # Spatiotemporal feature extraction
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        
        x = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
        x = BatchNormalization()(x)
        
        # Learn the residual (change from last year) instead of the absolute value
        residual = Conv2D(filters=self.channels, kernel_size=(1, 1), activation='tanh', padding='same')(x)
        
        # Scale residual down slightly so initialization favors pure persistence
        residual = Lambda(lambda x: x * 0.1, name='scale_residual')(residual)
        
        # Add residual to persistence and clip to valid normalized range [0, 1]
        added = Add()([last_step, residual])
        outputs = Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0), name='clip_output')(added)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=self.masked_huber_loss, metrics=['mae'])
        return model
        
    def train(
        self,
        X_train,
        Y_train=None,
        X_val=None,
        Y_val=None,
        epochs=50,
        batch_size=2,
        model_path=None,
        steps_per_epoch=None,
        validation_steps=None,
    ):
        import os
        if model_path is None:
            model_path = os.path.join('data', 'convlstm.weights.h5')

        monitor_metric = 'val_loss' if X_val is not None else 'loss'
        callbacks = [
            EarlyStopping(monitor=monitor_metric, patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=6, min_lr=1e-5, verbose=1),
            ModelCheckpoint(
                filepath=model_path,
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
        
        is_dataset = isinstance(X_train, tf.data.Dataset)
        if is_dataset:
            val_data = X_val if isinstance(X_val, tf.data.Dataset) else None
            fit_kw = dict(
                x=X_train,
                validation_data=val_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1,
            )
            if steps_per_epoch is not None:
                fit_kw["steps_per_epoch"] = int(steps_per_epoch)
            if validation_steps is not None and val_data is not None:
                fit_kw["validation_steps"] = int(validation_steps)
            history = self.model.fit(**fit_kw)
        else:
            val_data = (X_val, Y_val) if X_val is not None and Y_val is not None else None
            history = self.model.fit(
                X_train, Y_train,
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        # Keep a final snapshot too (best checkpoint is already saved).
        self.model.save_weights(model_path)
        return history
        
    def predict(self, X):
        # Suppress TF progress output silently
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test, Y_test):
        if self.scalers is None or self.mask is None:
            print("Scalers and mask required for proper evaluation.")
            return None, None, None, None

        predictions = self.predict(X_test)
        
        preds_denorm_rain = denormalize_grid(predictions[..., 0], self.scalers, is_temp=False)
        preds_denorm_temp = denormalize_grid(predictions[..., 1], self.scalers, is_temp=True)
        
        y_true_rain = denormalize_grid(Y_test[..., 0], self.scalers, is_temp=False)
        y_true_temp = denormalize_grid(Y_test[..., 1], self.scalers, is_temp=True)

        rmse_r = masked_rmse(y_true_rain, preds_denorm_rain, self.mask)
        mae_r = masked_mae(y_true_rain, preds_denorm_rain, self.mask)
        corr_r = masked_pearson_r(y_true_rain, preds_denorm_rain, self.mask)

        rmse_t = masked_rmse(y_true_temp, preds_denorm_temp, self.mask)
        mae_t = masked_mae(y_true_temp, preds_denorm_temp, self.mask)
        corr_t = masked_pearson_r(y_true_temp, preds_denorm_temp, self.mask)

        print(f"--- ConvLSTM Evaluation ---")
        print(f"Rainfall -> RMSE: {rmse_r:.2f}, MAE: {mae_r:.2f}, Pearson r: {corr_r:.3f}")
        print(f"Temp     -> RMSE: {rmse_t:.2f}, MAE: {mae_t:.2f}, Pearson r: {corr_t:.3f}")

        avg_rmse = (rmse_r + rmse_t) / 2
        avg_mae = (mae_r + mae_t) / 2
        avg_corr = (corr_r + corr_t) / 2

        return predictions, avg_rmse, avg_mae, avg_corr
