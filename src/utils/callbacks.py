import os
import tensorflow as tf
import glob
import logging

logger = logging.getLogger(__name__)

class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, 
                 save_weights_only=False, verbose=1):
        super().__init__(
            filepath=filepath,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            verbose=verbose
        )
        self.mode = mode  # Add this line to store mode as instance variable
        self.best_filepath = None
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def _is_improvement(self, current_value):
        """Check if the current value is an improvement over the best value."""
        if self.mode == 'min':
            return current_value < self.best_value
        return current_value > self.best_value

    def _delete_previous_model(self):
        """Delete the previous best model file if it exists."""
        if self.best_filepath and os.path.exists(self.best_filepath):
            try:
                os.remove(self.best_filepath)
                logger.info(f"Deleted previous model: {self.best_filepath}")
            except OSError as e:
                logger.error(f"Error deleting previous model {self.best_filepath}: {e}")

    def _save_model(self, epoch, batch, logs):
        """Save the model if it's an improvement over the previous best."""
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            logger.warning(f"Monitor value '{self.monitor}' not found in logs")
            return
        
        filepath = self._get_file_path(epoch, batch, logs)
        
        if self.save_best_only:
            if self._is_improvement(current_value):
                logger.info(f"Improvement found! {self.monitor}: {current_value:.5f} "
                          f"(previous best: {self.best_value:.5f})")
                
                # Delete previous best model
                self._delete_previous_model()
                
                # Save new best model
                super()._save_model(epoch, batch, logs)
                
                # Update tracking variables
                self.best_filepath = filepath
                self.best_value = current_value
                
                logger.info(f"Saved new best model to: {filepath}")
            else:
                logger.info(f"No improvement in {self.monitor}: {current_value:.5f} "
                          f"(best: {self.best_value:.5f})")
        else:
            # If not save_best_only, always save the model
            super()._save_model(epoch, batch, logs)
            self.best_filepath = filepath
            logger.info(f"Saved model to: {filepath}")
