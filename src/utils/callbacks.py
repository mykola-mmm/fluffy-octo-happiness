import os
import tensorflow as tf

class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_filepath = None

    def _save_model(self, epoch, batch, logs):
        # Delete previous best model if it exists
        if self.best_filepath and os.path.exists(self.best_filepath):
            os.remove(self.best_filepath)
        
        # Save new best model
        filepath = self._get_file_path(epoch, batch, logs)
        self.best_filepath = filepath
        super()._save_model(epoch, batch, logs)