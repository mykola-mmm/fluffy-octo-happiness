import os
import tensorflow as tf
import glob

class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_filepath = None

    def _save_model(self, epoch, batch, logs):
        # Delete previous model files if they exist
        if self.best_filepath:
            try:
                os.remove(self.best_filepath)
            except OSError as e:
                print(f"Error deleting file {self.best_filepath}: {e}")
            # model_dir = os.path.dirname(self.best_filepath)
            # # Get the base filename without epoch and metrics
            # base_filename = os.path.basename(self.best_filepath).split('model_')[0]
            
            # # Find and delete previous model files
            # model_pattern = os.path.join(model_dir, f"model_*.keras")
            # for old_model in glob.glob(model_pattern):
            #     try:
            #         os.remove(old_model)
            #     except OSError as e:
            #         print(f"Error deleting file {old_model}: {e}")
        
        # Save new best model
        filepath = self._get_file_path(epoch, batch, logs)
        self.best_filepath = filepath
        super()._save_model(epoch, batch, logs)
