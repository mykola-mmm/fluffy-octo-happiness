import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.callbacks import CustomModelCheckpoint
from src.utils.warmup_decay_schedule import WarmupDecaySchedule

logger = logging.getLogger(__name__)

class ClassificationModel(tf.keras.Model):
    def __init__(self, input_shape=(768, 768, 3), dropout_rate=0.1, pretrained=True, l1=0.01, l2=0.01):
        super().__init__()
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.pretrained = pretrained
        self.l1 = l1
        self.l2 = l2
        
        # Move build logic here
        weights = 'imagenet' if self.pretrained else None
        self.backbone = tf.keras.applications.VGG19(weights=weights, include_top=False, input_shape=self.input_shape, pooling='max')

        # Binary classification head
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            512, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L1(self.l1),
            # kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2)  # Add L1 and L2 regularization
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1, dtype=tf.float32)

    def call(self, inputs, training=False):
        # x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
        x = tf.keras.applications.vgg19.preprocess_input(inputs)
        x = self.backbone(x, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.output_layer(x)
        return x

    def compile_model(self, stage, steps_per_epoch=None, learning_rate=0.0001, decay_rate=0.9, warmup_epochs=5, min_learning_rate=0.000000001):
        if stage == "tl":
            initial_learning_rate = learning_rate
            decay_steps = steps_per_epoch  # Decay over 5 epochs
            decay_rate = decay_rate  # Reduces LR by 10% each decay_steps
            
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True  # True makes the decay stepwise, False makes it smooth
            )
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0)
        elif stage == "ft":
            warmup_steps = steps_per_epoch * warmup_epochs
            
            # Create the main decay schedule
            decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=steps_per_epoch,
                decay_rate=decay_rate,
                staircase=True
            )
            
            # Create combined warmup and decay schedule
            lr_schedule = WarmupDecaySchedule(
                initial_learning_rate=learning_rate,
                decay_schedule_fn=decay_schedule,
                warmup_steps=warmup_steps,
                min_learning_rate=min_learning_rate
            )
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0)

        loss_optimizer = tf.keras.optimizers.LossScaleOptimizer(optimizer, dynamic=True)

        loss = tf.keras.losses.MeanSquaredError()
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.RootMeanSquaredError(),
        ]

        super().compile(optimizer=loss_optimizer, loss=loss, metrics=metrics)

    def summary(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        logger.debug(f"Model summary: {model.summary()}")
        logger.debug(f"ClassificationModel summary: {self.backbone.summary()}")

    def set_backbone_trainable(self, trainable=False):
        self.backbone.trainable = trainable
        logger.info(f"Backbone trainable: {self.backbone.trainable}")

    def train(self, stage, train_data_loader, val_data_loader, epochs=10, train_steps_per_epoch=None, val_steps_per_epoch=None, save_path=None, logs_path=None, early_stopping_patience=None):
        save_path = os.path.join(save_path, stage)
        os.makedirs(save_path, exist_ok=True)  # Add this line
        logger.info(f"Save directory exists: {os.path.exists(save_path)}")
        # checkpoint_path = os.path.join(save_path, "model_{epoch:02d}-{val_loss:.2f}.keras")
        # logger.info(f"Models will be saved to: {checkpoint_path}")  # Add this line
        # checkpoint_callback = CustomModelCheckpoint(  # Changed from tf.keras.callbacks.ModelCheckpoint
        #     filepath=checkpoint_path,
        #     save_weights_only=False,
        #     save_best_only=True,
        #     monitor='val_loss',
        #     mode='min',
        #     verbose=1
        # )
        
        # reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.2,
        #     patience=3,
        #     min_lr=1e-10,
        #     verbose=1,
        #     mode='min'
        # )

        log_dir = os.path.join(logs_path, stage)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2
        )

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,  # Number of epochs with no improvement after which training will be stopped
            verbose=1,
            mode='min',
            restore_best_weights=True  # Restores the model weights from the epoch with the best value of the monitored quantity
        )
        
        history = self.fit(
            train_data_loader,
            steps_per_epoch=train_steps_per_epoch,
            epochs=epochs,
            validation_data=val_data_loader,
            validation_steps=val_steps_per_epoch,
            # callbacks=[reduce_lr_callback, tensorboard_callback]
            # callbacks=[checkpoint_callback,reduce_lr_callback, tensorboard_callback, early_stopping_callback]
            # callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback]
            callbacks=[tensorboard_callback, early_stopping_callback]
        )

        if stage == "tl":
            self.history_tl = history
        elif stage == "ft":
            self.history_ft = history

        # Save final model
        final_model_path = os.path.join(save_path, "final_model.keras")
        final_weights_path = os.path.join(save_path, "final_weights.weights.h5")
        self.save(final_model_path)
        self.save_weights(final_weights_path)
        logger.info(f"Final model saved to: {final_model_path}")

    def visualize_history(self, stage):
        if stage == "tl":
            history = self.history_tl
            mae_key = 'mean_absolute_error'
            rmse_key = 'root_mean_squared_error'
            val_mae_key = 'val_mean_absolute_error'
            val_rmse_key = 'val_root_mean_squared_error'
        elif stage == "ft":
            history = self.history_ft
            mae_key = 'mean_absolute_error'
            rmse_key = 'root_mean_squared_error'
            val_mae_key = 'val_mean_absolute_error'
            val_rmse_key = 'val_root_mean_squared_error'

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot training metrics
        axes[0, 0].plot(history.history[mae_key], label='Train MAE')
        axes[0, 0].plot(history.history[rmse_key], label='Train RMSE')
        axes[0, 0].set_title('Training Metrics')
        axes[0, 0].legend()

        # Plot validation metrics
        axes[0, 1].plot(history.history[val_mae_key], label='Val MAE')
        axes[0, 1].plot(history.history[val_rmse_key], label='Val RMSE')
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].legend()

        # Plot training loss
        axes[1, 0].plot(history.history['loss'], label='Train Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()

        # Plot validation loss
        axes[1, 1].plot(history.history['val_loss'], label='Val Loss')
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def run_inference(self, data_loader):
        for batch_index, (x, y) in enumerate(data_loader):
            if batch_index >= 10:
                break
            pred = self(x)
            for i in range(len(pred)):
                plt.imshow(x[i] / 255.0)
                plt.title(f"True: {y[i]}, Predicted: {pred[i].numpy()[0]:.1f}")
                plt.axis('off')
                plt.show()



