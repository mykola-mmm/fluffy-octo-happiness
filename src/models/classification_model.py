#TODO:
# 1. Add adamw optimizer
# 2. Add lr reducer for tl
# 3. Add lr warmup for ft

import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
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
        self.build()
        
    def build(self):
        if self.pretrained:
            weights = 'imagenet'
        else:
            weights = None

        self.backbone = tf.keras.applications.VGG19(weights=weights, include_top=False, input_shape=self.input_shape, pooling='max')

        # Binary classification head
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            512, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2)  # Add L1 and L2 regularization
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)

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

        # elif stage == "ft":
        #     # Warmup settings
        #     warmup_epochs = warmup_epochs
        #     warmup_steps = steps_per_epoch * warmup_epochs
            
        #     # Create warmup schedule that linearly increases from 0 to target learning rate
        #     warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        #         initial_learning_rate=min_learning_rate,
        #         decay_steps=warmup_steps,
        #         end_learning_rate=learning_rate,
        #         power=1.0  # Linear warmup
        #     )
            
        #     # Create main learning rate schedule (after warmup)
        #     main_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #         initial_learning_rate=learning_rate,
        #         decay_steps=steps_per_epoch,
        #         decay_rate=decay_rate,
        #         staircase=True
        #     )
            
        #     # Combine warmup and main schedules
        #     lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #         boundaries=[warmup_steps],
        #         values=[warmup_schedule, main_schedule]
        #     )
            
        #     optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0)

        # optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        metrics = [
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.F1Score(threshold=0.5),
        ]

        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        logger.debug(f"Model summary: {model.summary()}")
        logger.debug(f"ClassificationModel summary: {self.backbone.summary()}")

    def set_backbone_trainable(self, trainable=False):
        self.backbone.trainable = trainable
        logger.info(f"Backbone trainable: {self.backbone.trainable}")

    def train(self, stage, train_data_loader, val_data_loader, epochs=10, train_steps_per_epoch=None, val_steps_per_epoch=None, save_path=None, logs_path=None):
        save_path = os.path.join(save_path, stage)
        checkpoint_path = os.path.join(save_path, "model_{epoch:02d}-{val_loss:.2f}.keras")
        checkpoint_callback = CustomModelCheckpoint(  # Changed from tf.keras.callbacks.ModelCheckpoint
            filepath=checkpoint_path,
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )
        
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-10,
            verbose=1,
            mode='min'
        )

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
            patience=10,  # Number of epochs with no improvement after which training will be stopped
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
            callbacks=[checkpoint_callback,reduce_lr_callback, tensorboard_callback]
        )

        if stage == "tl":
            self.history_tl = history
        elif stage == "ft":
            self.history_ft = history
        
        

    def visualize_history(self, stage):
        if stage == "tl":
            history = self.history_tl
            f1score_key = 'f1_score'
            recall_key = 'recall'
            precision_key = 'precision'
            val_f1score_key = 'val_f1_score'
            val_recall_key = 'val_recall'
            val_precision_key = 'val_precision'
        elif stage == "ft":
            history = self.history_ft
            f1score_key = 'f1_score'
            recall_key = 'recall_1'
            precision_key = 'precision_1'
            val_f1score_key = 'val_f1_score'
            val_recall_key = 'val_recall_1'
            val_precision_key = 'val_precision_1'

        # logger.info(f"Visualizing history")
        # logger.info(f"History: {self.history.history}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a figure with 4 subplots

        # Plot training metrics
        axes[0, 0].plot(history.history[f1score_key], label='Train F1 Score')
        axes[0, 0].plot(history.history[recall_key], label='Train Recall')
        axes[0, 0].plot(history.history[precision_key], label='Train Precision')
        axes[0, 0].set_title('Training Metrics')
        axes[0, 0].legend()

        # Plot validation metrics
        axes[0, 1].plot(history.history[val_f1score_key], label='Val F1 Score')
        axes[0, 1].plot(history.history[val_recall_key], label='Val Recall')
        axes[0, 1].plot(history.history[val_precision_key], label='Val Precision')
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
            if batch_index >= 5:
                break
            pred = self(x)
            for i in range(len(pred)):
                plt.imshow(x[i] / 255.0)  # Normalize the image for display
                plt.title(f"True: {y[i]}, Predicted: {pred[i].numpy()[0]:.2f}")
                plt.axis('off')
                plt.show()
