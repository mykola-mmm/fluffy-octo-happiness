import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision
from src.utils.callbacks import CustomModelCheckpoint
from src.utils.warmup_decay_schedule import WarmupDecaySchedule

logger = logging.getLogger(__name__)

class SegmentationModel(tf.keras.Model):
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

        # VGG16 backbone as encoder
        self.backbone = tf.keras.applications.VGG16(
            weights=weights, 
            include_top=False, 
            input_shape=self.input_shape
        )
        
        # Get the encoder blocks for skip connections
        self.block1_conv2 = self.backbone.get_layer('block1_conv2').output
        self.block2_conv2 = self.backbone.get_layer('block2_conv2').output
        self.block3_conv3 = self.backbone.get_layer('block3_conv3').output
        self.block4_conv3 = self.backbone.get_layer('block4_conv3').output
        self.block5_conv3 = self.backbone.get_layer('block5_conv3').output

        # Decoder blocks
        self.up_conv4 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
        self.up_block4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')
        ])

        self.up_conv3 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        self.up_block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')
        ])

        self.up_conv2 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.up_block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        ])

        self.up_conv1 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.up_block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        ])

        # Final output layer
        self.final_conv = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')

    def call(self, inputs, training=False):
        # Encoder path (VGG16)
        x = self.backbone.input
        block1_out = self.block1_conv2
        block2_out = self.block2_conv2
        block3_out = self.block3_conv3
        block4_out = self.block4_conv3
        encoder_out = self.block5_conv3

        # Decoder path with skip connections
        x = self.up_conv4(encoder_out)
        x = tf.keras.layers.Concatenate()([x, block4_out])
        x = self.up_block4(x)

        x = self.up_conv3(x)
        x = tf.keras.layers.Concatenate()([x, block3_out])
        x = self.up_block3(x)

        x = self.up_conv2(x)
        x = tf.keras.layers.Concatenate()([x, block2_out])
        x = self.up_block2(x)

        x = self.up_conv1(x)
        x = tf.keras.layers.Concatenate()([x, block1_out])
        x = self.up_block1(x)

        # Final 1x1 convolution
        outputs = self.final_conv(x)

        return outputs
    
    def summary(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        logger.debug(f"Model summary: {model.summary()}")
        logger.debug(f"SegmentationModel summary: {self.backbone.summary()}")

    def visualize_model(self):
        # Create a Keras Model instance to visualize the full architecture
        inputs = tf.keras.Input(shape=self.input_shape)
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        # Plot the full model architecture
        tf.keras.utils.plot_model(
            model, 
            to_file='model_architecture.png',
            show_shapes=True, 
            show_layer_names=True
        )
        
        # Display the saved image using matplotlib
        plt.figure(figsize=(15, 15))
        img = plt.imread('model_architecture.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    # def compile_model(self, stage, steps_per_epoch=None, learning_rate=0.0001, decay_rate=0.9, warmup_epochs=5, min_learning_rate=0.000000001):
    #     if stage == "tl":
    #         initial_learning_rate = learning_rate
    #         decay_steps = steps_per_epoch  # Decay over 5 epochs
    #         decay_rate = decay_rate  # Reduces LR by 10% each decay_steps
            
    #         lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #             initial_learning_rate=initial_learning_rate,
    #             decay_steps=decay_steps,
    #             decay_rate=decay_rate,
    #             staircase=True  # True makes the decay stepwise, False makes it smooth
    #         )
    #         optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0)
    #     elif stage == "ft":
    #         warmup_steps = steps_per_epoch * warmup_epochs
            
    #         # Create the main decay schedule
    #         decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #             initial_learning_rate=learning_rate,
    #             decay_steps=steps_per_epoch,
    #             decay_rate=decay_rate,
    #             staircase=True
    #         )
            
    #         # Create combined warmup and decay schedule
    #         lr_schedule = WarmupDecaySchedule(
    #             initial_learning_rate=learning_rate,
    #             decay_schedule_fn=decay_schedule,
    #             warmup_steps=warmup_steps,
    #             min_learning_rate=min_learning_rate
    #         )
    #         optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0)


    #     loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    #     metrics = [
    #         tf.keras.metrics.Recall(),
    #         tf.keras.metrics.Precision(),
    #         tf.keras.metrics.F1Score(threshold=0.5),
    #     ]

    #     super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # def summary(self):
    #     inputs = tf.keras.Input(shape=self.input_shape)
    #     model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
    #     logger.debug(f"Model summary: {model.summary()}")
    #     logger.debug(f"ClassificationModel summary: {self.backbone.summary()}")

    # def set_backbone_trainable(self, trainable=False):
    #     self.backbone.trainable = trainable
    #     logger.info(f"Backbone trainable: {self.backbone.trainable}")

    # def train(self, stage, train_data_loader, val_data_loader, epochs=10, train_steps_per_epoch=None, val_steps_per_epoch=None, save_path=None, logs_path=None, early_stopping_patience=None):
    #     save_path = os.path.join(save_path, stage)
    #     os.makedirs(save_path, exist_ok=True)  # Add this line
    #     logger.info(f"Save directory exists: {os.path.exists(save_path)}")
    #     checkpoint_path = os.path.join(save_path, "model_{epoch:02d}-{val_loss:.2f}.keras")
    #     logger.info(f"Models will be saved to: {checkpoint_path}")  # Add this line
    #     checkpoint_callback = CustomModelCheckpoint(  # Changed from tf.keras.callbacks.ModelCheckpoint
    #         filepath=checkpoint_path,
    #         save_weights_only=False,
    #         save_best_only=True,
    #         monitor='val_loss',
    #         mode='min',
    #         verbose=1
    #     )
        
    #     # reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    #     #     monitor='val_loss',
    #     #     factor=0.2,
    #     #     patience=3,
    #     #     min_lr=1e-10,
    #     #     verbose=1,
    #     #     mode='min'
    #     # )

    #     log_dir = os.path.join(logs_path, stage)
    #     tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #         log_dir=log_dir,
    #         histogram_freq=1,
    #         write_graph=True,
    #         write_images=True,
    #         update_freq='epoch',
    #         profile_batch=2
    #     )

    #     early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    #         monitor='val_loss',
    #         patience=early_stopping_patience,  # Number of epochs with no improvement after which training will be stopped
    #         verbose=1,
    #         mode='min',
    #         restore_best_weights=True  # Restores the model weights from the epoch with the best value of the monitored quantity
    #     )
        
    #     history = self.fit(
    #         train_data_loader,
    #         steps_per_epoch=train_steps_per_epoch,
    #         epochs=epochs,
    #         validation_data=val_data_loader,
    #         validation_steps=val_steps_per_epoch,
    #         # callbacks=[reduce_lr_callback, tensorboard_callback]
    #         # callbacks=[checkpoint_callback,reduce_lr_callback, tensorboard_callback, early_stopping_callback]
    #         callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback]
    #     )

    #     if stage == "tl":
    #         self.history_tl = history
    #     elif stage == "ft":
    #         self.history_ft = history

    # def visualize_history(self, stage):
    #     if stage == "tl":
    #         history = self.history_tl
    #         f1score_key = 'f1_score'
    #         recall_key = 'recall'
    #         precision_key = 'precision'
    #         val_f1score_key = 'val_f1_score'
    #         val_recall_key = 'val_recall'
    #         val_precision_key = 'val_precision'
    #     elif stage == "ft":
    #         history = self.history_ft
    #         f1score_key = 'f1_score'
    #         recall_key = 'recall_1'
    #         precision_key = 'precision_1'
    #         val_f1score_key = 'val_f1_score'
    #         val_recall_key = 'val_recall_1'
    #         val_precision_key = 'val_precision_1'

    #     # logger.info(f"Visualizing history")
    #     # logger.info(f"History: {self.history.history}")

    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Create a figure with 4 subplots

    #     # Plot training metrics
    #     axes[0, 0].plot(history.history[f1score_key], label='Train F1 Score')
    #     axes[0, 0].plot(history.history[recall_key], label='Train Recall')
    #     axes[0, 0].plot(history.history[precision_key], label='Train Precision')
    #     axes[0, 0].set_title('Training Metrics')
    #     axes[0, 0].legend()

    #     # Plot validation metrics
    #     axes[0, 1].plot(history.history[val_f1score_key], label='Val F1 Score')
    #     axes[0, 1].plot(history.history[val_recall_key], label='Val Recall')
    #     axes[0, 1].plot(history.history[val_precision_key], label='Val Precision')
    #     axes[0, 1].set_title('Validation Metrics')
    #     axes[0, 1].legend()

    #     # Plot training loss
    #     axes[1, 0].plot(history.history['loss'], label='Train Loss')
    #     axes[1, 0].set_title('Training Loss')
    #     axes[1, 0].legend()

    #     # Plot validation loss
    #     axes[1, 1].plot(history.history['val_loss'], label='Val Loss')
    #     axes[1, 1].set_title('Validation Loss')
    #     axes[1, 1].legend()

    #     plt.tight_layout()
    #     plt.show()

    # def run_inference(self, data_loader):
    #     for batch_index, (x, y) in enumerate(data_loader):
    #         if batch_index >= 10:
    #             break
    #         pred = self(x)
    #         for i in range(len(pred)):
    #             plt.imshow(x[i] / 255.0)  # Normalize the image for display
    #             plt.title(f"True: {y[i]}, Predicted: {pred[i].numpy()[0]:.2f}")
    #             plt.axis('off')
    #             plt.show()




