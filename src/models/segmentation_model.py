import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.callbacks import CustomModelCheckpoint
# from src.utils.warmup_decay_schedule import WarmupDecaySchedule
from src.utils.dice_loss import DiceLoss
from src.utils.dice_loss import CombinedLoss
from src.utils.iou import IoU

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
        
        # Create an encoder model that outputs the required intermediate layers
        outputs = [
            self.backbone.get_layer('block1_conv2').output,
            self.backbone.get_layer('block2_conv2').output,
            self.backbone.get_layer('block3_conv3').output,
            self.backbone.get_layer('block4_conv3').output,
            self.backbone.get_layer('block5_conv3').output,
        ]
        self.encoder = tf.keras.Model(inputs=self.backbone.input, outputs=outputs)
        
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
        # Preprocess inputs
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
        
        # Get all intermediate outputs from the encoder at once
        block1_out, block2_out, block3_out, block4_out, encoder_out = self.encoder(x)
        
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

    def compile_model(self, steps_per_epoch=None, learning_rate=0.0001, decay_rate=0.9):
        initial_learning_rate = learning_rate
        decay_steps = steps_per_epoch
        decay_rate = decay_rate
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0)

        # Create loss functions
        # bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # dice_loss = DiceLoss(smooth=1e-6, name='dice_loss')
        iou = IoU(threshold=0.5, name='iou')
        comb_loss = CombinedLoss(dice_weight=1.0, bce_weight=1.0, name='combined_loss')
        # loss = CombinedLoss(dice_weight=1.0, bce_weight=1.0, name='combined_loss')
        loss = DiceLoss(smooth=1e-6, name='dice_loss')

        # # Combined loss function
        # def combined_loss(y_true, y_pred):
        #     return bce_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

        # metrics = [
        #     iou,
        #     dice_loss,
        #     combined_loss
        # ]
        metrics = [iou, comb_loss]
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def set_backbone_trainable(self, trainable=False):
        self.backbone.trainable = trainable
        logger.info(f"Backbone trainable: {self.backbone.trainable}")

    def train(self, train_data_loader, val_data_loader, epochs=10, train_steps_per_epoch=None, val_steps_per_epoch=None, save_path=None, logs_path=None, early_stopping_patience=None):
        save_path = os.path.join(save_path)
        os.makedirs(save_path, exist_ok=True)  # Add this line
        checkpoint_model_path = os.path.join(save_path, "model_{epoch:02d}-{val_loss:.2f}.keras")
        logger.info(f"BEST Model will be saved to: {checkpoint_model_path}")  # Add this line
        checkpoint_model_callback = CustomModelCheckpoint(  # Changed from tf.keras.callbacks.ModelCheckpoint
            filepath=checkpoint_model_path,
            save_weights_only=False,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        checkpoint_weights_path = os.path.join(save_path, "model_{epoch:02d}-{val_loss:.2f}.weights.h5")
        checkpoint_weights_callback = CustomModelCheckpoint(  # Changed from tf.keras.callbacks.ModelCheckpoint
            filepath=checkpoint_weights_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
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
        
        self.history = self.fit(
            train_data_loader,
            steps_per_epoch=train_steps_per_epoch,
            epochs=epochs,
            validation_data=val_data_loader,
            validation_steps=val_steps_per_epoch,
            # callbacks=[reduce_lr_callback, tensorboard_callback]
            # callbacks=[checkpoint_callback,reduce_lr_callback, tensorboard_callback, early_stopping_callback]
            # callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback]
            # callbacks=[tensorboard_callback, early_stopping_callback]
            callbacks=[checkpoint_model_callback, checkpoint_weights_callback, tensorboard_callback, early_stopping_callback]
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'dropout_rate': self.dropout_rate,
            'pretrained': self.pretrained,
            'l1': self.l1,
            'l2': self.l2,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            input_shape=config['input_shape'],
            dropout_rate=config['dropout_rate'],
            pretrained=config['pretrained'],
            l1=config['l1'],
            l2=config['l2']
        )
