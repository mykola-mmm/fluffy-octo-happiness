import os
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import mixed_precision
from src.loss import weighted_binary_crossentropy

class BinaryClassificationCNN(tf.keras.Model):
    def __init__(self, input_shape=(768, 768, 3), dropout_rate=0.1):
        super().__init__()
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.build()
        
    def build(self):
        # self.vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape, pooling='max')
        self.vgg19 = VGG19(include_top=False, input_shape=self.input_shape)

        # Addding extra conv layers to reduce the number of parameters 
        self.conv_block1 = self._create_vgg19_conv_block()
        self.conv_block2 = self._create_vgg19_conv_block()

        # Adding custom layers
        self.avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pooling = tf.keras.layers.GlobalMaxPooling2D()

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)
        
        super().build(self.input_shape)

    def _create_vgg19_conv_block(self, num_filters=512):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2))
        ])
    
    def set_vgg19_trainable(self, trainable=False):
        """
        Set the trainable status of the VGG19 layers.
        
        Args:
            trainable (bool): If True, make VGG19 layers trainable. If False, make them non-trainable.
        """
        for layer in self.vgg19.layers:
            layer.trainable = trainable
        
        status = "trainable" if trainable else "non-trainable"
        print(f"VGG19 layers set to {status}.")

    def call(self, inputs, training=False):
        x = self.vgg19(inputs)
        # x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = self.flatten(x)
        x = self.max_pooling(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        x = self.output_layer(x)
        return x

    def compile_model(self, learning_rate=0.001, weight_zero=0.5, weight_one=0.5):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        # metrics = ['accuracy', 'recall', 'f1_score']
        # metrics = ['accuracy']
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy_0.5', threshold=0.5),
            tf.keras.metrics.Recall(name='recall'),
        ]


        self.compile(optimizer=optimizer,
                    #  loss=weighted_binary_crossentropy(weight_zero, weight_one),
                     loss = tf.keras.losses.BinaryCrossentropy(),
                     metrics=metrics)

    def train(self, data_loader, validation_data, epochs=10, train_df_len=None, validation_df_len=None, batch_size=32, save_path=None, weight_zero=0.5, weight_one=0.5):
        # Create callbacks
        checkpoint_path = os.path.join(save_path, "checkpoints/model_{epoch:02d}-{val_loss:.2f}.weights.h5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_recall',
            mode='max',
            verbose=1
        )

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_recall',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2
        )

        train_steps_per_epoch = train_df_len // batch_size
        val_steps_per_epoch = validation_df_len // batch_size

        # Train the model
        self.history = self.fit(
            data_loader,
            epochs=epochs,
            steps_per_epoch=train_steps_per_epoch,
            validation_data=validation_data,
            validation_steps=val_steps_per_epoch,
            callbacks=[reduce_lr_callback, tensorboard_callback],
            # callbacks=[checkpoint_callback, reduce_lr_callback, tensorboard_callback],
            class_weight={0: weight_zero, 1: weight_one}
        )

        # Save the best model
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)

        best_model_path = os.path.join(save_path, "best_model.weights.h5")
        self.save_weights(best_model_path)
        self.save(os.path.join(save_path, "best_model.keras"))
        print(f"Best model saved to {best_model_path}")

    def summary(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        model.summary()


    def summary_vgg(self):
        self.vgg19.summary()


if __name__ == "__main__":
    model = BinaryClassificationCNN()
    model.summary()
    model.summary_vgg()
