import logging
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras import mixed_precision

logger = logging.getLogger(__name__)

class EfficientNet(tf.keras.Model):
    def __init__(self, input_shape=(768, 768, 3), dropout_rate=0.1):
        super().__init__()
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.build()
        
    def build(self):
        # Pretrained EfficientNetV2L model
        self.efficientnet = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=self.input_shape, pooling='max')
        # # Binary classification head
        # self.flatten = tf.keras.layers.Flatten()
        # self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        # self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        # self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        # self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        # self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)
        
        # Binary classification head
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)

    def call(self, inputs, training=False):
        x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
        x = self.efficientnet(x, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.output_layer(x)
        return x

#     def compile_model(self, learning_rate=0.001, weight_zero=0.5, weight_one=0.5):
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
#         optimizer = mixed_precision.LossScaleOptimizer(optimizer)

#         # metrics = ['accuracy', 'recall', 'f1_score']
#         # metrics = ['accuracy']
#         metrics = [
#             tf.keras.metrics.BinaryAccuracy(name='binary_accuracy_0.5', threshold=0.5),
#             tf.keras.metrics.Recall(name='recall'),
#         ]


#         self.compile(optimizer=optimizer,
#                     #  loss=weighted_binary_crossentropy(weight_zero, weight_one),
#                      loss = tf.keras.losses.BinaryCrossentropy(),
#                      metrics=metrics)

#     def train(self, data_loader, validation_data, epochs=10, train_df_len=None, validation_df_len=None, batch_size=32, save_path=None, weight_zero=0.5, weight_one=0.5):
#         # Create callbacks
#         checkpoint_path = os.path.join(save_path, "checkpoints/model_{epoch:02d}-{val_loss:.2f}.weights.h5")
#         checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#             filepath=checkpoint_path,
#             save_weights_only=True,
#             save_best_only=True,
#             monitor='val_recall',
#             mode='max',
#             verbose=1
#         )

#         reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_recall',
#             factor=0.2,
#             patience=5,
#             min_lr=1e-6,
#             verbose=1
#         )

#         tensorboard_callback = tf.keras.callbacks.TensorBoard(
#             log_dir='./logs',
#             histogram_freq=1,
#             write_graph=True,
#             write_images=True,
#             update_freq='epoch',
#             profile_batch=2
#         )

#         train_steps_per_epoch = train_df_len // batch_size
#         val_steps_per_epoch = validation_df_len // batch_size

#         # Train the model
#         self.history = self.fit(
#             data_loader,
#             epochs=epochs,
#             steps_per_epoch=train_steps_per_epoch,
#             validation_data=validation_data,
#             validation_steps=val_steps_per_epoch,
#             callbacks=[reduce_lr_callback, tensorboard_callback],
#             # callbacks=[checkpoint_callback, reduce_lr_callback, tensorboard_callback],
#             class_weight={0: weight_zero, 1: weight_one}
#         )

#         # Save the best model
#         if save_path:
#             os.makedirs(save_path, exist_ok=True)
#             os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)

#         best_model_path = os.path.join(save_path, "best_model.weights.h5")
#         self.save_weights(best_model_path)
#         self.save(os.path.join(save_path, "best_model.keras"))
#         print(f"Best model saved to {best_model_path}")

    def summary(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        model = tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
        logger.info(f"Model summary: {model.summary()}")
        logger.info(f"EfficientNet summary: {self.efficientnet.summary()}")


#     def summary_vgg(self):
#         self.vgg19.summary()
