import tensorflow as tf
from tensorflow.keras.applications import VGG19

class BinaryClassificationCNN(tf.keras.Model):
    def __init__(self, input_shape=(768, 768, 3), dropout_rate=0.5):
        super().__init__()
        self.input_shape = input_shape
        self.vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Freeze VGG19 layers (Optional: could unfreeze some top layers for fine-tuning)
        for layer in self.vgg19.layers:
            layer.trainable = False

        # Addding extra conv layers to reduce the number of parameters 
        self.conv_block1 = self._create_vgg19_conv_block()
        self.conv_block2 = self._create_vgg19_conv_block()

        # Adding custom layers
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def _create_vgg19_conv_block(self, num_filters=512):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2))
        ])

    def call(self, inputs, training=False):
        x = self.vgg19(inputs)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        x = self.output_layer(x)
        return x

    def compile_model(self, learning_rate=0.001):
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

    def train(self, train_data, train_labels, validation_data, epochs=10, batch_size=32):
        self.history = self.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data
        )

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