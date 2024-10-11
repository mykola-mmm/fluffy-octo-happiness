from tensorflow.keras.applications import VGG19, ResNet50
import tensorflow as tf

class ClassifierModel(tf.keras.Model):
    def __init__(self, input_shape=(768, 768, 3)):
        super(ClassifierModel, self).__init__()
        self.vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1000, activation='softmax')
        

    def summary(self):
        return self.vgg19.summary()
