import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        # Flatten the tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice Loss
        return 1 - dice

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        dice = self.dice_loss(y_true, y_pred)
        bce = self.bce_loss(y_true, y_pred)
        return self.bce_weight * bce + self.dice_weight * dice


