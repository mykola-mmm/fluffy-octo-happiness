import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_binary_crossentropy(weight_zero=0.5, weight_one=0.5):
    """
    Weighted Binary Crossentropy loss function.
    
    Args:
    weight_zero (float): Weight for class 0 (default: 0.5)
    weight_one (float): Weight for class 1 (default: 0.5)
    
    Returns:
    function: Weighted Binary Crossentropy loss function
    """
    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calculate binary crossentropy
        bce = y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred)
        
        # Apply weights
        weights = y_true * weight_one + (1 - y_true) * weight_zero
        
        # Calculate weighted average
        weighted_bce = weights * bce
        
        return -K.mean(weighted_bce, axis=-1)
    
    return loss
