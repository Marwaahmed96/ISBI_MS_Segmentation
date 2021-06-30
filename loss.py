import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coefficient_loss(y_true, y_pred,eps=1e-9):
    intersect = 2*(K.sum(y_true * y_pred))
    union = K.sum(y_pred) + K.sum(y_true) 
    dice_score=(intersect + eps) / (union+eps)
    return 1.0 - dice_score

def BCEDiceLoss(y_true, y_pred,eps=1e-9):
    intersect = 2*(K.sum(y_true * y_pred))
    union = K.sum(y_pred) + K.sum(y_true) 
    dice_score=(intersect + eps) / (union+eps)
    dice_loss=1.0 - dice_score
    bce = tf.keras.losses.BinaryCrossentropy()
    BCE =  bce(y_true, y_pred)
    return dice_loss + BCE