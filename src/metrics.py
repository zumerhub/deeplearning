import tensorflow as tf
from tensorflow.keras import backend as K

def jaccard_coef(y_true, y_pred):
    """
    Computes the Jaccard Coefficient (Intersection over Union / IoU).
    Optimized via graph-safe tensor flattening.
    """
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    
    # Smooth with 1.0 to handle edge cases where division by zero could occur
    return (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice Loss, which optimizes directly for regional overlap 
    and handles class imbalance effectively in semantic segmentation.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    dice_coef = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1.0 - dice_coef


def total_loss(y_true, y_pred):
    """
    Combined Loss Architecture: Dice Loss + Categorical Focal Crossentropy.
    - Dice Loss focuses on structural global feature overlap.
    - Focal Crossentropy forces the U-Net to learn from rare, hard-to-classify pixels 
      (like narrow roads or small buildings) by down-weighting easy background pixels.
    """
    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy()
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

# run
# python predict.py --input "C:/samcodebase/deeplearning/Raw_Image/sentinel2_downloaded.tif" --batch_size 64