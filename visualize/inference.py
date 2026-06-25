import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# 1. Bring in metrics/losses definitions for Keras loader
def jaccard_coef(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    return (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def total_loss(y_true, y_pred):
    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy()
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)


# 2. Set path to your new native .keras file
# (Adjust the folder path if the file saved into a specific directory)
model_path = r"best_unet_satellite_model_2026-06-19_08-46-40.keras"

model = load_model(model_path, custom_objects={
    "dice_loss": dice_loss,
    "total_loss": total_loss,
    "jaccard_coef": jaccard_coef
})

# Compile immediately after loading to silence Keras tracking warnings
model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy', jaccard_coef])


# 3. Pull a test image and expand its dimensions to a 4D batch array
from satellite_data import x_test, y_test
test_img = x_test[0] 
ground_truth = y_test[0]

input_batch = np.expand_dims(test_img, axis=0)


# 4. Predict and process probabilities to categorical map IDs
prediction = model.predict(input_batch)
predicted_mask = np.argmax(prediction[0], axis=-1)
true_mask = np.argmax(ground_truth, axis=-1)


# 5. Plot the result side-by-side
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(test_img[:,:,0] if test_img.shape[-1] == 1 else test_img, cmap='gray' if test_img.shape[-1] == 1 else None)
plt.title("Satellite Image Input")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(true_mask, cmap='jet')
plt.title("Ground Truth Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask, cmap='jet')
plt.title("U-Net Prediction Mask")
plt.axis('off')

plt.tight_layout()
plt.show()