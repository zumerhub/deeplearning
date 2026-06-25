import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# 1. Redefine custom metrics/losses so Keras can load the model safely
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


# 2. Load the trained model
# model_path = r"C:\samcodebase\deeplearning\Trained_Models\unet_satellite_streaming.keras" # 
model_path = "best_unet_model_v1.h5" # "Old_best_unet_model_v1.h5" #"Tbest_unet_model_v1.h5"
model = load_model(model_path, custom_objects={
    "dice_loss": dice_loss,
    "total_loss": total_loss,
    "jaccard_coef": jaccard_coef
})


# 3. Prepare your input data
# Replace this with your actual test image loading logic (e.g., cv2.imread or tifffile)
# For demonstration, we'll grab a random image from your existing x_test array
from satellite_data import x_test, y_test  
test_img = x_test[1]       # Shape: (256, 256, image_channels)
ground_truth = y_test[1]   # Shape: (256, 256, total_classes)

# CRITICAL: Models expect a 4D batch dimension -> (1, height, width, channels)
input_img = np.expand_dims(test_img, axis=0) 


# 4. Run Inference
prediction = model.predict(input_img) # Shape output: (1, 256, 256, total_classes)


# 5. Post-Process the Results
# prediction[0] strips the batch dimension -> (256, 256, total_classes)
# np.argmax converts probabilities into class IDs (e.g., 0, 1, 2, 3, 4)
predicted_mask = np.argmax(prediction[0], axis=-1) 

# Convert ground truth from One-Hot encoded back to 2D integer mask for plotting
true_mask = np.argmax(ground_truth, axis=-1)


# 6. Visualize the Results
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
# If grayscale (1 channel), use cmap='gray'. If RGB, remove cmap.
if test_img.shape[-1] == 1:
    plt.imshow(test_img[:,:,0], cmap='gray')
else:
    plt.imshow(test_img)
plt.title('Testing Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(true_mask, cmap='jet') # 'jet' or 'categorical' colormaps work well for classes
plt.title('True Mask (Ground Truth)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask, cmap='jet')
plt.title('Predicted Mask')
plt.axis('off')

plt.tight_layout()
plt.show()










# # run_inference.py

# from tensorflow.keras.models import load_model
# from satellite_data import image_height, image_width, x_test, y_test
# from training import dice_loss, total_loss, jaccard_coef
# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt

# # Paths
# model_path = r"C:\samcodebase\deeplearning\best_unet_model_v1.h5"
# test_img_path = r"C:\samcodebase\deeplearning\semantic-segmentation\Semantic_segmentation_dataset\Tile 1\images\image_part_005.jpg"

# # Load trained model (no training here)
# model = load_model(model_path,
#                    custom_objects={"dice_loss": dice_loss,
#                                    "total_loss": total_loss,
#                                    "jaccard_coef": jaccard_coef})

# # --- Option A: use dataset sample ---
# idx = 0
# test_img = x_test[idx]
# true_mask = y_test[idx]

# # --- Option B: use standalone image file ---
# # Uncomment if you want to test with a raw image file
# # test_img = load_img(test_img_path, target_size=(image_height, image_width))
# # test_img = img_to_array(test_img) / 255.0
# # true_mask = None   # no ground truth available for external image

# # Predict
# pred_mask = model.predict(np.expand_dims(test_img, axis=0))
# pred_mask = np.argmax(pred_mask, axis=-1)[0]

# # Visualize
# plt.figure(figsize=(12,4))
# plt.subplot(1,3,1); plt.title("Input"); plt.imshow(test_img.squeeze(), cmap="gray"); plt.axis("off")

# if true_mask is not None:
#     plt.subplot(1,3,2); plt.title("Ground Truth"); plt.imshow(np.argmax(true_mask, axis=-1), cmap="jet"); plt.axis("off")

# plt.subplot(1,3,3); plt.title("Prediction"); plt.imshow(pred_mask, cmap="jet"); plt.axis("off")
# plt.show()
