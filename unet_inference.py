"""
U-Net Inference & Evaluation Script
Multi-Class Production Version for Workstation Deployment
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"C:\samcodebase\deeplearning\Tbest_unet_model_v1.keras" #"/home/zumerhub/codebase/u-net-deepl-seg/Trained_Models/unet_satellite_streaming.keras"
PATCH_SIZE = 256
NUM_CLASSES = 6

# ==========================================
# 1. LOAD MODEL
# ==========================================
def load_model(model_path):
    """Load trained U-Net model."""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"📂 Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape:  {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    return model


# ==========================================
# 2. PREPROCESS IMAGE
# ==========================================
def preprocess_image(image_path):
    """Load and normalize image for inference."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img.astype(np.float32) / 255.0
    return img_normalized, img_rgb


# ==========================================
# 3. COLOR PALETTE MAPPER
# ==========================================
def decode_categorical_mask(class_map):
    """
    Maps 2D class integer map (0-5) back to RGB color spaces
    matching the original dataset design parameters.
    """
    h, w = class_map.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Custom config hex-to-RGB color conversions
    rgb_mask[class_map == 0] = [155, 155, 155]  # Unlabeled (#9B9B9B)
    rgb_mask[class_map == 1] = [80, 227, 194]   # Water (#50E3C2)
    rgb_mask[class_map == 2] = [245, 166, 35]   # Land (#F5A623)
    rgb_mask[class_map == 3] = [222, 89, 127]   # Road (#DE597F)
    rgb_mask[class_map == 4] = [208, 2, 27]     # Building (#D0021B)
    rgb_mask[class_map == 5] = [65, 117, 5]     # Vegetation (#417505)
    
    return rgb_mask


# ==========================================
# 4. FULL IMAGE INFERENCE (Multi-Class Tiling)
# ==========================================
def predict_full_image(model, image, patch_size=256, stride=128):
    """
    Predict multi-class probabilities across a full image using a sliding window.
    Maintains all 6 class probability profiles.
    """
    h, w = image.shape[:2]
    
    # Initialize accumulator map with channel dimension intact
    prediction_full = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
    count_map = np.zeros((h, w, 1), dtype=np.float32)
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Predict single patch shape -> (1, 256, 256, 6)
            batch = np.expand_dims(patch, axis=0)
            pred = model.predict(batch, verbose=0)[0]  # Shape: (256, 256, 6)
            
            prediction_full[y:y+patch_size, x:x+patch_size, :] += pred
            count_map[y:y+patch_size, x:x+patch_size, :] += 1.0
            
    # Normalize overlapped window points
    prediction_full /= np.maximum(count_map, 1e-7)
    
    # Reduce across channels to find absolute index winner per pixel
    final_class_map = np.argmax(prediction_full, axis=-1)
    # Extract structural prediction confidence mapping
    confidence_map = np.max(prediction_full, axis=-1)
    
    return final_class_map, confidence_map


# ==========================================
# 5. VISUALIZE RESULTS (MULTI-CLASS CONFIGURATED)
# ==========================================
def visualize_prediction(image_rgb, class_map, confidence_map, save_path=None):
    """Plots a 4-pane layout containing categorical land designations."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # Panel 1: Source Photo
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Panel 2: Model Certainty Density
    im1 = axes[1].imshow(confidence_map, cmap='jet', vmin=0.0, vmax=1.0)
    axes[1].set_title('Prediction Confidence')
    axes[1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Softmax Prob')
    
    # Panel 3: Colorful Decoded Class Mask
    predicted_rgb_mask = decode_categorical_mask(class_map)
    axes[2].imshow(predicted_rgb_mask)
    axes[2].set_title('Predicted Classes (6 Layers)')
    axes[2].axis('off')
    
    # Panel 4: Alpha-Blended Prediction Overlay
    overlay = cv2.addWeighted(image_rgb, 0.6, predicted_rgb_mask, 0.5, 0)
    axes[3].imshow(overlay)
    axes[3].set_title('Original + Class Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved multi-class visualization to: {save_path}")
    
    plt.show()


# ==========================================
# 6. RUN DEMO PIPELINE
# ==========================================
if __name__ == "__main__":
    print("="*70)
    print("MULTI-CLASS U-NET INFERENCE ENGINE")
    print("="*70)
    
    model = load_model(MODEL_PATH)
    if model is None:
        exit(1)
        
    demo_image_path = r"C:\samcodebase\deeplearning\semantic-segmentation\Semantic_segmentation_dataset\Tile 5\images\image_part_001.jpg" #"/home/zumerhub/codebase/u-net-deepl-seg/semantic-segmentation/Semantic_segmentation_datasets/Tile 5/images/image_part_008.jpg"
    
    if os.path.exists(demo_image_path):
        print(f"\n📷 Target verified! Loading image: {demo_image_path}")
        img, img_rgb = preprocess_image(demo_image_path)
        
        if img is not None:
            print("🔮 Running sliding-window multi-class inference...")
            class_map, confidence_map = predict_full_image(model, img)
            
            # Save and display full color-coded matrix charts
            visualize_prediction(
                img_rgb, 
                class_map, 
                confidence_map,
                save_path='prediction_example.png'
            )
            
            # Print accurate layer stats
            print(f"\n📊 Target Layout Stats:")
            print(f"   Mean prediction confidence: {confidence_map.mean():.4f}")
            for c in range(NUM_CLASSES):
                pct = np.sum(class_map == c) / class_map.size * 100
                print(f"   Class {c} distribution footprint: {pct:.2f}%")
    else:
        print(f"❌ Target missing file link: {demo_image_path}")

    
    # what is the visible are to work with at thr moment///////////////////
















# """
# U-Net Inference & Evaluation Script
# Load trained model and run predictions on images
# """

# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# # ==========================================
# # CONFIGURATION
# # ==========================================
# MODEL_PATH = '/home/zumerhub/codebase/u-net-deepl-seg/Trained_Models/unet_satellite_streaming.keras'
# PATCH_SIZE = 256



# CLASS_COLORS = np.array([
#     [0, 0, 0],        # Class 0
#     [255, 0, 0],      # Class 1
#     [0, 255, 0],      # Class 2
#     [0, 0, 255],      # Class 3
#     [255, 255, 0],    # Class 4
#     [255, 0, 255],    # Class 5
# ], dtype=np.uint8)

# def colorize_mask(mask):
#     return CLASS_COLORS[mask]

# # ==========================================
# # 1. LOAD MODEL
# # ==========================================
# def load_model(model_path):
#     """Load trained U-Net model."""
#     if not os.path.exists(model_path):
#         print(f"❌ Model not found: {model_path}")
#         return None
    
#     print(f"📂 Loading model from {model_path}...")
#     model = keras.models.load_model(model_path)
#     print(f"✅ Model loaded successfully!")
#     print(f"   Input shape:  {model.input_shape}")
#     print(f"   Output shape: {model.output_shape}")
    
#     return model


# # ==========================================
# # 2. PREPROCESS IMAGE
# # ==========================================
# def preprocess_image(image_path):
#     """Load and normalize image for inference."""
#     img = cv2.imread(image_path)
    
#     if img is None:
#         print(f"❌ Failed to load image: {image_path}")
#         return None
    
#     # Convert BGR to RGB for display
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Normalize for model input
#     img_normalized = img.astype(np.float32) / 255.0
    
#     return img_normalized, img_rgb


# # ==========================================
# # 3. PREDICT ON PATCH
# # ==========================================
# def predict_patch(model, image_patch):
#     batch = np.expand_dims(image_patch, axis=0)

#     pred = model.predict(batch, verbose=0)

#     # Convert probabilities → class index
#     pred_class = np.argmax(pred[0], axis=-1)

#     return pred_class.astype(np.uint8)


# # ==========================================
# # 4. FULL IMAGE INFERENCE (Tiling)
# # ==========================================
# def predict_full_image(model, image, patch_size=256, stride=128):
#     """
#     Predict on full image by sliding window.
#     Combines predictions from overlapping patches.
#     """
#     h, w = image.shape[:2]
    
#     # Initialize output accumulator
#     prediction_full = np.zeros((h, w), dtype=np.float32)
#     count_map = np.zeros((h, w), dtype=np.float32)
    
#     # Slide window
#     # for y in range(0, h - patch_size + 1, stride):
#     #     for x in range(0, w - patch_size + 1, stride):
#     #         # Extract patch
#     #         patch = image[y:y+patch_size, x:x+patch_size]
            
#     #         # Predict
#     #         pred = predict_patch(model, patch)
            
#     #         # Accumulate prediction
#     #         prediction_full[y:y+patch_size, x:x+patch_size] += pred
#     #         count_map[y:y+patch_size, x:x+patch_size] += 1

#     batch = np.expand_dims(patch, axis=0)

#     pred = model.predict(batch, verbose=0)[0]

#     prediction_full[y:y+patch_size, x:x+patch_size] += pred

#     count_map[y:y+patch_size, x:x+patch_size] += 1

    
#     # Average overlapping regions
#     # mask = count_map > 0
#     # prediction_full[mask] /= count_map[mask]
    
#     # return prediction_full
#     mask = count_map > 0

#     for c in range(6):
#         prediction_full[:, :, c][mask] /= count_map[mask]

#     final_prediction = np.argmax(prediction_full, axis=-1)

#     return final_prediction

# # ==========================================
# # 5. POSTPROCESS MASK
# # ==========================================
# def postprocess_mask(prediction, threshold=0.5):
#     """
#     Convert soft prediction to binary mask.
#     """
#     binary_mask = (prediction >= threshold).astype(np.uint8) * 255
#     return binary_mask


# # ==========================================
# # 6. VISUALIZE RESULTS
# # ==========================================
# # def visualize_prediction(image_rgb, prediction, mask_binary, title="Prediction", save_path=None):
# #     """
# #     Display original image, soft prediction, and binary mask.
# #     """
# #     fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
# #     # Original image
# #     axes[0].imshow(image_rgb)
# #     axes[0].set_title('Original Image')
# #     axes[0].axis('off')
    
# #     # Soft prediction
# #     axes[1].imshow(prediction, cmap='hot')
# #     axes[1].set_title('Soft Prediction (0-1)')
# #     axes[1].axis('off')
# #     cbar1 = plt.colorbar(axes[1].images[0], ax=axes[1])
# #     cbar1.set_label('Confidence')
    
# #     # Binary mask
# #     axes[2].imshow(mask_binary, cmap='gray')
# #     axes[2].set_title('Binary Mask (threshold=0.5)')
# #     axes[2].axis('off')
    
# #     # Overlay: image + mask
# #     img_with_mask = image_rgb.copy()
# #     mask_rgb = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2RGB)
# #     img_with_mask[mask_rgb > 0] = (
# #         0.5 * img_with_mask[mask_rgb > 0] + 
# #         0.5 * np.array([0, 255, 0])  # Green overlay
# #     ).astype(np.uint8)
    
# #     axes[3].imshow(img_with_mask)
# #     axes[3].set_title('Original + Predicted Mask')
# #     axes[3].axis('off')
    
# #     plt.tight_layout()
    
# #     if save_path:
# #         plt.savefig(save_path, dpi=150, bbox_inches='tight')
# #         print(f"💾 Saved visualization to: {save_path}")
    
# #     plt.show()

# def visualize_prediction(image_rgb, prediction):
    
#     colored_mask = colorize_mask(prediction)

#     overlay = cv2.addWeighted(
#         image_rgb,
#         0.6,
#         colored_mask,
#         0.4,
#         0
#     )

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#     axes[0].imshow(image_rgb)
#     axes[0].set_title("Original Image")
#     axes[0].axis("off")

#     axes[1].imshow(colored_mask)
#     axes[1].set_title("Multiclass Segmentation")
#     axes[1].axis("off")

#     axes[2].imshow(overlay)
#     axes[2].set_title("Overlay")
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.show()

    
# # ==========================================
# # 7. EVALUATE ON DATASET
# # ==========================================
# def evaluate_on_dataset(model, image_paths, mask_paths):
#     """
#     Evaluate model on multiple images.
#     Compute IoU, Dice coefficient, etc.
#     """
#     print("\n" + "="*70)
#     print("EVALUATION ON DATASET")
#     print("="*70)
    
#     ious = []
#     dices = []
    
#     for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
#         # Load and predict
#         img, img_rgb = preprocess_image(img_path)
#         ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
#         if img is None:
#             continue
        
#         # Predict
#         prediction = predict_full_image(model, img)
        
#         # Binarize
#         pred_binary = (prediction >= 0.5).astype(np.float32)
#         gt_binary = (ground_truth >= 0.5).astype(np.float32)
        
#         # Calculate IoU (Intersection over Union)
#         intersection = np.sum(pred_binary * gt_binary)
#         union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
#         iou = intersection / (union + 1e-7)
        
#         # Calculate Dice coefficient
#         dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary) + 1e-7)
        
#         ious.append(iou)
#         dices.append(dice)
        
#         print(f"  Image {idx+1}: IoU={iou:.4f} | Dice={dice:.4f}")
    
#     # Summary
#     if ious:
#         mean_iou = np.mean(ious)
#         mean_dice = np.mean(dices)
#         std_iou = np.std(ious)
#         std_dice = np.std(dices)
        
#         print(f"\n📊 Summary:")
#         print(f"   Mean IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
#         print(f"   Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
#         print("="*70)
        
#         return {
#             'ious': ious,
#             'dices': dices,
#             'mean_iou': mean_iou,
#             'mean_dice': mean_dice
#         }
    
#     return None


# # ==========================================
# # 8. BATCH PREDICTION
# # ==========================================
# def predict_directory(model, image_dir, output_dir=None):
#     """
#     Predict on all images in a directory.
#     Save predictions to output directory.
#     """
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     image_files = [f for f in os.listdir(image_dir) 
#                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     print(f"\n📁 Processing {len(image_files)} images from {image_dir}")
    
#     for idx, filename in enumerate(image_files):
#         image_path = os.path.join(image_dir, filename)
        
#         # Preprocess
#         img, img_rgb = preprocess_image(image_path)
#         if img is None:
#             continue
        
#         # Predict
#         prediction = predict_full_image(model, img)
#         # mask_binary = postprocess_mask(prediction)
        
#         # Save
#         if output_dir:
#             output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.png")
#             cv2.imwrite(output_path, mask_binary)
        
#         print(f"  [{idx+1}/{len(image_files)}] {filename}")
    
#     print(f"✅ Predictions saved to {output_dir}")


# # ==========================================
# # 9. MAIN DEMO
# # ==========================================
# if __name__ == "__main__":
#     print("="*70)
#     print("U-NET INFERENCE & EVALUATION")
#     print("="*70)
    
#     # Load model
#     model = load_model(MODEL_PATH)
    
#     if model is None:
#         print("\n⚠️  Please train the model first:")
#         print("   python unet_streaming_nomem.py")
#         exit(1)
    
#     # DEMO 1: Single image prediction
#     print("\n" + "="*70)
#     print("DEMO 1: SINGLE IMAGE PREDICTION")
#     print("="*70)
    
#     # You can modify this path to an actual image
#     demo_image_path = "/home/zumerhub/codebase/u-net-deepl-seg/semantic-segmentation/Semantic_segmentation_datasets/Tile 8/images/image_part_003.jpg" #"/content/drive/MyDrive/Colab Notebooks/semantic-segmentation/Semantic_segmentation_dataset/Tile_1/images/image_1.jpg"
    
#     if os.path.exists(demo_image_path):
#         print(f"\n📷 Loading image: {demo_image_path}")
#         img, img_rgb = preprocess_image(demo_image_path)
        
#         if img is not None:
#             # Predict
#             print("🔮 Running prediction...")
#             prediction = predict_full_image(model, img)
            
#             # Postprocess
#             mask_binary = postprocess_mask(prediction, threshold=0.5)
            
#             # Visualize
#             visualize_prediction(
#                 img_rgb, 
#                 prediction, 
#                 mask_binary,
#                 save_path='prediction_example.png'
#             )
            
#             # Print stats
#             coverage = np.sum(mask_binary > 0) / mask_binary.size * 100

#             unique, counts = np.unique(prediction, return_counts=True)

#             print("\nClass Distribution")

#             for cls, cnt in zip(unique, counts):
#                 pct = cnt / prediction.size * 100
#                 print(f"Class {cls}: {pct:.2f}%")

#             print(f"\n📊 Prediction Statistics:")
#             print(f"   Mask coverage: {coverage:.2f}%")
#             print(f"   Mean prediction: {prediction.mean():.4f}")
#             print(f"   Max prediction: {prediction.max():.4f}")
#             print(f"   Min prediction: {prediction.min():.4f}")
#     else:
#         print(f"⚠️  Demo image not found: {demo_image_path}")
#         print("   Update demo_image_path to test on actual images")
    
#     # DEMO 2: Batch prediction
#     print("\n" + "="*70)
#     print("DEMO 2: BATCH PREDICTION (Optional)")
#     print("="*70)
#     print("Uncomment the code below to run batch predictions on a directory")
#     print("# predict_directory(model, '/path/to/images', '/path/to/output')")
    
#     print("\n✅ Inference script ready!")
#     print("\n📝 Usage Examples:")
#     print("   1. Single patch: pred = predict_patch(model, image_patch)")
#     print("   2. Full image:   pred = predict_full_image(model, image)")
#     print("   3. Batch dir:    predict_directory(model, '/path/images', '/path/output')")
#     print("   4. Evaluate:     evaluate_on_dataset(model, img_list, mask_list)")

