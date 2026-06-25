# ================== updated predict ==================

"""
predict.py - Production Seamless Overlap GeoTIFF Inference Engine
Location: C:\\samcodebase\\deeplearning\\predict.py
Description: Ingests un-seen satellite images, applies 2D Gaussian sliding-window 
             patching with multi-class matrix scaling, and exports structural GIS maps.
"""

import os
import argparse
import numpy as np
import rasterio
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.data_utils import load_config, get_vectorized_mappings
from src.metrics import total_loss, jaccard_coef

def parse_args():
    parser = argparse.ArgumentParser(description="Production Seamless Overlap GeoTIFF Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input satellite GeoTIFF raster (.tif)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration yaml")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of patches passed concurrently to GPU")
    return parser.parse_args()

def generate_gaussian_weights(patch_size):
    """Generates a 2D Gaussian window to smoothly blend overlapping edges."""
    w = cv2.getGaussianKernel(patch_size, patch_size / 3)
    w_2d = np.outer(w, w)
    return w_2d / w_2d.max()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    weights_path = config["model"]["weights_path"]
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights missing at: {weights_path}")
        
    print(f"🏗️ Initializing model graphs and loading custom loss states...")
    
    # Senior Dev MLOps Fix: Explicitly pass the structural Keras class object definition to custom_objects
    model = load_model(
        weights_path, 
        custom_objects={
            "total_loss": total_loss, 
            "jaccard_coef": jaccard_coef,
            "CategoricalFocalCrossentropy": tf.keras.losses.CategoricalFocalCrossentropy
        }
    )
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Target image asset missing at: {args.input}")
        
    print(f"Instantiating overlapping streaming grids from source GeoTIFF...")
    with rasterio.open(args.input) as src:
        geo_profile = src.profile.copy()
        target_bands = config["model"].get("raster_bands", [1, 2, 3])
        large_img_rasterio = src.read(target_bands)
        large_img = np.transpose(large_img_rasterio, (1, 2, 0))
        h, w, c = large_img.shape
        print(f"✅ Loaded native bounds canvas: {w}x{h} ({c} channels)")

    patch_size = config["model"]["patch_size"]
    num_classes = config["model"].get("num_classes", 6)
    
    # 🎯 STEP OVERLAP: Move by half a patch size to blend boundaries smoothly
    stride = patch_size // 2  
    blend_window = generate_gaussian_weights(patch_size)
    
    # Accumulation canvases for probability mapping lists
    prob_canvas = np.zeros((h, w, num_classes), dtype=np.float32)
    weight_canvas = np.zeros((h, w), dtype=np.float32)
    
    # Generate coordinates across the image dimensions using a strict stride
    y_coords = list(range(0, h - patch_size + 1, stride))
    if y_coords[-1] != h - patch_size: y_coords.append(h - patch_size)
    x_coords = list(range(0, w - patch_size + 1, stride))
    if x_coords[-1] != w - patch_size: x_coords.append(w - patch_size)
    
    batch_patches = []
    batch_meta = []
    
    print(f"⚡ Processing overlapping spatial grid array passes...")
    for y in y_coords:
        for x in x_coords:
            y_end, x_end = y + patch_size, x + patch_size
            patch = large_img[y:y_end, x:x_end]
            
            normalized_patch = patch.astype("float32") / 255.0
            batch_patches.append(normalized_patch)
            batch_meta.append((y, y_end, x, x_end))
            
            if len(batch_patches) == args.batch_size:
                predictions = model.predict(np.array(batch_patches), batch_size=args.batch_size, verbose=0)
                for i in range(len(predictions)):
                    by, by_end, bx, bx_end = batch_meta[i]
                    # Accumulate model probabilities multiplied by Gaussian blending weights
                    prob_canvas[by:by_end, bx:bx_end] += predictions[i] * blend_window[..., np.newaxis]
                    weight_canvas[by:by_end, bx:bx_end] += blend_window
                batch_patches = []
                batch_meta = []
                
    if batch_patches:
        predictions = model.predict(np.array(batch_patches), batch_size=len(batch_patches), verbose=0)
        for i in range(len(predictions)):
            by, by_end, bx, bx_end = batch_meta[i]
            prob_canvas[by:by_end, bx:bx_end] += predictions[i] * blend_window[..., np.newaxis]
            weight_canvas[by:by_end, bx:bx_end] += blend_window

    # Normalize across accumulated layers to protect scaling limits
    print(f"🎨 Normalizing overlap weight channels and stitching...")
    weight_canvas[weight_canvas == 0] = 1.0
    prob_canvas /= weight_canvas[..., np.newaxis]
    stitched_mask = np.argmax(prob_canvas, axis=-1).astype(np.uint8)

    # Apply color mappings
    target_colors, _ = get_vectorized_mappings(config)
    colored_output = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, rgb_val in target_colors.items():
        colored_output[stitched_mask == class_idx] = rgb_val
        
    output_dir = os.path.join(config["dataset"]["root_path"], config["dataset"]["output_folder"])
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(args.input))[0]
    out_suffix = config["dataset"]["output_suffix"]
    
    tif_save_path = os.path.join(output_dir, f"{filename}_{out_suffix}.tif")
    png_save_path = os.path.join(output_dir, f"{filename}_{out_suffix}.png")
    
    # Save the true georeferenced GeoTIFF
    geo_profile.update(
        driver="GTiff", 
        height=h, 
        width=w, 
        count=3, 
        dtype=rasterio.uint8, 
        tiled=True
    )
    # Safely clear block properties inherited from the original imagery asset
    geo_profile.pop("blockxsize", None)
    geo_profile.pop("blockysize", None)

    # Save the true georeferenced GeoTIFF
    with rasterio.open(tif_save_path, "w", **geo_profile) as dst:
        dst.write(np.transpose(colored_output, (2, 0, 1)))
        
    # Save the compressed visual PNG layout
    cv2.imwrite(png_save_path, cv2.cvtColor(colored_output, cv2.COLOR_RGB2BGR))
    print(f"🎉 Complete!\n➡️ GIS Asset: {tif_save_path}\n➡️ Visual: {png_save_path}")

if __name__ == "__main__":
    main()























# # ============== mask.pn and mask.tif georeference working =======
# import os
# import argparse
# import numpy as np
# import rasterio
# import cv2
# from tensorflow.keras.models import load_model

# from src.data_utils import load_config, get_vectorized_mappings
# from src.metrics import total_loss, jaccard_coef

# def parse_args():
#     parser = argparse.ArgumentParser(description="Production Seamless Overlap GeoTIFF Inference")
#     parser.add_argument("--input", type=str, required=True, help="Path to input satellite GeoTIFF raster (.tif)")
#     parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration yaml")
#     parser.add_argument("--batch_size", type=int, default=32, help="Number of patches passed concurrently to GPU")
#     return parser.parse_args()

# def generate_gaussian_weights(patch_size):
#     """Generates a 2D Gaussian window to smoothly blend overlapping edges."""
#     w = cv2.getGaussianKernel(patch_size, patch_size / 3)
#     w_2d = np.outer(w, w)
#     return w_2d / w_2d.max()

# def main():
#     args = parse_args()
#     config = load_config(args.config)
    
#     weights_path = config["model"]["weights_path"]
#     if not os.path.exists(weights_path):
#         raise FileNotFoundError(f"Model weights missing at: {weights_path}")
        
#     print(f"🏗️ Initializing model graphs and loading custom loss states...")
#     model = load_model(weights_path, custom_objects={"total_loss": total_loss, "jaccard_coef": jaccard_coef})
    
#     if not os.path.exists(args.input):
#         raise FileNotFoundError(f"Target image asset missing at: {args.input}")
        
#     print(f"Instantiating overlapping streaming grids from source GeoTIFF...")
#     with rasterio.open(args.input) as src:
#         geo_profile = src.profile.copy()
#         target_bands = config["model"].get("raster_bands", [1, 2, 3])
#         large_img_rasterio = src.read(target_bands)
#         large_img = np.transpose(large_img_rasterio, (1, 2, 0))
#         h, w, c = large_img.shape
#         print(f"✅ Loaded native bounds canvas: {w}x{h} ({c} channels)")

#     patch_size = config["model"]["patch_size"]
#     num_classes = config["model"].get("num_classes", 6)
    
#     # 🎯 STEP OVERLAP: Move by half a patch size to blend boundaries smoothly
#     stride = patch_size // 2  
#     blend_window = generate_gaussian_weights(patch_size)
    
#     # Accumulation canvases for probability mapping lists
#     prob_canvas = np.zeros((h, w, num_classes), dtype=np.float32)
#     weight_canvas = np.zeros((h, w), dtype=np.float32)
    
#     # Generate coordinates across the image dimensions using a strict stride
#     y_coords = list(range(0, h - patch_size + 1, stride))
#     if y_coords[-1] != h - patch_size: y_coords.append(h - patch_size)
#     x_coords = list(range(0, w - patch_size + 1, stride))
#     if x_coords[-1] != w - patch_size: x_coords.append(w - patch_size)
    
#     batch_patches = []
#     batch_meta = []
    
#     print(f"⚡ Processing overlapping spatial grid array passes...")
#     for y in y_coords:
#         for x in x_coords:
#             y_end, x_end = y + patch_size, x + patch_size
#             patch = large_img[y:y_end, x:x_end]
            
#             normalized_patch = patch.astype("float32") / 255.0
#             batch_patches.append(normalized_patch)
#             batch_meta.append((y, y_end, x, x_end))
            
#             if len(batch_patches) == args.batch_size:
#                 predictions = model.predict(np.array(batch_patches), batch_size=args.batch_size, verbose=0)
#                 for i in range(len(predictions)):
#                     by, by_end, bx, bx_end = batch_meta[i]
#                     # Accumulate model probabilities multiplied by Gaussian blending weights
#                     prob_canvas[by:by_end, bx:bx_end] += predictions[i] * blend_window[..., np.newaxis]
#                     weight_canvas[by:by_end, bx:bx_end] += blend_window
#                 batch_patches = []
#                 batch_meta = []
                
#     if batch_patches:
#         predictions = model.predict(np.array(batch_patches), batch_size=len(batch_patches), verbose=0)
#         for i in range(len(predictions)):
#             by, by_end, bx, bx_end = batch_meta[i]
#             prob_canvas[by:by_end, bx:bx_end] += predictions[i] * blend_window[..., np.newaxis]
#             weight_canvas[by:by_end, bx:bx_end] += blend_window

#     # Normalize across accumulated layers to protect scaling limits
#     print(f"🎨 Normalizing overlap weight channels and stitching...")
#     weight_canvas[weight_canvas == 0] = 1.0
#     prob_canvas /= weight_canvas[..., np.newaxis]
#     stitched_mask = np.argmax(prob_canvas, axis=-1).astype(np.uint8)

#     # Apply color mappings
#     target_colors, _ = get_vectorized_mappings(config)
#     colored_output = np.zeros((h, w, 3), dtype=np.uint8)
#     for class_idx, rgb_val in target_colors.items():
#         colored_output[stitched_mask == class_idx] = rgb_val
        
#     output_dir = os.path.join(config["dataset"]["root_path"], config["dataset"]["output_folder"])
#     os.makedirs(output_dir, exist_ok=True)
#     filename = os.path.splitext(os.path.basename(args.input))[0]
#     out_suffix = config["dataset"]["output_suffix"]
    
#     tif_save_path = os.path.join(output_dir, f"{filename}_{out_suffix}.tif")
#     png_save_path = os.path.join(output_dir, f"{filename}_{out_suffix}.png")
    
#     # Save the true georeferenced GeoTIFF
#     geo_profile.update(
#         driver="GTiff", 
#         height=h, 
#         width=w, 
#         count=3, 
#         dtype=rasterio.uint8, 
#         tiled=True
#         )
#     # Safely clear block properties inherited from the original imagery asset
#     geo_profile.pop("blockxsize", None)
#     geo_profile.pop("blockysize", None)

#     # Save the true georeferenced GeoTIFF
#     with rasterio.open(tif_save_path, "w", **geo_profile) as dst:
#         dst.write(np.transpose(colored_output, (2, 0, 1)))
        
#     # Save the compressed visual PNG layout
#     cv2.imwrite(png_save_path, cv2.cvtColor(colored_output, cv2.COLOR_RGB2BGR))
#     print(f"🎉 Complete!\n➡️ GIS Asset: {tif_save_path}\n➡️ Visual: {png_save_path}")

# if __name__ == "__main__":
#     main()


# # run 
# python predict.py --input "C:\samcodebase\deeplearning\semantic-segmentation\User_Uploads\1.tif" --batch_size 32
# python predict.py --input "C:\samcodebase\deeplearning\Raw_Image\sentinel2_downloaded.tif" --batch_size 64
# python predict.py --input "C:\samcodebase\deeplearning\Raw_Image\sentinel2_downloaded.tif" --batch_size 128
# python predict.py --input "C:\samcodebase\deeplearning\Raw_Image\sentinel2_downloaded.tif" --batch_size 256
# python predict.py --input "C:\samcodebase\deeplearning\Raw_Image\sentinel2_downloaded.tif" --batch_size 512




