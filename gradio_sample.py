"""
app.py - Gradio UI for Satellite Landcover Segmentation
Description: Interactive web application to drag-and-drop satellite imagery 
             and view real-time U-Net segmentation predictions.
"""

import os
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import load_model

# 1. Import your custom metrics so load_model doesn't crash
from src.metrics import total_loss, jaccard_coef, dice_loss

# 2. Path to your existing model
MODEL_PATH = r"C:\samcodebase\deeplearning\Trained_Models\model_v1\best_unet_satellite_model_v1.22026-06-19_12-56-31.keras"

print("[+] Loading trained U-Net model...")
model = load_model(
    MODEL_PATH, 
    custom_objects={
        "dice_loss": dice_loss,
        "total_loss": total_loss,
        "jaccard_coef": jaccard_coef,
        "CategoricalFocalCrossentropy": tf.keras.losses.CategoricalFocalCrossentropy
    }
)
print("[+] Model loaded successfully!")

# 3. Define the prediction pipeline function
def segment_satellite_image(input_image):
    if input_image is None:
        return None
        
    # Capture original shape to resize the output mask back to its original dimensions later
    orig_h, orig_w, _ = input_image.shape
    
    # Preprocess: Resize to match your U-Net input shape (256x256) and normalize
    img_resized = tf.image.resize(input_image, (256, 256))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0) # Add batch dimension -> (1, 256, 256, 3)
    
    # Run Inference
    prediction = model.predict(img_expanded) # Output shape: (1, 256, 256, 6)
    
    # Postprocess: Take the argmax across the 6 channels to get the dominant class per pixel
    mask_indices = np.argmax(prediction[0], axis=-1) # Shape: (256, 256)
    
    # Define a distinct color palette for your landcover classes (RGB values)
    # Customize these colors to match whatever scheme you used in your tutorials!
    colors = np.array([
        [0, 255, 255],    # Class 0: e.g., Water (Cyan)
        [255, 255, 0],    # Class 1: e.g., Land/Bare Soil (Yellow)
        [255, 0, 255],    # Class 2: e.g., Urban/Roads (Magenta)
        [0, 255, 0],      # Class 3: e.g., Vegetation/Trees (Green)
        [0, 0, 255],      # Class 4: e.g., Buildings (Blue)
        [0, 0, 0]         # Class 5: e.g., Background/Unclassified (Black)
    ], dtype=np.uint8)
    
    # Map the mask indices to actual RGB colors
    rgb_mask = colors[mask_indices]
    
    # Resize back to original user uploaded dimensions for clean presentation
    final_mask = tf.image.resize(rgb_mask, (orig_h, orig_w), method="nearest").numpy().astype(np.uint8)
    
    return final_mask

# 4. Construct the Web Layout using Blocks for a polished UI
with gr.Blocks(title="Satellite Landcover Segmentation") as demo:
    gr.Markdown("## 🛰️ U-Net Satellite Imagery Landcover Segmentation Dashboard")
    gr.Markdown("Drag and drop any high-resolution satellite imagery frame to see your trained U-Net segment classes in real-time.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Satellite Tile Frame", type="numpy")
            submit_btn = gr.Button("Analyze Landscape Structure", variant="primary")
        with gr.Column():
            output_img = gr.Image(label="Predicted Multi-Class Segmentation Mask", type="numpy")
            
    # Set button execution trigger
    submit_btn.click(fn=segment_satellite_image, inputs=input_img, outputs=output_img)

# 5. Launch local server
if __name__ == "__main__":
    demo.launch()