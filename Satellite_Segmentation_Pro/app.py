"""
app.py - Live Hugging Face Space Deployment
Framework: Gradio 6.19.0
"""

import os
import numpy as np
import tensorflow as tf
import gradio as gr

# Force TensorFlow to run strictly on CPU inside the Space container
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 1. Load your local model weights safely
MODEL_PATH = "model.keras"

print("[+] Initializing model loader inside Space backend...")
# compile=False allows us to load the U-Net architecture instantly without needing your local loss source code
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("[+] Model built and cached successfully!")

# 2. Setup inference pipeline
def predict_satellite_mask(input_img):
    if input_img is None:
        return None
        
    orig_h, orig_w, _ = input_img.shape
    
    # Preprocess to match your U-Net input expectations (256x256, normalized)
    img_resized = tf.image.resize(input_img, (256, 256))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    # Forward Pass
    prediction = model.predict(img_expanded)
    mask_indices = np.argmax(prediction[0], axis=-1) # Extract dominant class index per pixel
    
    # Map class indices to distinct visualization colors (RGB)
    colors = np.array([
        [0, 255, 255],    # Class 0: Cyan
        [255, 255, 0],    # Class 1: Yellow
        [255, 0, 255],    # Class 2: Magenta
        [0, 255, 0],      # Class 3: Green
        [0, 0, 255],      # Class 4: Blue
        [0, 0, 0]         # Class 5: Black
    ], dtype=np.uint8)
    
    rgb_mask = colors[mask_indices]
    
    # Resize back to match the original uploaded resolution beautifully
    final_output = tf.image.resize(rgb_mask, (orig_h, orig_w), method="nearest").numpy().astype(np.uint8)
    return final_output

# 3. Design the Web App Interface Layout
with gr.Blocks(title="Satellite Segmentation Pro") as demo:
    gr.Markdown("# 🛰️ Satellite Segmentation Pro Dashboard")
    gr.Markdown("Drop any aerial frame or satellite tile below to run real-time semantic segmentation via your trained deep learning U-Net.")
    
    with gr.Row():
        with gr.Column():
            src_input = gr.Image(label="Source Satellite Image Tile", type="numpy")
            btn_run = gr.Button("Execute Semantic Extraction", variant="primary")
        with gr.Column():
            mask_output = gr.Image(label="Segmented Landcover Mask Output", type="numpy")
            
    btn_run.click(fn=predict_satellite_mask, inputs=src_input, outputs=mask_output)

if __name__ == "__main__":
    demo.launch()