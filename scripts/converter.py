import rasterio
import numpy as np
from PIL import Image
import os

# 1. File Paths
input_tif = r"C:\samcodebase\deeplearning\Raw_Image\lagos1.tif"
output_png = r"C:\samcodebase\deeplearning\Outputs_Converted_Image\lagos1.png"

# 2. Create output directory if it doesn’t exist
png_dir = os.path.dirname(output_png)
os.makedirs(png_dir, exist_ok=True)
print(f"📂 Output directory ready: {png_dir}")

print("🔄 Opening raw satellite TIF image...")

with rasterio.open(input_tif) as src:
    # Read RGB bands if available, otherwise fallback to grayscale
    if src.count >= 3:
        raw_data = src.read([1, 2, 3]) 
        img_array = np.transpose(raw_data, (1, 2, 0))  # (H, W, RGB)
    else:
        raw_data = src.read(1)
        img_array = raw_data

    print(f"📊 Data type: {img_array.dtype} | Max value: {img_array.max()}")

    # Normalize to 0–255 for PNG
    img_min, img_max = img_array.min(), img_array.max()
    if img_max - img_min > 0:
        img_scaled = ((img_array - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        img_scaled = img_array.astype(np.uint8)

    # 3. Resize to a “Windows-friendly” display size (e.g., max width 1600px)
    pil_img = Image.fromarray(img_scaled)
    pil_img.thumbnail((1600, 1600))  # keeps aspect ratio, max size 1600px

    # 4. Save as PNG
    pil_img.save(output_png, format="PNG", optimize=True, quality=85)
    print(f"🎉 Success! Converted preview saved to:\n👉 {output_png}")




# import rasterio
# import numpy as np
# from PIL import Image
# import os

# # 1. File Paths
# #"C:\samcodebase\deeplearning\Raw_Image\lagos1.tif"
# input_tif = r"C:\samcodebase\deeplearning\Raw_Image\lagos1.tif" 
# output_png = r"C:\samcodebase\deeplearning\Outputs_Converted_Image\lagos1.png"

# png_dir = os.path.dirname(output_png)

# if not os.path.exists(png_dir):
#     os.makedirs(png_dir, exist_ok=True)
#     print(f"Created directory for converted PNG output : {png_dir}")

# print("🔄 Opening raw satellite TIF image...")

# with rasterio.open(input_tif) as src:
#     # Read the first 3 bands (Assuming Band 1=Red, Band 2=Green, Band 3=Blue)
#     # If your TIF has fewer bands, change this to src.read(1)
#     if src.count >= 3:
#         raw_data = src.read([1, 2, 3]) 
#         # Transpose from rasterio shape (Bands, H, W) to image shape (H, W, RGB)
#         img_array = np.transpose(raw_data, (1, 2, 0))
#     else:
#         # Fallback for single-band grayscale images
#         raw_data = src.read(1)
#         img_array = raw_data

#     print(f"📊 Original Data Type: {img_array.dtype} | Original Max Value: {img_array.max()}")

#     # 2. Normalize values strictly to 0-255 uint8 format so PNG can read it
#     img_min = img_array.min()
#     img_max = img_array.max()
    
#     # Avoid dividing by zero if the image is blank
#     if img_max - img_min > 0:
#         # Scale the data range seamlessly to 0-255
#         img_scaled = ((img_array - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
#     else:
#         img_scaled = img_array.astype(np.uint8)

#     # 3. Save as standard PNG using Pillow
#     print("💾 Compressing and saving to PNG format...")
#     pil_img = Image.fromarray(img_scaled)
#     pil_img.save(output_png)

#     print(f"🎉 Success! Raw view preview saved to:\n👉 {output_png}")