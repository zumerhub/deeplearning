# img_raw_1 = r"C:\samcodebase\deeplearning\Raw_Image\sentinel2_downloaded.png"
# img_pred_2 = r"C:\samcodebase\deeplearning\New_1_sentinel2_downloaded_mask_quickview.png"

img_raw = r"C:\samcodebase\deeplearning\Outputs_Converted_Image\lagos1.png"


# from PIL import Image

# img = Image.open(img_raw)
# # img = img.resize((1024, 1024))  # shrink dimensions
# img.thumbnail((2048, 2048))  # keep aspect ratio, max size
# img.save("compressed_new.png", optimize=True, width=800, quality=85)


import streamlit as st
from PIL import Image

# Load original image
img = Image.open(img_raw)

# Get browser width (approximate via Streamlit layout)
window_width = st.get_option("browser.gatherUsageStats")  # not exact, but you can set manually

# Resize based on desired display width
target_width = 1200  # match your app’s layout width
aspect_ratio = img.height / img.width
new_height = int(target_width * aspect_ratio)

img_resized = img.resize((target_width, new_height))
img_resized.save("Lagos_compressed.png", optimize=True, quality=85)

# Display with Streamlit
st.image("compressed.png", use_column_width=True)



# import rasterio
# from rasterio.windows import Window

# input_tif = r"C:\samcodebase\deeplearning\Raw_Image\lagos1_view.png"
# test_tif = r"C:\samcodebase\deeplearning\Raw_Image\Compressed_lagos1_mini.png"

# with rasterio.open(input_tif) as src:
#     # Crop a 1024x1024 window from the top-left corner
#     window = Window(0, 0, 1024, 1024)
#     kwargs = src.meta.copy()
#     kwargs.update({
#         'height': 1024,
#         'width': 1024,
#         'transform': rasterio.windows.transform(window, src.transform)
#     })
    
#     with rasterio.open(test_tif, 'w', **kwargs) as dst:
#         dst.write(src.read(window=window))

# print("⚡ Mini 1024x1024 test tile created at lagos1_mini.tif!")