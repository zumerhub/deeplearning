import numpy as np

def slide_over_image(image, patch_size=256):
    """
    Slices a continuous geospatial raster into uniform matrix frames.
    Dynamically pads edge boundaries to prevent U-Net shape dropouts.
    """
    height, width, channels = image.shape
    
    # Loop over rows (Y axis) stepping by patch size
    for y in range(0, height, patch_size):
        # Loop over columns (X axis) stepping by patch size
        for x in range(0, width, patch_size):
            
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            
            # Slice out the native local patch matrix
            patch = image[y:y_end, x:x_end, :]
            
            # 🎯 FIX: Calculate boundary padding discrepancies dynamically
            pad_y = patch_size - patch.shape[0]
            pad_x = patch_size - patch.shape[1]
            
            # If the patch hits an edge, pad it safely with zeros to keep the shape uniform
            if pad_y > 0 or pad_x > 0:
                patch = np.pad(
                    patch, 
                    ((0, pad_y), (0, pad_x), (0, 0)), 
                    mode="constant", 
                    constant_values=0.0
                )
            
            # Yield the perfectly shaped patch alongside structural metadata for reassembly
            yield patch, y, y_end, x, x_end, pad_y, pad_x




# import numpy as np

# def slide_over_image(image, patch_size=256):
#     height, width, channels = image.shape
    
#     # Loop over rows (Y axis) stepping by patch size
#     for y in range(0, height, patch_size):
#         # Loop over columns (X axis) stepping by patch size
#         for x in range(0, width, patch_size):
            
#             # Find boundary constraints so we don't bleed past the image edge
#             y_end = min(y + patch_size, height)
#             x_end = min(x + patch_size, width)
            
#             # Slice out the local patch matrix
#             patch = image[y:y_end, x:x_end, :]
            
#             # (Optional) If the patch is too small at the edge, pad it with zeros
#             # Then pass it to model.predict()
#             yield patch, y, y_end, x, x_end


            