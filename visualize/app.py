
#  ================= with upload func ================================
# import os
# import sys
# import numpy as np
# import pandas as pd
# import rasterio
# import streamlit as st
# from streamlit_image_comparison import image_comparison

# # Ensure backend engines match system workspace directories
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.data_utils import load_config, get_vectorized_mappings

# def calculate_geospatial_metrics(tif_path, config):
#     """Reads a 3-band RGB GeoTIFF and counts pixels matching class RGB signatures."""
#     if not os.path.exists(tif_path) or not tif_path.lower().endswith(('.tif', '.tiff')):
#         return None
#     try:
#         with rasterio.open(tif_path) as src:
#             transform = src.transform
#             pixel_area_m2 = abs(transform[0]) * abs(transform[4])
            
#             # Read all 3 channels (R, G, B)
#             img_rgb = src.read([1, 2, 3])  # Shape: (3, Height, Width)
#             img_rgb = np.transpose(img_rgb, (1, 2, 0))
            
#         analytics_report = []
#         total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
#         config_classes = config.get("classes", {})
        
#         for class_idx, class_info in config_classes.items():
#             rgb_val = class_info["rgb"] 
#             class_name = class_info["name"]
            
#             # Match pixels where all 3 color values align with the class mapping
#             match_mask = (img_rgb[:, :, 0] == rgb_val[0]) & \
#                          (img_rgb[:, :, 1] == rgb_val[1]) & \
#                          (img_rgb[:, :, 2] == rgb_val[2])
                         
#             pixel_count = np.sum(match_mask)
#             if pixel_count == 0: 
#                 continue
            
#             percentage = (pixel_count / total_pixels) * 100
#             hectares = (pixel_count * pixel_area_m2) / 10000.0
            
#             # 🎯 FIXED: Keys are explicitly bound to match the UI loop variables below
#             analytics_report.append({
#                 "Classification Label": class_name,
#                 "Coverage (%)": percentage,
#                 "Surface Area (Hectares)": hectares,
#                 "color": rgb_val
#             })
            
#         return analytics_report
#     except Exception as e:
#         print(f"⚠️ Metrics Extraction Error: {str(e)}")
#         return None

# def main():
#     st.set_page_config(page_title="Geospatial AI Analytics", layout="wide")
#     config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/config.yaml")
#     config = load_config(config_path)
    
#     st.title("🛰️ Production Geospatial Landcover Analytics Platform")
#     st.markdown("---")
    
#     st.sidebar.header("Input Target Raster")
#     uploaded_file = st.sidebar.file_uploader("Upload a Satellite GeoTIFF (.tif)", type=["tif", "tiff"])

#     raw_img_path, pred_mask_path, tif_spatial_path = None, None, None
#     selected_region_name = "User Uploaded Content"

#     if uploaded_file is not None:
#         uploaded_dir = os.path.join(config["dataset"]["root_path"], "User_Uploads")
#         os.makedirs(uploaded_dir, exist_ok=True)

#         # 1. Track the actual spatial master TIFF file
#         tif_master_input = os.path.join(uploaded_dir, uploaded_file.name)
#         with open(tif_master_input, "wb") as f:
#             f.write(uploaded_file.getbuffer())

#         # 🎯 FIXED: Force-generate a lightweight PNG file so the web slider doesn't choke on a raw 50MB TIFF
#         base_name = os.path.splitext(uploaded_file.name)[0]
#         raw_img_path = os.path.join(uploaded_dir, f"{base_name}_web_view.png")
        
#         if not os.path.exists(raw_img_path):
#             try:
#                 with rasterio.open(tif_master_input) as src:
#                     preview_arr = src.read([1, 2, 3])
#                     preview_arr = np.transpose(preview_arr, (1, 2, 0))
#                     if preview_arr.dtype != np.uint8:
#                         preview_arr = ((preview_arr - preview_arr.min()) / (preview_arr.max() - preview_arr.min()) * 255).astype(np.uint8)
#                     import cv2
#                     cv2.imwrite(raw_img_path, cv2.cvtColor(preview_arr, cv2.COLOR_RGB2BGR))
#             except Exception as e:
#                 raw_img_path = tif_master_input

#         # Map predictions inside output folder structure
#         out_folder = os.path.join(config["dataset"]["root_path"], config["dataset"]["output_folder"])
#         suffix = config["dataset"]["output_suffix"]
        
#         pred_mask_path = os.path.join(out_folder, f"{base_name}_{suffix}.png")
#         tif_spatial_path = os.path.join(out_folder, f"{base_name}_{suffix}.tif")
        
#         if not os.path.exists(tif_spatial_path):
#             st.sidebar.warning("⚡ No active classification mask found for this file.")
#             if st.sidebar.button("🚀 Run AI Segmentation Pipeline", use_container_width=True):
#                 with st.spinner("Streaming overlapping matrix patches to neural network graphs..."):
#                     os.system(f'python predict.py --input "{tif_master_input}" --batch_size 32')
#                     st.rerun()
#                 return
#         else:
#             st.sidebar.success(f"✅ Active mask located!")
#     else:
#         # Configuration Fallback
#         regions = config.get("dashboard", {}).get("regions", [])
#         if not regions:
#             st.error("🚨 No analysis regions found under dashboard configurations in config.yaml.")
#             return
#         region_names = [r["name"] for r in regions]
#         selected_region_name = st.sidebar.selectbox("🎯 Select Preloaded Region", region_names)
#         region_data = next(r for r in regions if r["name"] == selected_region_name)
        
#         raw_img_path = os.path.normpath(region_data["raw_path"])
#         pred_mask_path = os.path.normpath(region_data["pred_path"])
#         base, _ = os.path.splitext(pred_mask_path)
#         tif_spatial_path = base + ".tif"

#     # 📊 2. METRICS EXTRACTION & REPORT EXPORTS
#     st.sidebar.header("📊 Real-World Surface Metrics")
#     metrics_data = calculate_geospatial_metrics(tif_spatial_path, config)
    
#     if metrics_data:
#         for item in metrics_data:
#             hex_color = '#%02x%02x%02x' % tuple(item["color"])
#             st.sidebar.markdown(f"### <span style='color:{hex_color};'>■</span> {item['Classification Label']}", unsafe_allow_html=True)
#             c1, c2 = st.sidebar.columns(2)
#             c1.metric("Coverage", f"{item['Coverage (%)']:.2f}%")
#             c2.metric("Hectares", f"{item['Surface Area (Hectares)']:.1f} ha")
#             st.sidebar.markdown("---")
            
#         st.sidebar.header("📤 Export Matrix Report")
        
#         # CSV Export
#         df_metrics = pd.DataFrame(metrics_data).drop(columns=["color"]) 
#         csv_data = df_metrics.to_csv(index=False).encode('utf-8')
#         st.sidebar.download_button(
#             label="📊 Download Metrics Report (.CSV)",
#             data=csv_data,
#             file_name=f"{selected_region_name.lower().replace(' ', '_')}_landcover_report.csv",
#             mime="text/csv",
#             use_container_width=True
#         )
        
#         # PNG Export
#         if os.path.exists(pred_mask_path):
#             with open(pred_mask_path, "rb") as file:
#                 png_bytes = file.read()
#             st.sidebar.download_button(
#                 label="🎨 Download Presentation Mask (.PNG)",
#                 data=png_bytes,
#                 file_name=f"{selected_region_name.lower().replace(' ', '_')}_mask.png",
#                 mime="image/png",
#                 use_container_width=True
#             )
            
#         # GeoTIFF Export
#         if os.path.exists(tif_spatial_path):
#             with open(tif_spatial_path, "rb") as file:
#                 tif_bytes = file.read()
#             st.sidebar.download_button(
#                 label="🗺️ Download GeoTIFF Layer (.TIF)",
#                 data=tif_bytes,
#                 file_name=f"{selected_region_name.lower().replace(' ', '_')}_spatial.tif",
#                 mime="image/tiff",
#                 use_container_width=True
#             )
#     else:
#         st.sidebar.info("ℹ️ Spatial metrics unavailable. Upload a valid GeoTIFF map layer file.")

#     # 🗺️ MAIN MAP DISPLAY VIEWPORTS
#     st.subheader(f"Inspection Vector View: {selected_region_name}")
#     zoom = st.slider("🔍 Interactive Zoom Level (%)", min_value=50, max_value=200, value=100, step=10)
#     dynamic_width = int(config.get("dashboard", {}).get("default_width", 1200) * zoom / 100)
    
#     # 🎯 FIXED: Direct path verification variables check prior to rendering the view canvas layout
#     if raw_img_path and pred_mask_path and os.path.exists(raw_img_path) and os.path.exists(pred_mask_path):
#         image_comparison(
#             img1=raw_img_path, img2=pred_mask_path,
#             label1="Original Input Satellite View", label2="Model Prediction Mask Output",
#             starting_position=50, show_labels=True, make_responsive=False,
#             width=dynamic_width, in_memory=False
#         )
#     else:
#         st.info("👋 Upload a satellite `.tif` image from the sidebar panel workspace to begin automated landcover extraction mapping.")

# if __name__ == "__main__":
#     main()












# ===================== without upload =======================
import os
import sys
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from streamlit_image_comparison import image_comparison

# Ensure backend engines match system workspace directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_config, get_vectorized_mappings



def calculate_geospatial_metrics(tif_path, config):
    """Reads a 3-band RGB GeoTIFF and counts pixels matching class RGB signatures."""
    if not os.path.exists(tif_path) or not tif_path.lower().endswith(('.tif', '.tiff')):
        return None
    try:
        with rasterio.open(tif_path) as src:
            transform = src.transform
            pixel_area_m2 = abs(transform[0]) * abs(transform[4])
            
            # Read all 3 channels (R, G, B)
            img_rgb = src.read([1, 2, 3])  # Shape: (3, Height, Width)
            img_rgb = np.transpose(img_rgb, (1, 2, 0))
            
        analytics_report = []
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        
        #  FIX: Read raw integer classes directly from the base configuration file layer
        config_classes = config.get("classes", {})
        
        for class_idx, class_info in config_classes.items():
            # Get the exact 0-255 integer RGB array from your config
            rgb_val = class_info["rgb"] 
            class_name = class_info["name"]
            
            # Match pixels where all 3 color values align with the class mapping
            match_mask = (img_rgb[:, :, 0] == rgb_val[0]) & \
                         (img_rgb[:, :, 1] == rgb_val[1]) & \
                         (img_rgb[:, :, 2] == rgb_val[2])
                         
            pixel_count = np.sum(match_mask)
            if pixel_count == 0: 
                continue
            
            percentage = (pixel_count / total_pixels) * 100
            hectares = (pixel_count * pixel_area_m2) / 10000.0
            
            analytics_report.append({
                "label": class_name,  # 🎯 Clean string output ("Water", "Vegetation")
                "pct": percentage,
                "hectares": hectares,
                "color": rgb_val
            })
            
        return analytics_report
    except Exception as e:
        print(f"⚠️ Metrics Extraction Error: {str(e)}")
        return None




def main():
    st.set_page_config(page_title="Geospatial AI Analytics", layout="wide")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/config.yaml")
    config = load_config(config_path)
    
    st.title("🛰️ Production Geospatial Landcover Analytics Platform")
    st.markdown("---")
    
           # 1. Selector Matrix
    regions = config.get("dashboard", {}).get("regions", [])
    if not regions:
        st.error("🚨 No analysis regions found under dashboard configurations in config.yaml.")
        return
            
    region_names = [r["name"] for r in regions]
    selected_region_name = st.sidebar.selectbox("🎯 Select Analysis Target Area", region_names)
        
        # Fetch parameters for the chosen region
    region_data = next(r for r in regions if r["name"] == selected_region_name)
        
        # Normalize paths to resolve OS-specific slash mismatches
    raw_img_path = os.path.normpath(region_data["raw_path"])
    pred_mask_path = os.path.normpath(region_data["pred_path"])
        
        # Force clean cross-platform path resolution
    base, ext = os.path.splitext(pred_mask_path)
    tif_spatial_path = base + ".tif"
    
    # 📊 SIDEBAR: Calculate landcover asset distribution metrics
    st.sidebar.header("📊 Real-World Surface Metrics")
    
    # Debug info hidden cleanly in an expander for inspection
    with st.sidebar.expander("🔍 Diagnostics Path Check"):
        st.write(f"Looking for mask image: `{pred_mask_path}`")
        st.write(f"Looking for spatial TIF: `{tif_spatial_path}`")
        st.write(f"TIF File exists? `{os.path.exists(tif_spatial_path)}`")

    metrics_data = calculate_geospatial_metrics(tif_spatial_path, config)
    
    if metrics_data:
        for item in metrics_data:
            hex_color = '#%02x%02x%02x' % tuple(item["color"])
            
            # 🟢 FIXED: Split the markdown and column assignment onto separate lines correctly
            st.sidebar.markdown(f"### <span style='color:{hex_color};'>■</span> {item['label']}", unsafe_allow_html=True)
            c1, c2 = st.sidebar.columns(2)
            c1.metric("Coverage", f"{item['pct']:.2f}%")
            c2.metric("Hectares", f"{item['hectares']:.1f} ha")
            st.sidebar.markdown("---")
    else:
        st.sidebar.info("ℹ️ Spatial metrics unavailable for this region target (Requires an active GeoTIFF source map).")

    # 🗺️ MAIN MAP DISPLAY VIEWPORTS
    st.subheader(f"Inspection Vector View: {selected_region_name}")
    
# 🔍 ZOOM SLIDER ENGINE: Single clean configuration instantiation
    zoom = st.slider("🔍 Interactive Zoom Level (%)", min_value=50, max_value=200, value=100, step=10)
    
    # Calculate target display width based on your configuration defaults
    base_width = config.get("dashboard", {}).get("default_width", 1200)
    dynamic_width = int(base_width * zoom / 100)
    
    if os.path.exists(raw_img_path) and os.path.exists(pred_mask_path):
        image_comparison(
            img1=raw_img_path,
            img2=pred_mask_path,
            label1=region_data.get("label_raw", "Original Satellite Imagery"),
            label2=region_data.get("label_pred", "Model Prediction Mask"),
            starting_position=50,
            show_labels=True,
            # make_responsive=region_data.get("responsive", True),
            make_responsive=False,
            width=dynamic_width,
            in_memory=False
        )
    else:
        st.error("🚨 Path validation error on current dashboard configuration choice!")
        st.write(f"Verify Input Path: `{raw_img_path}`")
        st.write(f"Verify Output Path: `{pred_mask_path}`")



if __name__ == "__main__":
    main()

