# ============== updated working ==================
"""
satellite_data.py - Robust Data Ingestion & Tokenization Pipeline
Location: C:\\samcodebase\\deeplearning\\satellite_data.py
Description: Vectorized, error-resilient patching utility with uniform scaling, 
             strict classification depth, and seamless training exports.
"""

import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


from src.data_utils import load_config


# =====================================================================
# 1. PATH CONFIGURATIONS & GLOBAL PARAMETERS
# =====================================================================
dataset_root_folder = r"C:\samcodebase\deeplearning\semantic-segmentation"
dataset_name = "Semantic_segmentation_dataset"
image_patch_size = 256

image_dataset = []
mask_dataset = []

image_extension = "jpg"
mask_extension = "png"

# =====================================================================
# 2. DEFINITIVE UNIFIED TAXONOMY REFERENCE (6 Classes Native)
# =====================================================================


# Senior Dev Path Fix: Construct an explicit absolute path to your configuration file
# Base path: C:\samcodebase\deeplearning
project_root = r"C:\samcodebase\deeplearning"
config_path = os.path.join(project_root, "config", "config.yaml")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"[-] Central configuration brain missing at absolute path: {config_path}")

# Safely load configurations dynamically using the src utility
config = load_config(config_path)

# Extract mappings dynamically from YAML
class_data = config["classes"]
TRUE_CLASS_COUNT = len(class_data)

# Reconstruct class_colors dictionary and numpy target matching vectors dynamically
class_colors = {int(idx): info["rgb"] for idx, info in class_data.items()}

class_water      = np.array(class_colors[0])
class_land       = np.array(class_colors[1])
class_road       = np.array(class_colors[2])
class_building   = np.array(class_colors[3])
class_vegetation = np.array(class_colors[4])
class_unlabeled  = np.array(class_colors[5])

# =====================================================================
# 3. DIRECTORY SCANNING & VECTOR CHIPPING PIPELINE
# =====================================================================
print("[*] Initiating Geospatial Raster Extraction Sequence...")

for tile_id in range(1, 9):  # Tiles 1–8
    for image_id in range(1, 10):  # Scans part items 1-9 per directory
        
        image_path = os.path.join(dataset_root_folder, dataset_name, f"Tile {tile_id}", "images", f"image_part_{image_id:03d}.{image_extension}")
        mask_path = os.path.join(dataset_root_folder, dataset_name, f"Tile {tile_id}", "masks", f"image_part_{image_id:03d}.{mask_extension}")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        # Ingest arrays via OpenCV standard configurations
        img = cv2.imread(image_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure regular RGB sequence tracking
        mask = cv2.imread(mask_path, 1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Slice down layout dimensions to clean patches multiples
        size_x = (img.shape[1] // image_patch_size) * image_patch_size
        size_y = (img.shape[0] // image_patch_size) * image_patch_size

        img = img[0:size_y, 0:size_x]
        mask = mask[0:size_y, 0:size_x]

        # Break frames down into individual uniform spatial patches
        patched_images = patchify(img, (image_patch_size, image_patch_size, 3), step=image_patch_size)
        patched_masks = patchify(mask, (image_patch_size, image_patch_size, 3), step=image_patch_size)

        for i in range(patched_images.shape[0]):
            for j in range(patched_images.shape[1]):
                img_patch = patched_images[i, j, 0]
                
                # Senior Dev Optimization: Uniform min-max channel arrays mapping safely without distortion
                img_patch_normalized = img_patch.astype(np.float32) / 255.0
                image_dataset.append(img_patch_normalized)

                mask_patch = patched_masks[i, j, 0]
                mask_dataset.append(mask_patch)

# Cast raw list variables to structural arrays
image_dataset = np.array(image_dataset, dtype=np.float32)
mask_dataset = np.array(mask_dataset, dtype=np.uint8)

print(f"[+] Total Extracted Dataset Patches: {len(image_dataset)}")

# =====================================================================
# 4. VECTORIZED RGB TO LABEL INTEGER INDEX TRANSFORMATION
# =====================================================================
def rgb_to_label(mask_array):
    """Executes high-speed matrix mapping to generate structural class index integers."""
    label_segment = np.zeros(mask_array.shape[:2], dtype=np.uint8)
    
    label_segment[np.all(mask_array == class_water, axis=-1)] = 0
    label_segment[np.all(mask_array == class_land, axis=-1)] = 1
    label_segment[np.all(mask_array == class_road, axis=-1)] = 2
    label_segment[np.all(mask_array == class_building, axis=-1)] = 3
    label_segment[np.all(mask_array == class_vegetation, axis=-1)] = 4
    label_segment[np.all(mask_array == class_unlabeled, axis=-1)] = 5
    
    return label_segment

print("[*] Remapping color masks to indexed categorical matrices...")
labels = np.array([rgb_to_label(m) for m in mask_dataset], dtype=np.uint8)
labels = np.expand_dims(labels, axis=-1) # Dimensions set cleanly to (N, 256, 256, 1)

# Ensure output classes depth is strictly locked to target metrics
labels_categorical_dataset = to_categorical(labels, num_classes=TRUE_CLASS_COUNT)

# =====================================================================
# 5. STRATIFIED MODEL TRAINING SEPARATION & EXPORTS
# =====================================================================
x_train, x_test, y_train, y_test = train_test_split(
    image_dataset, labels_categorical_dataset, test_size=0.15, random_state=100
)

# Export metadata attributes cleanly to upstream files like training.py
image_height = x_train.shape[1]
image_width = x_train.shape[2]
image_channels = x_train.shape[3]
total_classes = TRUE_CLASS_COUNT

print("\n[+] System Metadata Exports Ready:")
print(f"    - Training Shape inputs: {x_train.shape}")
print(f"    - Target Classification Depth: {total_classes} Classes locked.")





# ================ working ==============================
# import os
# import cv2
# import numpy as np
# from PIL import Image
# from patchify import patchify
# from sklearn.preprocessing import MinMaxScaler
# from matplotlib import pyplot as plt
# import random
# import matplotlib.colors as mcolors


# dataset_root_folder = r"C:\samcodebase\deeplearning\semantic-segmentation"
# dataset_name = "Semantic_segmentation_dataset"

# image_patch_size = 256
# minmaxscaler = MinMaxScaler()

# image_dataset = []
# mask_dataset = []

# # keep extensions explicit
# image_extension = "jpg"
# mask_extension = "png"

# for tile_id in range(1, 9):  # Tiles 1–8
#     for image_id in range(1, 10):  # adjust to actual number of images per tile
#         # image path
#         image_path = (
#             f"{dataset_root_folder}\\{dataset_name}\\Tile {tile_id}\\images\\image_part_{image_id:03d}.{image_extension}"
#         )
#         # mask path
#         mask_path = (
#             f"{dataset_root_folder}\\{dataset_name}\\Tile {tile_id}\\masks\\image_part_{image_id:03d}.{mask_extension}"
#         )

#         image = cv2.imread(image_path, 1)
#         mask  = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # load mask in BGR

#         if image is None:
#             print(f"Could not read image {image_path}")
#             continue
#         if mask is None:
#             print(f"Could not read mask {mask_path}")
#             continue

#         # convert mask to RGB if needed
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

#         # crop sizes to patch multiples
#         size_x = (image.shape[1] // image_patch_size) * image_patch_size
#         size_y = (image.shape[0] // image_patch_size) * image_patch_size

#         # crop image and mask
#         image = Image.fromarray(image).crop((0, 0, size_x, size_y))
#         mask  = Image.fromarray(mask).crop((0, 0, size_x, size_y))

#         image = np.array(image)
#         mask  = np.array(mask)

#         # patchify
#         patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)
#         patched_masks  = patchify(mask, (image_patch_size, image_patch_size, 3), step=image_patch_size)

#         # print(f"Tile {tile_id}, Image {image_id}, size: {image.shape}, patches: {patched_images.shape[0] * patched_images.shape[1]}")

#         for i in range(patched_images.shape[0]):
#             for j in range(patched_images.shape[1]):
#                 # image patch
#                 img_patch = patched_images[i, j, :, :][0]
#                 reshaped = img_patch.reshape(-1, img_patch.shape[-1])
#                 scaled   = minmaxscaler.fit_transform(reshaped)
#                 img_patch = scaled.reshape(img_patch.shape)
#                 image_dataset.append(img_patch)

#                 # mask patch
#                 individual_patched_masks = patched_masks[i, j, :, :][0]
#                 mask_dataset.append(individual_patched_masks)
# # convert into np array
# image_dataset = np.array(image_dataset)
# mask_dataset = np.array(mask_dataset)
              

# print(f"Total image patches: {len(image_dataset)}")
# print(f"Total mask patches: {len(mask_dataset)}")

# # #  ======== visualize the image ==============
# # random_image_id = random.randint(0, len(image_dataset))
# # plt.figure(figsize=(14, 5))   # height must be > 0

# # plt.subplot(121)
# # plt.imshow(image_dataset[random_image_id].squeeze())   # now shape is (256, 256, 3)
# # plt.title("Image Patch")

# # plt.subplot(122)
# # plt.imshow(mask_dataset[random_image_id].squeeze())    # now shape is (256, 256, 3) or (256, 256)
# # plt.title("Mask Patch")

# # plt.show()
# # image_dataset = np.array(image_dataset[0])
# # print(image_dataset)
# # mask_dataset = np.array(mask_dataset[0])
# # print(mask_dataset)



# # ========= Colour class =======
# class_colors = {
#     0: [60, 16, 152],    # Building (#3C1098)
#     1: [132, 41, 246],   # Land (#8429F6)
#     2: [110, 193, 228],  # Road (#6EC1E4)
#     3: [254, 221, 58],   # Vegetation (#FEDD3A)
#     4: [226, 169, 41],   # Water (#E2A929)
#     5: [155, 155, 155]   # Unlabeled (#9B9B9B)
# }
 
# # Normalize RGB values to 0–1
# color_list = [np.array(rgb)/255.0 for rgb in class_colors.values()]

# # Create custom colormap
# custom_cmap = mcolors.ListedColormap(color_list)

# # ================= label Color ==================
# class_building = '#3C1098'
# class_building = class_building.lstrip('#')
# class_building = np.array(tuple(int(class_building[i:i+2], 16) for i in (0,2,4)))

# class_land = '#8429F6'
# class_land = class_land.lstrip('#')
# class_land = np.array(tuple(int(class_land[i:i+2], 16) for i in (0,2,4)))

# class_road = '#6EC1E4'
# class_road = class_road.lstrip('#')
# class_road = np.array(tuple(int(class_road[i:i+2], 16) for i in (0,2,4)))

# class_vegetation = '#FEDD3A'
# class_vegetation = class_vegetation.lstrip('#')
# class_vegetation = np.array(tuple(int(class_vegetation[i:i+2], 16) for i in (0,2,4)))

# class_water = '#E2A929'
# class_water = class_water.lstrip('#')
# class_water = np.array(tuple(int(class_water[i:i+2], 16) for i in (0,2,4)))

# class_unlabeled = '#9B9B9B'
# class_unlabeled = class_unlabeled.lstrip('#')
# class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2], 16) for i in (0,2,4)))

# # ==================== label injection ===============

# mask_dataset.shape[0]
# print("this working : ", mask_dataset.shape)

# label = individual_patched_masks

# def rgb_to_label(label):

#     label_segment = np.zeros(label.shape[:2], dtype=np.uint8) # 2D label map
#     # print("segment label shape-------", label_segment.shape)
#     label_segment[np.all(label == class_water, axis=-1)] = 0
#     label_segment[np.all(label == class_land, axis=-1)] = 1
#     label_segment[np.all(label == class_road, axis=-1)] = 2
#     label_segment[np.all(label == class_building, axis=-1)] = 3
#     label_segment[np.all(label == class_vegetation, axis=-1)] = 4
#     label_segment[np.all(label == class_unlabeled, axis=-1)] = 5

#     # label_segment = label_segment[:,:,0]  # index
#     # print("label segment: ", label_segment)
      
#     return label_segment

# labels = []
# for i in range(mask_dataset.shape[0]):
#     label = rgb_to_label(mask_dataset[i])
#     labels.append(label)
# labels = np.array(labels)
# # print("Labels shape: ", labels.shape)
# # print("len --- shape: ", len(labels))
# # print("label range: ", labels[3])
# labels = np.expand_dims(labels, axis=3)
# # print("expand-label is -- ", labels)

# unique_label = np.unique(labels)
# print("Total unique labels based on masks :", format(np.unique(labels)))



# #  ======== visualize the image ==============
# # random_image_id = random.randint(0, len(image_dataset)-1)
# random_image_id = np.random.choice(len(image_dataset))
# plt.figure(figsize=(14, 5))   # height must be > 0
# plt.subplot(121)
# plt.imshow(image_dataset[random_image_id].squeeze())   # now shape is (256, 256, 3)
# plt.title("Image Patch")

# plt.subplot(122)
# plt.imshow(mask_dataset[random_image_id].squeeze())    # now shape is (256, 256, 3) or (256, 256)
# # plt.imshow(labels[random_image_id][:, :, 0])    # now shape is (256, 256, 3) or (256, 256)
# plt.imshow(labels[random_image_id], cmap=custom_cmap, vmin=0, vmax=len(class_colors)-1)
# plt.title("Mask Patch (Original Colors)")

# # plt.show()


# # ====== Master Training dataset ==========
# total_classes = len(np.unique(labels))
# print(total_classes)


# from tensorflow.keras.utils import to_categorical
# labels_categorical_dataset = to_categorical(labels, num_classes=total_classes)
# print("label_categorical_datasets: ", labels_categorical_dataset.shape)


# master_training_dataset = image_dataset
# master_training_dataset.shape
# print("master_training", master_training_dataset.shape)
# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(master_training_dataset, labels_categorical_dataset, test_size=0.15, random_state=100) 
# print(f"Image sizes =======================")
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)


# image_height = x_train.shape[1]
# image_width = x_train.shape[2]
# image_channels = x_train.shape[3]
# total_classes = y_train.shape[-1]

# print("image ============================================")
# print(f"Image height is : ", image_height)
# print(f"Image width is : ", image_width)
# print(f"Total channels is : ", image_channels)
# print(f"Total classes is : ", total_classes)


# # ============ U-Net Model Setup package (pip install -U segmentation-models) =================










































# def rgb_to_label(label):
#     # Create an empty array with the same shape as label
#     label_segment = np.zeros(label.shape, dtype=np.uint8)
#     print(label_segment.shape)
    
#     # Example: set all pixels to 1 (you can replace this with your logic)
#     label_segment[:] = 1
    
#     return label_segment

# labels = []
# for i in range(mask_dataset.shape[0]):
#     label = rgb_to_label(mask_dataset[i])
#     labels.append(label)

# print(len(labels))

# labels = np.array(labels)
















# # $env:TF_ENABLE_ONEDNN_OPTS=0

# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# from PIL import Image
# from patchify import patchify
# from sklearn.preprocessing import MinMaxScaler, StandardScaler


# dataset_root_folder = r"C:\samcodebase\deeplearning\semantic-segmentation" #"/home/zumerhub/codebase/u-net-deepl-seg/semantic-segmentation/Semantic_segmentation_datasets"
# # dataset_name = os.path.join(dataset_root_folder, "Semantic_segmentation_dataset")
# dataset_name = "Semantic_segmentation_dataset"

# minmaxscaler = MinMaxScaler()

# for path, subdirs, files in os.walk(os.path.join(dataset_root_folder, dataset_name)):
#     dir_name = path.split(os.path.sep)[-1]
#     # print(f"\n📁 Directory: {dir_name}")
#     if dir_name == "images":  # masks
#         images = os.listdir(path)
#         print(path)
#         # print(images)

#         for i, image_name in enumerate(images):
#             if (image_name.endswith('.jpg')):   # .png
#                 # print(image_name)
#                 a = True


# # image = cv2.imread(f"{dataset_root_folder}\{dataset_name}\Tile 1\images\image_part_001.jpg ")

# # image = cv2.imread(r"C:\samcodebase\deeplearning\semantic-segmentation\Semantic_segmentation_dataset\Tile 1\images\image_part_001.jpg",1)
# # print(image)
# # print(image.shape)


# # =============== Processing Satellite Images ===========


# image_patch_size = 256
# # (image.shape[0]//image_patch_size) * image_patch_size


# image_dataset = []
# mask_dataset = []

# for image_type in ['imaged', 'masks']:
#     if image_type == 'images':
#         image_extension = "jpg"
#     elif image_type == "masks":
#         image_extension = "png"

# for tile_id in range(1, 9):  # Tiles 1–8
#     for image_id in range(1, 10):  # adjust to actual number of images per tile
#         # image path
#         image_path = (
#             f"{dataset_root_folder}\\{dataset_name}\\Tile {tile_id}\\images\\image_part_{image_id:03d}.{image_extension}"
#         )
     
#         image = cv2.imread(image_path, 1)
      
#         if image is None:
#             print(f"Could not read image {image_path}")
#             if image_type == 'masks':
#                 image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # grayscale mask
#         else:
#             # print(f"Loaded image {image.shape}")
#             image_dataset.append(image)       
            
#             # crop sizes to patch multiples
#             size_x = (image.shape[1] // image_patch_size) * image_patch_size
#             size_y = (image.shape[0] // image_patch_size) * image_patch_size
#             # print("\nThis image patch size:")
#             # print(f"The {image.shape} --- {size_x} - {size_y}")
#             # print('{}  --- {} -- {}'.format(image.shape, size_x, size_y))

#             # crop funtion
#             image = Image.fromarray(image)
#             image = image.crop((0,0, size_x, size_y))
#             # print("( {}, {} )".format(image.size[0], image.size[1]))

#             # patches image into Numpy array
#             image = np.array(image)
#             patched_images = patchify(image, (image_patch_size, image_patch_size, 3), step=image_patch_size)
#             # print(len(patched_images))

#             # ============= Slice ===============
#             for i in range(patched_images.shape[0]):
#                 for j in range(patched_images.shape[1]):
               
#                     if image_type == 'images':
#                         individual_patched_image = patched_images[i,j,:,:]
#                         # print("Before Scaling", individual_patched_image.shape)

#                         # Flatten to (pixels, channels)
#                         reshaped = individual_patched_image.reshape(-1, individual_patched_image.shape[-1])
#                         # Scale
#                         scaled = minmaxscaler.fit_transform(reshaped)
#                         # Reshape back to original image shape
#                         individual_patched_image = scaled.reshape(individual_patched_image.shape)
#                         # print("After scaling:", individual_patched_image.shape)
            
#                         image_dataset.append(individual_patched_image)
#                     elif image_type == "masks":
#                         individual_patched_mask = patched_images[i,j,:,:]
#                         individual_patched_mask = individual_patched_mask[0]
#                         mask_dataset.append(individual_patched_image)

                    

# print(f" The new image_dataset:", len(image_dataset))
# print(f" The new mask_dataset:", len(mask_dataset))

# image_dataset[0]
# print(image_dataset)
# mask_dataset[0]
# print(mask_dataset)