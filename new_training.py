import os
import argparse
import glob
import numpy as np
import cv2
import datetime
from patchify import patchify
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Import components from our modular engine backend
from src.data_utils import load_config, get_vectorized_mappings, rgb_to_label_batch
from src.unet_model import compile_production_model

def parse_args():
    parser = argparse.ArgumentParser(description="Production U-Net Training Pipeline Orchestrator")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size for GPU parallel processing")
    return parser.parse_args()

def load_and_preprocess_dataset(config):
    """
    Orchestrates ingestion by dynamically scanning directories.
    Eliminates hardcoded loop ranges for tile IDs and image indices.
    """
    dataset_root = config["dataset"]["root_path"]
    dataset_name = config["dataset"]["name"]
    patch_size = config["model"]["patch_size"]
    
    dataset_path = os.path.join(dataset_root, dataset_name)
    
    raw_images = []
    raw_masks = []
    
    # 🔍 DYNAMIC SCANNING: Discover all available geographic tile subfolders automatically
    tile_folders = glob.glob(os.path.join(dataset_path, "Tile *"))
    if not tile_folders:
        raise FileNotFoundError(f"No directories matching pattern 'Tile *' found inside: {dataset_path}")
        
    print(f" Found {len(tile_folders)} dataset tiles. Initiating dynamic parallel ingestion...")
    
    for tile_folder in tile_folders:
        # Match all images within the current tile context
        image_paths = glob.glob(os.path.join(tile_folder, "images", "*.jpg"))
        
        for img_path in image_paths:
            # Derive corresponding mask file dynamically by swapping string patterns
            filename = os.path.basename(img_path)
            mask_filename = filename.replace(".jpg", ".png")
            msk_path = os.path.join(tile_folder, "masks", mask_filename)
            
            if not os.path.exists(msk_path):
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            msk = cv2.imread(msk_path, cv2.IMREAD_COLOR)
            
            if img is None or msk is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            
            # Snap dimensions to perfect divisible patch multiples
            size_x = (img.shape[1] // patch_size) * patch_size
            size_y = (img.shape[0] // patch_size) * patch_size
            
            img = img[0:size_y, 0:size_x]
            msk = msk[0:size_y, 0:size_x]
            
            # Matrix-slice continuous tiles into targeted patch shapes
            patched_imgs = patchify(img, (patch_size, patch_size, 3), step=patch_size)
            patched_msks = patchify(msk, (patch_size, patch_size, 3), step=patch_size)
            
            for i in range(patched_imgs.shape[0]):
                for j in range(patched_imgs.shape[1]):
                    # Direct index parsing avoids loop overhead
                    img_patch = patched_imgs[i, j, 0].astype("float32") / 255.0
                    raw_images.append(img_patch)
                    raw_masks.append(patched_msks[i, j, 0])
                    
    return np.array(raw_images, dtype="float32"), np.array(raw_masks, dtype="uint8")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 1. Ingest and Compile Tensor Matrices
    raw_images, raw_masks = load_and_preprocess_dataset(config)
    print(f" Successfully parsed array block shape: {raw_images.shape}")
    
    # 2. Vectorized Color-to-Index Parsing
    target_colors, _ = get_vectorized_mappings(config)
    print(" Converting color profile representations to optimized class index maps...")
    labels = rgb_to_label_batch(raw_masks, target_colors)
    
    # 3. Structural One-Hot Categorization & Train/Val Splitting
    num_classes = config["model"]["num_classes"]
    labels_categorical = to_categorical(labels, num_classes=num_classes)
    
    x_train, x_val, y_train, y_val = train_test_split(
        raw_images, labels_categorical, test_size=0.15, random_state=100
    )
    
    # 4. Instantiate and Compile Keras Functional Graph
    input_shape = (config["model"]["patch_size"], config["model"]["patch_size"], config["model"]["channels"])
    print(f" Constructing and compiling U-Net model graphs on active GPU environment...")
    model = compile_production_model(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=float(config["model"].get("learning_rate", 1e-4))
    )
    
    # 5. Production Callback Architecture Setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_output_dir = os.path.dirname(config["model"]["weights_path"])
    os.makedirs(weights_output_dir, exist_ok=True)
    
    # Extract presentation and logging configurations dynamically
    log_cfg = config.get("logging", {})
    metrics_prefix = log_cfg.get("metrics_prefix", "training_metrics")
    chart_prefix = log_cfg.get("chart_prefix", "loss_convergence")
    dpi_val = log_cfg.get("chart_dpi", 150)
    fig_w = log_cfg.get("chart_width", 12)
    fig_h = log_cfg.get("chart_height", 5)
    
    # Dynamic log path aligned with your config values
    log_file = os.path.join(weights_output_dir, f"{metrics_prefix}_{timestamp}.csv")
    checkpoint_file = os.path.join(weights_output_dir, f"best_unet_model_v2_{timestamp}.keras")
    
    callbacks = [
        ModelCheckpoint(checkpoint_file, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1), 
        CSVLogger(log_file, separator=",", append=False, verbose=1)
    ]
    
    # 6. Kick Off Model Evolution Training Cycle (Capture History Object)
    print(f" Initializing learning optimization. Export target: {checkpoint_file}")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    print(" Pipeline optimization execution run successfully completed!")

    # 7. Auto-Export Convergence History Performance Charts (100% Decoupled)
    theme_colors = log_cfg.get("colors", {})
    color_train = theme_colors.get("primary", "#3C1098")
    color_val = theme_colors.get("secondary", "#FEDD3A")
    
    history_png = os.path.join(weights_output_dir, f"{chart_prefix}_{timestamp}.png")
    
    plt.figure(figsize=(fig_w, fig_h))
    
    # Subplot 1: Total focal+dice loss curve trajectories
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color=color_train, linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color=color_val, linewidth=2)
    plt.title('Loss Convergence Profiles')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Intersection-Over-Union / Jaccard Metric Tracking
    plt.subplot(1, 2, 2)
    plt.plot(history.history['jaccard_coef'], label='Training Jaccard', color=color_train, linewidth=2)
    plt.plot(history.history['val_jaccard_coef'], label='Validation Jaccard', color=color_val, linewidth=2)
    plt.title('Intersection over Union (IoU) Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Coefficient')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(history_png, dpi=dpi_val)
    print(f" Loss curves and IoU graphs successfully saved to: {history_png}")

if __name__ == "__main__":
    main()

# run
# python train.py --epochs 40 --batch_size 32




# ============================= ===================
# import glob
# import os
# import argparse
# import numpy as np
# import cv2
# from patchify import patchify
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
# import datetime

# # Import our production-grade modular system components
# from src.data_utils import load_config, get_vectorized_mappings, rgb_to_label_batch
# from src.unet_model import compile_production_model

# def parse_args():
#     parser = argparse.ArgumentParser(description="Production U-Net Training Pipeline Orchestrator")
#     parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration yaml")
#     parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
#     parser.add_argument("--batch_size", type=int, default=16, help="Training batch size for GPU processing")
#     return parser.parse_args()

# import glob

# def load_and_preprocess_dataset(config):
#     dataset_root = config["dataset"]["root_path"]
#     dataset_name = config["dataset"]["name"]
#     patch_size = config["model"]["patch_size"]
    
#     dataset_path = os.path.join(dataset_root, dataset_name)
    
#     raw_images = []
#     raw_masks = []
    
#     # 🔍 DYNAMIC SCANNING: Find every folder starting with "Tile " automatically
#     tile_folders = glob.glob(os.path.join(dataset_path, "Tile *"))
    
#     print(f"⏳ Found {len(tile_folders)} tiles. Extracting patches...")
#     for tile_folder in tile_folders:
#         # Scan all JPG images inside the current tile folder dynamically
#         image_search_path = os.path.join(tile_folder, "images", "*.jpg")
#         image_paths = glob.glob(image_search_path)
        
#         for img_path in image_paths:
#             # Derive the matching mask name based on the image filename
#             filename = os.path.basename(img_path)
#             mask_filename = filename.replace(".jpg", ".png")
#             msk_path = os.path.join(tile_folder, "masks", mask_filename)
            
#             if not os.path.exists(msk_path):
#                 continue
                
#             img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#             msk = cv2.imread(msk_path, cv2.IMREAD_COLOR)
            
#             if img is None or msk is None:
#                 continue
                
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            
#             # Continuous boundary cropping and patchify steps remain the same...
#             size_x = (img.shape[1] // patch_size) * patch_size
#             size_y = (img.shape[0] // patch_size) * patch_size
            
#             img = img[0:size_y, 0:size_x]
#             msk = msk[0:size_y, 0:size_x]
            
#             patched_imgs = patchify(img, (patch_size, patch_size, 3), step=patch_size)
#             patched_msks = patchify(msk, (patch_size, patch_size, 3), step=patch_size)
            
#             for i in range(patched_imgs.shape[0]):
#                 for j in range(patched_imgs.shape[1]):
#                     img_patch = patched_imgs[i, j, 0].astype("float32") / 255.0
#                     raw_images.append(img_patch)
#                     raw_masks.append(patched_msks[i, j, 0])
                    
#     return np.array(raw_images, dtype="float32"), np.array(raw_masks, dtype="uint8")

# def main():
#     args = parse_args()
#     config = load_config(args.config)
    
#     # 1. Pipeline Dataset Extraction
#     raw_images, raw_masks = load_and_preprocess_dataset(config)
#     print(f" Loaded contiguous image patch block array: {raw_images.shape}")
    
#     # 2. Convert Color Masks to Vectorized Integer Arrays
#     target_colors, _ = get_vectorized_mappings(config)
#     print(" Mapping color profiles to semantic classification indices...")
#     labels = rgb_to_label_batch(raw_masks, target_colors)
    
#     # 3. Categorical One-Hot Conversion & Stratified Splits
#     num_classes = config["model"]["num_classes"]
#     labels_categorical = to_categorical(
#         labels, 
#         num_classes=num_classes)
    
#     x_train, x_val, y_train, y_val = train_test_split(
#         raw_images, 
#         labels_categorical, 
#         test_size=0.15, 
#         random_state=100
#     )
    
#     # 4. Initialize and Compile U-Net Graph Model
#     input_shape = (config["model"]["patch_size"], config["model"]["patch_size"], config["model"]["channels"])
#     print(f" Compiling U-Net graph on active GPU context...")
#     model = compile_production_model(
#         input_shape=input_shape,
#         num_classes=num_classes,
#         learning_rate=float(config["model"].get("learning_rate", 1e-4))
#     )
    
#     # 5. Production Optimization Callbacks Configuration
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     weights_output_dir = os.path.dirname(config["model"]["weights_path"])
#     csv_filename = os.path.dirname(config["csv"]["weights_path"])
#     os.makedirs(weights_output_dir, exist_ok=True)
    
#     checkpoint_file = os.path.join(weights_output_dir, f"best_unet_model_v1_{timestamp}.keras")
    
#     callbacks = [
#         # Automatically save only the best weights based on validation loss performance
#         ModelCheckpoint(
#             checkpoint_file,
#             monitor="val_loss", 
#             save_best_only=True, 
#             verbose=1
#             ),
#         # Stop training if the validation loss stalls for 7 epochs to prevent overfitting
#         EarlyStopping(
#             monitor="val_loss", 
#             patience=7, 
#             restore_best_weights=True, 
#             verbose=1),
#         # Automatically drop learning rate if validation loss stalls to fine-tune features
#         ReduceLROnPlateau(
#             monitor="val_loss", 
#             factor=0.2, 
#             patience=3, 
#             min_lr=1e-6, 
#             verbose=1),
#             CSVLogger(csv_filename, append=False)
#     ]
    
#     # 6. Execute Training Engine Optimization Loop
#     print(f" Starting U-Net execution run. Exporting weights to: {checkpoint_file}")
#     history = model.fit(
#         x_train, y_train,
#         validation_data=(x_val, y_val),
#         epochs=args.epochs,
#         batch_size=args.batch_size,
#         callbacks=callbacks,
#         verbose=1
#     )
#     print(" Training pipeline processing complete!")

# if __name__ == "__main__":
#     main()



# run
# python train.py --epochs 30 --batch_size 32






# ==================================   =========================
# import os
# import argparse
# import random
# import cv2
# import numpy as np
# from PIL import Image
# from patchify import patchify
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical

# def main():
#     # 1. Dynamic CLI Arguments (Removes Hardcoding)
#     parser = argparse.ArgumentParser(description="Production U-Net Remote Sensing Preprocessing Engine")
#     parser.add_argument("--root", type=str, default=r"C:\samcodebase\deeplearning\semantic-segmentation", help="Dataset root directory")
#     parser.add_argument("--dataset", type=str, default="Semantic_segmentation_dataset", help="Folder container name")
#     parser.add_argument("--patch_size", type=int, default=256, help="Target patch dimensions")
#     args = parser.parse_args()

#     dataset_path = os.path.join(args.root, args.dataset)
#     image_patch_size = args.patch_size

#     image_dataset = []
#     mask_dataset = []

#     # 2. Vectorized Training Target Class Configuration (With High Contrast Mapping)
#     target_colors = {
#         0: [226, 169, 41],   # Water -> Ochre / Muted Amber (#E2A929)
#         1: [132, 41, 246],  # Land -> Electric Purple / Vivid Violet (#8429F6)
#         2: [110, 193, 228],  # Road -> Sky Blue / Soft Cyan (#6EC1E4)
#         3: [60, 16, 152],    # Building -> Deep Indigo / Imperial Blue (#3C1098)
#         4: [254, 221, 58],   # Vegetation -> Canary Yellow / Cyber Yellow (#FEDD3A)
#         5: [155, 155, 155]   # Unlabeled -> Medium Gray / Battleship Gray (#9B9B9B)
#     }

#     # 3. Data Ingestion & Patch Generation Loop
#     print("⏳ Processing tiles and extracting patches...")
#     for tile_id in range(1, 9):  # Tiles 1–8
#         for image_id in range(1, 10):
#             # Safe cross-platform paths
#             image_path = os.path.join(dataset_path, f"Tile {tile_id}", "images", f"image_part_{image_id:03d}.jpg")
#             mask_path = os.path.join(dataset_path, f"Tile {tile_id}", "masks", f"image_part_{image_id:03d}.png")

#             if not os.path.exists(image_path) or not os.path.exists(mask_path):
#                 continue

#             # Load images securely
#             img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#             mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

#             if img is None or mask is None:
#                 continue

#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

#             # Crop dimensions to match patch sizes cleanly
#             size_x = (img.shape[1] // image_patch_size) * image_patch_size
#             size_y = (img.shape[0] // image_patch_size) * image_patch_size

#             img = img[0:size_y, 0:size_x]
#             mask = mask[0:size_y, 0:size_x]

#             # Patchify execution
#             patched_imgs = patchify(img, (image_patch_size, image_patch_size, 3), step=image_patch_size)
#             patched_msks = patchify(mask, (image_patch_size, image_patch_size, 3), step=image_patch_size)

#             for i in range(patched_imgs.shape[0]):
#                 for j in range(patched_imgs.shape[1]):
#                     # Extract patch structure
#                     img_patch = patched_imgs[i, j, 0]
#                     mask_patch = patched_msks[i, j, 0]

#                     # 🔍 PERFORMANCE OPTIMIZATION: Faster float division instead of loop scaling
#                     img_patch = img_patch.astype("float32") / 255.0

#                     image_dataset.append(img_patch)
#                     mask_dataset.append(mask_patch)

#     image_dataset = np.array(image_dataset, dtype="float32")
#     mask_dataset = np.array(mask_dataset, dtype="uint8")

#     print(f"✅ Total Extracted Patches: {image_dataset.shape[0]}")

#     # 4. Optimized Multi-Class Label Encoding
#     def rgb_to_label(label_rgb):
#         label_segment = np.zeros(label_rgb.shape[:2], dtype=np.uint8)
#         for class_idx, rgb_val in target_colors.items():
#             match_mask = np.all(label_rgb == rgb_val, axis=-1)
#             label_segment[match_mask] = class_idx
#         return label_segment

#     print("⚡ Converting color masks to target training integers...")
#     labels = np.array([rgb_to_label(m) for m in mask_dataset])
#     labels = np.expand_dims(labels, axis=-1)

#     # 5. Categorical One-Hot Conversion & Train-Test Split
#     total_classes = len(target_colors)
#     labels_categorical = to_categorical(labels, num_classes=total_classes)

#     x_train, x_test, y_train, y_test = train_test_split(
#         image_dataset, labels_categorical, test_size=0.15, random_state=100
#     )

#     print("\n" + "="*40 + "\n📊 MASTER PRODUCTION MATRIX REPORT\n" + "="*40)
#     print(f"Train Images Shape:  {x_train.shape} (Ready for model.fit)")
#     print(f"Train Labels Shape:  {y_train.shape}")
#     print(f"Test Images Shape:   {x_test.shape}")
#     print(f"Test Labels Shape:   {y_test.shape}")
#     print(f"Input Matrix Dimensions: {x_train.shape[1]}x{x_train.shape[2]} with {x_train.shape[3]} channels")
#     print(f"Total Architecture Classification Targets: {total_classes}")

# if __name__ == "__main__":
#     main()


# run
# python train.py --epochs 30 --batch_size 32






# ============================= OLD VERSION ========================
# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, Conv2DTranspose
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
# from matplotlib import pyplot as plt
# import datetime

# # Importing data and configurations
# from satellite_data import total_classes, image_height, image_width, image_channels, x_test, x_train, y_test, y_train

# # Generate timestamp strings
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# csv_filename = f"training_history_{timestamp}.csv"
# png_filename = f"training_history_{timestamp}.png"
# model_filename = f"best_unet_satellite_model_{timestamp}.keras"


# # ============ Evaluation Metrics & Loss Functions ============= 
# def jaccard_coef(y_true, y_pred):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)
#     intersection = K.sum(y_true_flatten * y_pred_flatten)
#     final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
#     return final_coef_value

# def dice_loss(y_true, y_pred, smooth=1e-6):
#     y_true_f = tf.reshape(y_true, [-1])
#     y_pred_f = tf.reshape(y_pred, [-1])
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# focal_loss = tf.keras.losses.CategoricalFocalCrossentropy()

# def total_loss(y_true, y_pred):
#     return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)


# # ============ U-Net Model Definition ============= 
# def multi_unet_model(n_classes=5, image_height=256, image_width=256, image_channels=1):
#     inputs = Input((image_height, image_width, image_channels))
    
#     # Encoder (Downsampling)
#     c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
#     c1 = Dropout(0.2)(c1)
#     c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
#     p1 = MaxPooling2D((2,2))(c1)

#     c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
#     c2 = Dropout(0.2)(c2)
#     c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
#     p2 = MaxPooling2D((2,2))(c2)

#     c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
#     c3 = Dropout(0.2)(c3)
#     c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
#     p3 = MaxPooling2D((2,2))(c3)

#     c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
#     c4 = Dropout(0.2)(c4)
#     c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
#     p4 = MaxPooling2D((2,2))(c4)

#     # Bottleneck
#     c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
#     c5 = Dropout(0.2)(c5)
#     c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

#     # Decoder (Upsampling)
#     u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
#     c6 = Dropout(0.2)(c6)
#     c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

#     u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
#     c7 = Dropout(0.2)(c7)
#     c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

#     u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
#     c8 = Dropout(0.2)(c8)
#     c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

#     u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
#     u9 = concatenate([u9, c1], axis=3)
#     c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
#     c9 = Dropout(0.2)(c9)
#     c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

#     outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)

#     model = Model(inputs=[inputs], outputs=[outputs])
#     return model


# # ======================= Initialization & Compilation =======================
# tf.keras.backend.clear_session()

# model = multi_unet_model(n_classes=total_classes, 
#                          image_height=image_height, 
#                          image_width=image_width, 
#                          image_channels=image_channels)

# # model.compile(
# #     # optimizer='adam', 
# #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
# #     # loss=total_loss,
# #     loss='sparse_categorical_crossentropy', 
# #     metrics=metrics
# #     )

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#     loss=total_loss,  
#     metrics=['accuracy', jaccard_coef]  
# )

# model.summary()

# # ============= Model Training =================
# custom_callbacks = [
#     EarlyStopping(
#         monitor="val_loss", 
#         patience=5, 
#         restore_best_weights=True
#         ),
#     ModelCheckpoint(
#         model_filename, 
#         monitor="val_loss", 
#         save_best_only=True, 
#         verbose=1
#         ),
#     CSVLogger(csv_filename, append=False)
# ]

# model_history = model.fit(
#     x_train, y_train,
#     batch_size=10,
#     verbose=1,
#     epochs=100,
#     validation_data=(x_test, y_test),
#     shuffle=True,
#     callbacks=custom_callbacks 
# )

# # Load the saved model with custom metrics/losses mapped properly
# model = load_model(model_filename, 
#                    custom_objects={"dice_loss": dice_loss,
#                                    "total_loss": total_loss,
#                                    "jaccard_coef": jaccard_coef})


# # ========== History Loss Visualization ==================================
# def plot_history(history_obj, save_png_name):
#     loss = history_obj.history['loss']          
#     val_loss = history_obj.history['val_loss']  
#     jaccard_coef_values = history_obj.history['jaccard_coef']
#     val_jaccard_coef_values = history_obj.history['val_jaccard_coef']
#     epochs = range(1, len(loss) + 1)

#     plt.figure(figsize=(12, 4))

#     # Loss graph
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, loss, 'y', label='Training Loss')
#     plt.plot(epochs, val_loss, 'r', label='Validation Loss') 
#     plt.title("Training Vs Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()

#     # Jaccard IoU graph
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, jaccard_coef_values, 'y', label="Training IoU")
#     plt.plot(epochs, val_jaccard_coef_values, 'r', label="Validation IoU")
#     plt.title("Training Vs Validation IoU")
#     plt.xlabel("Epochs")
#     plt.ylabel("IoU")
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig(save_png_name) # Saves file smoothly before showing
#     plt.show()

# # Call after training
# plot_history(model_history, png_filename)