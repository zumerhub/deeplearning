"""
version_3
U-Net Segmentation with Streaming Data Generator (No OOM Crashes)
Multi-Class Configuration for Prodramp Satellite Dataset Workflow
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================================
# CONFIGURATION - MULTI-CLASS OPTIMIZED
# ==========================================
PATCH_SIZE = 256
STRIDE = 128
MIN_VALID_PIXELS = 0.1
BATCH_SIZE = 8  # CPU 4
EPOCHS = 30  # CPU 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 6  # 6 distinct land cover classes from the course

# Max patches to generate per image (to prevent memory explosion)
MAX_PATCHES_PER_IMAGE = 20  # CPU 10

# EXACT local path adjustments
dataset_root_folder = r"C:\samcodebase\deeplearning\semantic-segmentation"
dataset_name = "Semantic_segmentation_datasets"
# full_datasets_path = os.path.join(dataset_root_folder, dataset_name)


full_datasets_path = r"C:\samcodebase\deeplearning\semantic-segmentation\Semantic_segmentation_dataset"


# ==========================================
# HELPER: MULTI-CLASS COLOR MAPPER
# ==========================================
def rgb_to_landscape_labels(mask_rgb):
    """
    Explicitly maps 3-channel RGB colors from the Dubai dataset 
    to continuous sparse integer indices (0 to 5).
    Expects BGR array format from OpenCV input.
    """
    label_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    
    # Class color arrays structured in OpenCV BGR format: [Blue, Green, Red]
    building   = [152, 16, 60]
    land       = [124, 41, 132]
    road       = [196, 196, 196]
    vegetation = [35, 81, 20]
    water      = [255, 243, 9]
    
    # Compute boolean masks and assign corresponding category integer IDs
    label_mask[np.all(mask_rgb == building, axis=-1)] = 1
    label_mask[np.all(mask_rgb == land, axis=-1)] = 2
    label_mask[np.all(mask_rgb == road, axis=-1)] = 3
    label_mask[np.all(mask_rgb == vegetation, axis=-1)] = 4
    label_mask[np.all(mask_rgb == water, axis=-1)] = 5
    # Unlabeled / Background defaults cleanly to 0
    
    return label_mask


# ==========================================
# 1. SCAN DATASET & BUILD FILE MANIFEST
# ==========================================
def build_file_manifest(dataset_path):
    manifest = []
    for path, subdirs, files in os.walk(dataset_path):
        dir_name = os.path.basename(path)
        if dir_name == 'images':
            tile_folder = os.path.dirname(path)
            mask_folder = os.path.join(tile_folder, 'masks')
            if not os.path.exists(mask_folder):
                mask_folder = os.path.join(tile_folder, 'masks_new')
            
            for file in sorted(files):
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                
                image_path = os.path.join(path, file)
                mask_file = file.replace(".jpg", ".png").replace(".jpeg", ".png")
                mask_path = os.path.join(mask_folder, mask_file)
                
                if os.path.exists(mask_path):
                    manifest.append((image_path, mask_path))
    return manifest


# ==========================================
# 2. STREAMING PATCH GENERATOR (INTEGRATED)
# ==========================================
def patch_generator(manifest, patch_size=256, stride=128, max_per_image=20):
    patch_count = 0
    
    for image_path, mask_path in manifest:
        try:
            img = cv2.imread(image_path)
            mask_raw_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            
            if img is None or mask_raw_bgr is None:
                continue
            
            # Normalize image inputs to [0.0, 1.0]
            img = img.astype(np.float32) / 255.0       
            
            # FIXED: Map raw RGB masks to categorical sparse integers [0 to 5]
            mask = rgb_to_landscape_labels(mask_raw_bgr)
            mask = mask.astype(np.float32)             
            
            h, w = img.shape[:2]
            patches_this_image = 0
            
            for y in range(0, h - patch_size + 1, stride):
                if patches_this_image >= max_per_image:
                    break
                    
                for x in range(0, w - patch_size + 1, stride):
                    if patches_this_image >= max_per_image:
                        break
                    
                    img_crop = img[y:y+patch_size, x:x+patch_size]
                    mask_crop = mask[y:y+patch_size, x:x+patch_size]
                    
                    # Filter empty background patches
                    gray = cv2.cvtColor((img_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    non_zero = np.count_nonzero(gray) / (patch_size * patch_size)
                    
                    if non_zero >= MIN_VALID_PIXELS:
                        # Yield 2D mask matching target signature (No extra channel dimension)
                        yield img_crop, mask_crop
                        
                        patches_this_image += 1
                        patch_count += 1
                        
                        if patch_count % 100 == 0:
                            print(f"  📦 Processed {patch_count} patches...")
            
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
            continue


# ==========================================
# 3. CREATE TF.DATA PIPELINE FROM GENERATOR
# ==========================================
def create_streaming_dataset(manifest, batch_size=8):
    dataset = tf.data.Dataset.from_generator(
        lambda: patch_generator(manifest),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256), dtype=tf.float32)  
        )
    )
    
    dataset = dataset.repeat()
    
    dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# ==========================================
# 4. MULTI-CLASS U-NET ARCHITECTURE
# ==========================================
def double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def encoder_block(x, n_filters):
    skip = double_conv_block(x, n_filters)
    pool = layers.MaxPool2D(2)(skip)
    return skip, pool

def decoder_block(x, skip, n_filters):
    x = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = double_conv_block(x, n_filters)
    return x

def build_unet(input_shape=(256, 256, 3), num_classes=6):
    inputs = layers.Input(input_shape)
    
    # Encoder
    skip1, pool1 = encoder_block(inputs, 64)
    skip2, pool2 = encoder_block(pool1, 128)
    skip3, pool3 = encoder_block(pool2, 256)
    skip4, pool4 = encoder_block(pool3, 512)
    
    # Bridge
    bridge = double_conv_block(pool4, 1024)
    
    # Decoder
    decoder1 = decoder_block(bridge, skip4, 512)
    decoder2 = decoder_block(decoder1, skip3, 256)
    decoder3 = decoder_block(decoder2, skip2, 128)
    decoder4 = decoder_block(decoder3, skip1, 64)
    
    # Output layer uses Softmax activation with 6 distinct channel layers
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(decoder4)
    
    return models.Model(inputs, outputs, name="U-Net_MultiClass_Streaming")


# ==========================================
# 5. TRAINING WITH STREAMING DATA
# ==========================================
def train_streaming_unet(manifest, manifest_val=None):
    print("\n🏗️  Building Multi-Class U-Net Architecture...")
    unet_model = build_unet(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)
    
    unet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint('Trained_Models/unet_satellite_streaming_v1.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("\n⛓️  Creating streaming training dataset...")
    train_dataset = create_streaming_dataset(manifest, batch_size=BATCH_SIZE)
    
    if manifest_val is None:
        manifest_val = manifest
    
    print("⛓️  Creating streaming validation dataset...")
    val_dataset = create_streaming_dataset(manifest_val, batch_size=BATCH_SIZE)
    
    print("\n🚀 Starting Training (Streaming Mode)...")
    history = unet_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=  100,  # CPU 40, GPU 100
        validation_steps=20,    # CPU 10, GPU 20
        callbacks=callbacks,
        verbose=1
    )
    
    return unet_model, history


# ==========================================
# 6. METRICS & MONITORING UTILITIES
# ==========================================
def print_memory_info():
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"\n💾 Memory Usage: {mem_info.rss / 1024**2:.1f} MB")

def print_training_summary(history):
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"✅ Training Complete!")
    print(f"   Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"   Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print("="*70)


# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-CLASS U-NET STREAMING PIPELINE")
    print("=" * 70)
    
    os.makedirs('Trained_Models', exist_ok=True)
    
    try:
        print("\n🔍 Building dataset manifest...")
        manifest = build_file_manifest(full_datasets_path)
        print(f"✅ Found {len(manifest)} image-mask pairs")
        
        if len(manifest) == 0:
            print("❌ No image-mask pairs found! Check dataset path.")
            exit(1)
        
        split_idx = int(len(manifest) * 0.8)
        manifest_train = manifest[:split_idx]
        manifest_val = manifest[split_idx:]
        
        # Run the training loop with integrated mapping
        unet_model, history = train_streaming_unet(manifest_train, manifest_val)
        
        unet_model.save('Trained_Models/unet_satellite_streaming_v1.keras')
        print("\n✅ Model securely saved inside 'Trained_Models/unet_satellite_streaming_v1.keras'")
        
        print_training_summary(history)
        print_memory_info()
        
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")





# """
# version_2
# U-Net Segmentation with Streaming Data Generator (No OOM Crashes)
# Multi-Class Configuration for Prodramp Satellite Dataset Workflow
# """

# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # ==========================================
# # CONFIGURATION - MULTI-CLASS OPTIMIZED
# # ==========================================
# PATCH_SIZE = 256
# STRIDE = 128
# MIN_VALID_PIXELS = 0.1
# BATCH_SIZE = 8  
# EPOCHS = 30
# LEARNING_RATE = 1e-4
# NUM_CLASSES = 6  # 6 distinct land cover classes from the course

# # Max patches to generate per image (to prevent memory explosion)
# MAX_PATCHES_PER_IMAGE = 20

# # Dataset paths (MODIFY FOR YOUR SETUP)
# dataset_root_folder = "/content/drive/MyDrive/Colab Notebooks/semantic-segmentation/"
# dataset_name = "Semantic_segmentation_dataset"
# full_datasets_path = os.path.join(dataset_root_folder, dataset_name)

# # ==========================================
# # 1. SCAN DATASET & BUILD FILE MANIFEST
# # ==========================================
# def build_file_manifest(dataset_path):
#     manifest = []
#     for path, subdirs, files in os.walk(dataset_path):
#         dir_name = os.path.basename(path)
#         if dir_name == 'images':
#             tile_folder = os.path.dirname(path)
#             mask_folder = os.path.join(tile_folder, 'masks')
#             if not os.path.exists(mask_folder):
#                 mask_folder = os.path.join(tile_folder, 'masks_new')
            
#             for file in sorted(files):
#                 if not file.lower().endswith((".png", ".jpg", ".jpeg")):
#                     continue
                
#                 image_path = os.path.join(path, file)
#                 mask_file = file.replace(".jpg", ".png").replace(".jpeg", ".png")
#                 mask_path = os.path.join(mask_folder, mask_file)
                
#                 if os.path.exists(mask_path):
#                     manifest.append((image_path, mask_path))
#     return manifest


# # ==========================================
# # 2. STREAMING PATCH GENERATOR (MULTI-CLASS)
# # ==========================================
# def patch_generator(manifest, patch_size=256, stride=128, max_per_image=20):
#     patch_count = 0
    
#     for image_path, mask_path in manifest:
#         try:
#             img = cv2.imread(image_path)
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
#             if img is None or mask is None:
#                 continue
            
#             # Normalize image inputs to [0.0, 1.0]
#             img = img.astype(np.float32) / 255.0       
#             # Keep original multi-class integer indices [0, 1, 2, 3, 4, 5]
#             mask = mask.astype(np.float32)             
            
#             h, w = img.shape[:2]
#             patches_this_image = 0
            
#             for y in range(0, h - patch_size + 1, stride):
#                 if patches_this_image >= max_per_image:
#                     break
                    
#                 for x in range(0, w - patch_size + 1, stride):
#                     if patches_this_image >= max_per_image:
#                         break
                    
#                     img_crop = img[y:y+patch_size, x:x+patch_size]
#                     mask_crop = mask[y:y+patch_size, x:x+patch_size]
                    
#                     # Filter empty background patches
#                     gray = cv2.cvtColor((img_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
#                     non_zero = np.count_nonzero(gray) / (patch_size * patch_size)
                    
#                     if non_zero >= MIN_VALID_PIXELS:
#                         # CRITICAL: Yield 2D mask (No channel dimension!) to match target signature
#                         yield img_crop, mask_crop
                        
#                         patches_this_image += 1
#                         patch_count += 1
                        
#                         if patch_count % 100 == 0:
#                             print(f"  📦 Processed {patch_count} patches...")
            
#         except Exception as e:
#             print(f"❌ Error loading {image_path}: {e}")
#             continue


# # ==========================================
# # 3. CREATE TF.DATA PIPELINE FROM GENERATOR
# # ==========================================
# def create_streaming_dataset(manifest, batch_size=8):
#     dataset = tf.data.Dataset.from_generator(
#         lambda: patch_generator(manifest),
#         output_signature=(
#             tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
#             tf.TensorSpec(shape=(256, 256), dtype=tf.float32)  # Shape: (256, 256) matches target.ndim - 1
#         )
#     )
    
#     dataset = dataset.shuffle(buffer_size=256)
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#     return dataset


# # ==========================================
# # 4. MULTI-CLASS U-NET ARCHITECTURE
# # ==========================================
# def double_conv_block(x, n_filters):
#     x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
#     x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
#     return x

# def encoder_block(x, n_filters):
#     skip = double_conv_block(x, n_filters)
#     pool = layers.MaxPool2D(2)(skip)
#     return skip, pool

# def decoder_block(x, skip, n_filters):
#     x = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same")(x)
#     x = layers.Concatenate()([x, skip])
#     x = double_conv_block(x, n_filters)
#     return x

# def build_unet(input_shape=(256, 256, 3), num_classes=6):
#     inputs = layers.Input(input_shape)
    
#     # Encoder
#     skip1, pool1 = encoder_block(inputs, 64)
#     skip2, pool2 = encoder_block(pool1, 128)
#     skip3, pool3 = encoder_block(pool2, 256)
#     skip4, pool4 = encoder_block(pool3, 512)
    
#     # Bridge
#     bridge = double_conv_block(pool4, 1024)
    
#     # Decoder
#     decoder1 = decoder_block(bridge, skip4, 512)
#     decoder2 = decoder_block(decoder1, skip3, 256)
#     decoder3 = decoder_block(decoder2, skip2, 128)
#     decoder4 = decoder_block(decoder3, skip1, 64)
    
#     # CRITICAL: Output layer must use Softmax activation with 6 distinct channel layers
#     outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(decoder4)
    
#     return models.Model(inputs, outputs, name="U-Net_MultiClass_Streaming")


# # ==========================================
# # 5. TRAINING WITH STREAMING DATA
# # ==========================================
# def train_streaming_unet(manifest, manifest_val=None):
#     print("\n🏗️  Building Multi-Class U-Net Architecture...")
#     unet_model = build_unet(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)
    
#     # Compile with Sparse Categorical Crossentropy
#     unet_model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     callbacks = [
#         EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
#         ModelCheckpoint('Trained_Models/unet_satellite_streaming_v1.keras', monitor='val_loss', save_best_only=True, verbose=1)
#     ]
    
#     print("\n⛓️  Creating streaming training dataset...")
#     train_dataset = create_streaming_dataset(manifest, batch_size=BATCH_SIZE)
    
#     if manifest_val is None:
#         manifest_val = manifest
    
#     print("⛓️  Creating streaming validation dataset...")
#     val_dataset = create_streaming_dataset(manifest_val, batch_size=BATCH_SIZE)
    
#     print("\n🚀 Starting Training (Streaming Mode)...")
#     history = unet_model.fit(
#         train_dataset,
#         validation_data=val_dataset,
#         epochs=EPOCHS,
#         steps_per_epoch=100,  
#         validation_steps=20,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     return unet_model, history

# # ==========================================
# # 6. METRICS & MONITORING UTILITIES
# # ==========================================
# def print_memory_info():
#     import psutil
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     print(f"\n💾 Memory Usage: {mem_info.rss / 1024**2:.1f} MB")

# def print_training_summary(history):
#     print("\n" + "="*70)
#     print("TRAINING SUMMARY")
#     print("="*70)
#     print(f"✅ Training Complete!")
#     print(f"   Final Training Loss: {history.history['loss'][-1]:.4f}")
#     print(f"   Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
#     print("="*70)

# # ==========================================
# # 7. MAIN EXECUTION
# # ==========================================
# if __name__ == "__main__":
#     print("=" * 70)
#     print("MULTI-CLASS U-NET STREAMING PIPELINE")
#     print("=" * 70)
    
#     os.makedirs('Trained_Models', exist_ok=True)
    
#     try:
#         print("\n🔍 Building dataset manifest...")
#         manifest = build_file_manifest(full_datasets_path)
#         print(f"✅ Found {len(manifest)} image-mask pairs")
        
#         if len(manifest) == 0:
#             print("❌ No image-mask pairs found! Check dataset path.")
#             exit(1)
        
#         split_idx = int(len(manifest) * 0.8)
#         manifest_train = manifest[:split_idx]
#         manifest_val = manifest[split_idx:]
        
#         # Train with multi-class configuration
#         unet_model, history = train_streaming_unet(manifest_train, manifest_val)
        
#         unet_model.save('Trained_Models/unet_satellite_streaming_v1.keras')
#         print("\n✅ Model securely saved inside 'Trained_Models/unet_satellite_streaming_v1.keras'")
        
#         print_training_summary(history)
#         print_memory_info()
        
#     except Exception as e:
#         print(f"\n❌ Execution Error: {e}")







# """
# version_1
# U-Net Segmentation with Streaming Data Generator (No OOM Crashes)
# Loads patches on-the-fly instead of entire dataset into RAM
# """

# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # ==========================================
# # CONFIGURATION - MEMORY OPTIMIZED
# # ==========================================
# PATCH_SIZE = 256
# STRIDE = 128
# MIN_VALID_PIXELS = 0.1
# BATCH_SIZE = 8  # REDUCED: Was 16, now 8 for safety
# EPOCHS = 30
# LEARNING_RATE = 1e-4

# # Max patches to generate per image (to prevent memory explosion)
# MAX_PATCHES_PER_IMAGE = 20

# # Dataset paths (MODIFY FOR YOUR SETUP)
# dataset_root_folder = "/content/drive/MyDrive/Colab Notebooks/semantic-segmentation/"
# dataset_name = "Semantic_segmentation_dataset"
# full_datasets_path = os.path.join(dataset_root_folder, dataset_name)

# # ==========================================
# # 1. SCAN DATASET & BUILD FILE MANIFEST
# # ==========================================
# def build_file_manifest(dataset_path):
#     """
#     Walk dataset ONCE and return list of (image_path, mask_path) tuples.
#     Does NOT load images yet — just builds the index.
#     """
#     manifest = []
    
#     for path, subdirs, files in os.walk(dataset_path):
#         dir_name = os.path.basename(path)
        
#         if dir_name == 'images':
#             tile_folder = os.path.dirname(path)
#             mask_folder = os.path.join(tile_folder, 'masks')
#             if not os.path.exists(mask_folder):
#                 mask_folder = os.path.join(tile_folder, 'masks_new')
            
#             for file in sorted(files):
#                 if not file.lower().endswith((".png", ".jpg", ".jpeg")):
#                     continue
                
#                 image_path = os.path.join(path, file)
#                 mask_file = file.replace(".jpg", ".png").replace(".jpeg", ".png")
#                 mask_path = os.path.join(mask_folder, mask_file)
                
#                 if os.path.exists(mask_path):
#                     manifest.append((image_path, mask_path))
    
#     return manifest


# # ==========================================
# # 2. STREAMING PATCH GENERATOR
# # ==========================================
# def patch_generator(manifest, patch_size=256, stride=128, max_per_image=20):
#     """
#     Generator function: yields one patch at a time.
#     NEVER loads entire dataset into RAM — processes sequentially.
#     """
#     patch_count = 0
    
#     for image_path, mask_path in manifest:
#         try:
#             img = cv2.imread(image_path)
#             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
#             if img is None or mask is None:
#                 print(f"⚠️  Skipped: {os.path.basename(image_path)}")
#                 continue
            
#             # Normalize image
#             img = img.astype(np.float32) / 255.0
#             mask = mask.astype(np.float32)
            
#             h, w = img.shape[:2]
#             patches_this_image = 0
            
#             # Slide window
#             for y in range(0, h - patch_size + 1, stride):
#                 if patches_this_image >= max_per_image:
#                     break
                    
#                 for x in range(0, w - patch_size + 1, stride):
#                     if patches_this_image >= max_per_image:
#                         break
                    
#                     img_crop = img[y:y+patch_size, x:x+patch_size]
#                     mask_crop = mask[y:y+patch_size, x:x+patch_size]
                    
#                     # Filter empty patches
#                     gray = cv2.cvtColor((img_crop * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
#                     non_zero = np.count_nonzero(gray) / (patch_size * patch_size)
                    
#                     if non_zero >= MIN_VALID_PIXELS:
#                         # Add channel dimension for mask
#                         mask_crop = np.expand_dims(mask_crop, axis=-1)
                        
#                         yield img_crop, mask_crop
#                         patches_this_image += 1
#                         patch_count += 1
                        
#                         # Status every 100 patches
#                         if patch_count % 100 == 0:
#                             print(f"  📦 Processed {patch_count} patches...")
            
#         except Exception as e:
#             print(f"❌ Error loading {image_path}: {e}")
#             continue


# # ==========================================
# # 3. CREATE TF.DATA PIPELINE FROM GENERATOR
# # ==========================================
# def create_streaming_dataset(manifest, batch_size=8):
#     """
#     Create tf.data.Dataset from generator — patches loaded on-the-fly.
#     """
#     dataset = tf.data.Dataset.from_generator(
#         lambda: patch_generator(manifest),
#         output_signature=(
#             tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
#             tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32)
#         )
#     )
    
#     # Shuffle a small buffer (can't shuffle entire infinite stream)
#     dataset = dataset.shuffle(buffer_size=256)
    
#     # Batch and prefetch
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
#     return dataset


# # ==========================================
# # 4. U-NET ARCHITECTURE (Same as before)
# # ==========================================
# def double_conv_block(x, n_filters):
#     """Two consecutive Conv2D blocks."""
#     x = layers.Conv2D(
#         n_filters, 3, padding="same",
#         activation="relu", kernel_initializer="he_normal"
#     )(x)
#     x = layers.Conv2D(
#         n_filters, 3, padding="same",
#         activation="relu", kernel_initializer="he_normal"
#     )(x)
#     return x


# def encoder_block(x, n_filters):
#     """Encoder: convolution + max pooling."""
#     skip = double_conv_block(x, n_filters)
#     pool = layers.MaxPool2D(2)(skip)
#     return skip, pool


# def decoder_block(x, skip, n_filters):
#     """Decoder: transpose convolution + concatenation + convolution."""
#     x = layers.Conv2DTranspose(n_filters, 2, strides=2, padding="same")(x)
#     x = layers.Concatenate()([x, skip])
#     x = double_conv_block(x, n_filters)
#     return x


# def build_unet(input_shape=(256, 256, 3), num_classes=1):
#     """Build U-Net model."""
#     inputs = layers.Input(input_shape)
    
#     # Encoder
#     skip1, pool1 = encoder_block(inputs, 64)
#     skip2, pool2 = encoder_block(pool1, 128)
#     skip3, pool3 = encoder_block(pool2, 256)
#     skip4, pool4 = encoder_block(pool3, 512)
    
#     # Bridge
#     bridge = double_conv_block(pool4, 1024)
    
#     # Decoder
#     decoder1 = decoder_block(bridge, skip4, 512)
#     decoder2 = decoder_block(decoder1, skip3, 256)
#     decoder3 = decoder_block(decoder2, skip2, 128)
#     decoder4 = decoder_block(decoder3, skip1, 64)
    
#     # Output
#     outputs = layers.Conv2D(
#         num_classes, 1, padding="same", activation="sigmoid"
#     )(decoder4)
    
#     return models.Model(inputs, outputs, name="U-Net_Streaming")


# # ==========================================
# # 5. TRAINING WITH STREAMING DATA
# # ==========================================
# def train_streaming_unet(manifest, manifest_val=None):
#     """Train U-Net with streaming data generator."""
    
#     # Build model
#     print("\n🏗️  Building U-Net Architecture...")
#     unet_model = build_unet(input_shape=(256, 256, 3), num_classes=1)
    
#     # Compile
#     unet_model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
    
#     print(unet_model.summary())
    
#     # Callbacks
#     callbacks = [
#         EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True,
#             verbose=1
#         ),
#         ModelCheckpoint(
#             'best_unet_satellite_model.keras',
#             monitor='val_loss',
#             save_best_only=True,
#             verbose=1
#         )
#     ]
    
#     # Create streaming training dataset
#     print("\n⛓️  Creating streaming training dataset...")
#     train_dataset = create_streaming_dataset(manifest, batch_size=BATCH_SIZE)
    
#     # Create validation dataset (or use same manifest for simplicity)
#     if manifest_val is None:
#         # Use same manifest but stop after a certain number of steps
#         manifest_val = manifest
    
#     print("⛓️  Creating streaming validation dataset...")
#     val_dataset = create_streaming_dataset(manifest_val, batch_size=BATCH_SIZE)
    
#     # Train
#     print("\n🚀 Starting Training (Streaming Mode)...")
#     history = unet_model.fit(
#         train_dataset,
#         validation_data=val_dataset,
#         epochs=EPOCHS,
#         steps_per_epoch=100,  # CRITICAL: Limit steps to prevent infinite iteration
#         validation_steps=20,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     return unet_model, history


# # ==========================================
# # 6. MEMORY MONITORING UTILITY
# # ==========================================
# def print_memory_info():
#     """Print current memory usage."""
#     import psutil
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     print(f"\n💾 Memory Usage: {mem_info.rss / 1024**2:.1f} MB")


# # ==========================================
# # 6B. TRAINING METRICS & VISUALIZATION
# # ==========================================
# def plot_training_history(history, save_path='training_history.png'):
#     """Plot and save training/validation curves."""
#     try:
#         import matplotlib.pyplot as plt
        
#         fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
#         # Loss plot
#         axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
#         axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
#         axes[0].set_xlabel('Epoch')
#         axes[0].set_ylabel('Loss')
#         axes[0].set_title('Training & Validation Loss')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3)
        
#         # Accuracy plot
#         axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
#         axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
#         axes[1].set_xlabel('Epoch')
#         axes[1].set_ylabel('Accuracy')
#         axes[1].set_title('Training & Validation Accuracy')
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"\n📊 Training history saved to: {save_path}")
#         plt.close()
        
#     except ImportError:
#         print("⚠️  Matplotlib not installed — skipping plots")
#     except Exception as e:
#         print(f"⚠️  Could not save plots: {e}")


# def print_training_summary(history):
#     """Print training summary statistics."""
#     print("\n" + "="*70)
#     print("TRAINING SUMMARY")
#     print("="*70)
    
#     final_train_loss = history.history['loss'][-1]
#     final_val_loss = history.history['val_loss'][-1]
#     final_train_acc = history.history['accuracy'][-1]
#     final_val_acc = history.history['val_accuracy'][-1]
    
#     best_val_loss = min(history.history['val_loss'])
#     best_epoch = history.history['val_loss'].index(best_val_loss) + 1
    
#     print(f"✅ Training Complete!")
#     print(f"\n📈 Final Metrics:")
#     print(f"   Training Loss:     {final_train_loss:.4f}")
#     print(f"   Validation Loss:   {final_val_loss:.4f}")
#     print(f"   Training Accuracy: {final_train_acc:.4f}")
#     print(f"   Validation Acc:    {final_val_acc:.4f}")
    
#     print(f"\n🏆 Best Validation Loss:")
#     print(f"   Loss: {best_val_loss:.4f} @ Epoch {best_epoch}")
    
#     print(f"\n📊 Improvement:")
#     improvement = ((history.history['val_loss'][0] - best_val_loss) / history.history['val_loss'][0]) * 100
#     print(f"   Validation Loss improved by {improvement:.1f}%")
    
#     print("="*70)


# # ==========================================
# # 7. MAIN EXECUTION
# # ==========================================
# if __name__ == "__main__":
#     print("=" * 70)
#     print("U-NET SEGMENTATION WITH STREAMING DATA (MEMORY OPTIMIZED)")
#     print("=" * 70)
    
#     print(f"\n📱 GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
#     print_memory_info()
    
#     try:
#         # Step 1: Build file manifest (fast, no image loading)
#         print("\n🔍 Building dataset manifest (index only, no loading)...")
#         manifest = build_file_manifest(full_datasets_path)
#         print(f"✅ Found {len(manifest)} image-mask pairs")
        
#         if len(manifest) == 0:
#             print("❌ No image-mask pairs found! Check dataset path.")
#             exit(1)
        
#         print_memory_info()
        
#         # Step 2: Split manifest for train/val
#         split_idx = int(len(manifest) * 0.8)
#         manifest_train = manifest[:split_idx]
#         manifest_val = manifest[split_idx:]
        
#         print(f"📊 Train: {len(manifest_train)} | Val: {len(manifest_val)}")
        
#         # Step 3: Train with streaming
#         unet_model, history = train_streaming_unet(manifest_train, manifest_val)
        
#         # Step 4: Save
#         unet_model.save('unet_satellite_streaming_v1.keras')
#         print("\n✅ Training complete! Model saved as 'unet_satellite_streaming_v1.keras'")
        
#         # Step 5: Print summary
#         print_training_summary(history)
        
#         # Step 6: Plot training curves
#         plot_training_history(history)
        
#         print_memory_info()
        
#         print("\n" + "="*70)
#         print("📝 Next Steps:")
#         print("="*70)
#         print("1. Load model:  model = keras.models.load_model('unet_satellite_streaming_v1.keras')")
#         print("2. Predict:     prediction = model.predict(image_patch)")
#         print("3. Evaluate:    metrics = model.evaluate(val_dataset)")
#         print("="*70)
        
#     except KeyboardInterrupt:
#         print("\n⚠️  Training interrupted by user")
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()