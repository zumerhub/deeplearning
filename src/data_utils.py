import numpy as np
import yaml

def load_config(config_path="config/config.yaml"):
    """
    Safely opens configuration maps across cross-platform environments.
    Enforces UTF-8 encoding to securely handle symbols, comments, and emojis.
    """
    with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
        return yaml.safe_load(f)

def get_vectorized_mappings(config):
    """Generates optimal structures for parsing masks and applying colormaps."""
    class_data = config["classes"]
    target_colors = {idx: info["rgb"] for idx, info in class_data.items()}
    
    # Generate mapping normalized dynamically for matplotlib previews
    color_list = [np.array(rgb)/255.0 for rgb in target_colors.values()]
    return target_colors, color_list

def rgb_to_label_batch(mask_batch, target_colors):
    """
    Highly optimized batch operation utilizing vector masks.
    Eliminates append loop bottlenecks.
    """
    # Initialize output array matching batch dimensions (B, H, W, 1)
    batch_segments = np.zeros((
        mask_batch.shape[0], 
        mask_batch.shape[1], 
        mask_batch.shape[2], 1), 
        dtype=np.uint8)
    
    for class_idx, rgb_val in target_colors.items():
        # Evaluate color matching simultaneously across all elements in the array
        match_mask = np.all(mask_batch == rgb_val, axis=-1)
        batch_segments[match_mask] = class_idx
        
    return batch_segments








# import numpy as np
# import yaml

# def load_config(config_path="config/config.yaml"):
#     with open(config_path, "r") as f:
#         return yaml.safe_load(f)

# def get_vectorized_mappings(config):
#     """Generates optimal structures for parsing masks and applying colormaps."""
#     class_data = config["classes"]
#     target_colors = {idx: info["rgb"] for idx, info in class_data.items()}
    
#     # Generate mapping normalized dynamically for matplotlib previews
#     color_list = [np.array(rgb)/255.0 for rgb in target_colors.values()]
#     return target_colors, color_list

# def rgb_to_label_batch(mask_batch, target_colors):
#     """
#     Highly optimized batch operation utilizing vector masks.
#     Eliminates append loop bottlenecks.
#     """
#     # Initialize output array matching batch dimensions (B, H, W, 1)
#     batch_segments = np.zeros((mask_batch.shape[0], mask_batch.shape[1], mask_batch.shape[2], 1), dtype=np.uint8)
    
#     for class_idx, rgb_val in target_colors.items():
#         # Evaluate color matching simultaneously across all elements in the array
#         match_mask = np.all(mask_batch == rgb_val, axis=-1)
#         batch_segments[match_mask] = class_idx
        
#     return batch_segments