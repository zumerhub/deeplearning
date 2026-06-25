import matplotlib.pyplot as plt
import numpy as np

# from tiling import CLASS_COLORS


import numpy as np
import matplotlib.pyplot as plt

# Class mapping
CLASS_NAMES = {
    0: "Water",
    1: "Land",
    2: "Road",
    3: "Building",
    4: "Vegetation",
    5: "Unlabeled"
}

CLASS_COLORS = {
    0: (226,169,41),   # Water
    1: (132,41,246),   # Land
    2: (110,193,228),  # Road
    3: (60,16,152),    # Building
    4: (254,221,58),   # Vegetation
    5: (155,155,155)   # Unlabeled
}

# Your class colors (same mapping used for masks)
CLASS_COLORS = {
    0: (226,169,41),   # Water
    1: (132,41,246),   # Land
    2: (110,193,228),  # Road
    3: (60,16,152),    # Building
    4: (254,221,58),   # Vegetation
    5: (155,155,155)   # Unlabeled
}

# Class names in the same order as your IDs
CLASS_NAMES = ["Water", "Land", "Road", "Building", "Vegetation", "Unlabeled"]

def plot_per_class_iou(ious, class_names=CLASS_NAMES):
    plt.figure(figsize=(8,5))
    bars = plt.bar(class_names, ious, color=["#E2A929","#8429F6","#6EC1E4","#3C1098","#FEDD3A","#9B9B9B"])
    plt.ylim(0,1)
    plt.ylabel("IoU Score")
    plt.title("Per-Class IoU (Jaccard Index)")
    
    # Annotate bars with values
    for bar, iou in zip(bars, ious):
        if not np.isnan(iou):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{iou:.2f}", ha="center", va="bottom")
    plt.show()


def label_to_rgb(label_mask):
    """Convert integer mask (H,W) into RGB image using CLASS_COLORS."""
    h, w = label_mask.shape
    rgb_mask = np.zeros((h,w,3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb_mask[label_mask == class_id] = color
    return rgb_mask

def visualize_prediction(image, y_true, y_pred):
    """
    Show original image, ground truth mask, and predicted mask side by side.
    image: (256,256,3)
    y_true: (256,256,1)
    y_pred: (256,256,num_classes) softmax output
    """
    # Convert predictions to class IDs
    pred_classes = np.argmax(y_pred, axis=-1)

    # Convert masks to RGB
    true_rgb = label_to_rgb(np.squeeze(y_true))
    pred_rgb = label_to_rgb(pred_classes)

    # Plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(true_rgb)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(pred_rgb)
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.show()