# ============ upadete using src files  =================
"""
training.py - Production U-Net Training Pipeline (Modular MLOps Edition)
Location: C:\\samcodebase\\deeplearning\\training.py
Description: Optimized training execution loop leveraging centralized model utilities
             and metrics for seamless scalability.
"""

import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from matplotlib import pyplot as plt

# Core Structural Imports from your local source directory
from src.unet_model import compile_production_model
from src.metrics import total_loss, jaccard_coef, dice_loss

# Import upstream dataset arrays securely
from satellite_data import total_classes, image_height, image_width, image_channels, x_test, x_train, y_test, y_train

# Clear lingering backend sessions to optimize memory layout
tf.keras.backend.clear_session()

# Generate production timestamp names
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"training_history_v1.2_{timestamp}.csv"
png_filename = f"training_history_v1.2_{timestamp}.png"
model_filename = f"best_unet_satellite_model_v1.2_{timestamp}.keras"

# =====================================================================
# 1. MODEL INITIALIZATION VIA SRC UTILITIES
# =====================================================================
# This calls your deep, modern 4-level U-Net architecture with Batch Normalization
model = compile_production_model(
    input_shape=(image_height, image_width, image_channels),
    num_classes=total_classes,
    learning_rate=1e-4
)

model.summary()

# =====================================================================
# 2. TRAINING EXECUTION LOOPS
# =====================================================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint(model_filename, monitor="val_loss", save_best_only=True, verbose=1), 
    CSVLogger(csv_filename, append=False)
] 

model_history = model.fit(
    x_train, y_train,
    batch_size=2,
    verbose=1,
    epochs=100,
    validation_data=(x_test, y_test),
    shuffle=True,  # Keeps mini-batches highly randomized across epochs
    callbacks=callbacks
)

# Robustly reload best trained asset to verify disk serialization state
model = load_model(
    model_filename, 
    custom_objects={
        "dice_loss": dice_loss,
        "total_loss": total_loss,
        "jaccard_coef": jaccard_coef,
        "CategoricalFocalCrossentropy": tf.keras.losses.CategoricalFocalCrossentropy
    }
)

# =====================================================================
# 3. METRIC VISUALIZATION PLOTTER
# =====================================================================
def plot_history(history, output_png):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Evaluation Metrics
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    ax1.plot(epochs, loss, 'y-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
    ax1.set_title("Training vs Validation Loss Map")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Jaccard IoU Score Metrics
    jaccard_values = history.history['jaccard_coef']
    val_jaccard_values = history.history['val_jaccard_coef']
    
    ax2.plot(epochs, jaccard_values, 'y-', label="Training IoU")
    ax2.plot(epochs, val_jaccard_values, 'r-', label="Validation IoU")
    ax2.set_title("Training vs Validation IoU Performance")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("IoU Score")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_png)
    print(f"[+] Performance curves exported cleanly -> {output_png}")
    plt.show()

# Run visualization processing loop
plot_history(model_history, png_filename)








# # =========== updated ==============

# """
# training.py - Production U-Net Training Pipeline
# Location: C:\\samcodebase\\deeplearning\\training.py
# Description: Optimized training execution loop with unified custom object handling,
#              fixed plotting lookups, and enabled data shuffling.
# """

# import datetime
# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Conv2DTranspose, Dropout
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
# from matplotlib import pyplot as plt

# # Import upstream data variables securely
# from satellite_data import total_classes, image_height, image_width, image_channels, x_test, x_train, y_test, y_train

# # Clear any lingering backend sessions before initiating initialization routines
# tf.keras.backend.clear_session()

# # Generate production-ready timestamp profiles
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# csv_filename = f"training_history_v1.2_{timestamp}.csv"
# png_filename = f"training_history_v1.2_{timestamp}.png"
# model_filename = f"best_unet_satellite_model_v1.2_{timestamp}.keras"

# # =====================================================================
# # 1. EVALUATION METRICS & LOSS ARCHITECTURE
# # =====================================================================
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

# # Instantiate Focal Crossentropy Loss Instance
# focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy()

# def total_loss(y_true, y_pred):
#     return dice_loss(y_true, y_pred) + focal_loss_fn(y_true, y_pred)


# # =====================================================================
# # 2. MULTI-CLASS U-NET MODEL BUILDER
# # =====================================================================
# def multi_unet_model(n_classes=5, image_height=256, image_width=256, image_channels=1):
#     inputs = Input((image_height, image_width, image_channels))
    
#     # Encoder Pathway
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

#     # Bottleneck Center Matrix
#     c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
#     c5 = Dropout(0.2)(c5)
#     c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)

#     # Decoder Pathway (Skip Connections integrated via Concatenate layer components)
#     u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
#     u6 = Concatenate()([u6, c4])
#     c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
#     c6 = Dropout(0.2)(c6)
#     c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

#     u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
#     u7 = Concatenate()([u7, c3])
#     c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
#     c7 = Dropout(0.2)(c7)
#     c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

#     u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
#     u8 = Concatenate()([u8, c2])
#     c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
#     c8 = Dropout(0.2)(c8)
#     c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

#     u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
#     u9 = Concatenate(axis=3)([u9, c1])
#     c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
#     c9 = Dropout(0.2)(c9)
#     c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

#     # Classification Layer Output Allocation
#     outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)
    
#     return Model(inputs=[inputs], outputs=[outputs])

# # Initialize and Compile
# model = multi_unet_model(n_classes=total_classes, 
#                          image_height=image_height, 
#                          image_width=image_width, 
#                          image_channels=image_channels)

# model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy', jaccard_coef])
# model.summary()

# # =====================================================================
# # 3. TRAINING EXECUTION LOOPS
# # =====================================================================
# callbacks = [
#     EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True), # Increased patience slightly to allow convergence
#     ModelCheckpoint(model_filename, monitor="val_loss", save_best_only=True, verbose=1), 
#     CSVLogger(csv_filename, append=False)
# ] 

# model_history = model.fit(
#     x_train, y_train,
#     batch_size=2,
#     verbose=1,
#     epochs=100,
#     validation_data=(x_test, y_test),
#     shuffle=True,  # Senior Dev Fix: Changed to True to break up local spatial clustering patterns
#     callbacks=callbacks
# )

# # Senior Dev Fix: Explicitly register structural focal_loss configurations to prevent loading crash
# model = load_model(
#     model_filename, 
#     custom_objects={
#         "dice_loss": dice_loss,
#         "total_loss": total_loss,
#         "jaccard_coef": jaccard_coef,
#         "CategoricalFocalCrossentropy": tf.keras.losses.CategoricalFocalCrossentropy
#     }
# )

# # =====================================================================
# # 4. METRIC VISUALIZATION PLOTTER
# # =====================================================================
# def plot_history(history, output_png):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#     # Loss Metrics Plots Execution
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
    
#     ax1.plot(epochs, loss, 'y-', label='Training Loss')
#     ax1.plot(epochs, val_loss, 'r-', label='Validation Loss')
#     ax1.set_title("Training vs Validation Loss Map")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend()

#     # Jaccard IoU Plots Execution
#     jaccard_values = history.history['jaccard_coef']
#     val_jaccard_values = history.history['val_jaccard_coef']
    
#     # Senior Dev Fix: Re-assigned lookup target safely to local array variables
#     ax2.plot(epochs, jaccard_values, 'y-', label="Training IoU")
#     ax2.plot(epochs, val_jaccard_values, 'r-', label="Validation IoU")
#     ax2.set_title("Training vs Validation IoU Performance")
#     ax2.set_xlabel("Epochs")
#     ax2.set_ylabel("IoU Score")
#     ax2.legend()

#     plt.tight_layout()
#     plt.savefig(output_png)
#     print(f"[+] Performance curves exported cleanly -> {output_png}")
#     plt.show()

# # Run performance visualization evaluation
# plot_history(model_history, png_filename)



















# # from keras.models import Model
# # from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
# # from keras.layers import concatenate, BatchNormalization, Dropout, Lambda

# # from keras import backend as K

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
# from tensorflow.keras.layers import concatenate, BatchNormalization, Dropout, Lambda
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
# import datetime

# # Generate timestamp string
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# csv_filename = f"training_history_v1.2{timestamp}.csv"
# png_filename = f"training_history_v1.2{timestamp}.png"
# model_filename = f"best_unet_satellite_model_v1.2{timestamp}.keras"



# from tensorflow.keras.models import load_model

# import tensorflow as tf

# from matplotlib import pyplot as plt

# # import segmentation-models 
# # as sm
# # sm.set_framework('tf.keras')
# # sm.framework()  

# from satellite_data import total_classes, image_height, image_width, image_channels, x_test, x_train, y_test, y_train

# # modelleing layers 2D 

# # ============ Evaluation Jaccard ============= 
# def jaccard_coef(y_true, y_pred):
#     y_true_flatten = K.flatten(y_true)
#     y_pred_flatten = K.flatten(y_pred)
#     intersection = K.sum(y_true_flatten * y_pred_flatten)
#     final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
#     return final_coef_value

# def multi_unet_model(n_classes=5, image_height=256, image_width=256, image_channels=1):
    
#     inputs = Input((image_height, image_width, image_channels))

#     source_input = inputs

# # first layer
#     c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(source_input)
#     c1 = Dropout(0.2)(c1)
#     c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
#     p1 = MaxPooling2D((2,2))(c1)
# # second layer
#     c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
#     c2 = Dropout(0.2)(c2)
#     c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
#     p2 = MaxPooling2D((2,2))(c2)
# # Third layer
#     c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
#     c3 = Dropout(0.2)(c3)
#     c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
#     p3 = MaxPooling2D((2,2))(c3)
# # Fourth layer
#     c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
#     c4 = Dropout(0.2)(c4)
#     c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
#     p4 = MaxPooling2D((2,2))(c4)
# # Fifth layer
#     c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
#     c5 = Dropout(0.2)(c5)
#     c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
# # Sixth layer
#     u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
#     c6 = Dropout(0.2)(c6)
#     c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)
# # Seventh layer
#     u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
#     c7 = Dropout(0.2)(c7)
#     c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)
# # Eigth layer
#     u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
#     c8 = Dropout(0.2)(c8)
#     c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)
# # Ninth layer
#     u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
#     u9 = concatenate([u9, c1], axis=3)
#     c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
#     c9 = Dropout(0.2)(c9)
#     c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

# # # Output Layer
#     outputs = Conv2D(n_classes, (1,1), activation="softmax")(c9)
#     print(f'outputs =====', outputs )

#     model = Model(inputs=[inputs], outputs=[outputs])
#     return model

# # ========== Metrics Config ==========
# metrics = ['accuracy', jaccard_coef]

# def get_deep_learning_model():
#     return multi_unet_model(n_classes=total_classes, 
#                             image_height=image_height, 
#                             image_width=image_width, 
#                             image_channels=image_channels)
    
# model = get_deep_learning_model()

# # model.get_config()

# print(f"hello: ", model)

# #  =========== Generating Loss Function: Focal (Cross Entropy Loss Extension) ================
# # dice loss > Focal Loss > Total Loss
# # Total Loss = (Dice loss + (1*Focal Loss))
# weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]

# # dice_loss = sm.losses.DiceLoss(class_weights = weights)
# # focal_loss = sm.losses.CategoricalFocalLoss()
# # total_loss = dice_loss + (1 * focal_loss)

# def dice_loss(y_true, y_pred, smooth=1e-6):
#     y_true_f = tf.reshape(y_true, [-1])
#     y_pred_f = tf.reshape(y_pred, [-1])
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# focal_loss = tf.keras.losses.CategoricalFocalCrossentropy()

# def total_loss(y_true, y_pred):
#     return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

# # ======================= Model compilation ===============

# tf.keras.backend.clear_session()
# model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
# # model.compile(optimizer='adam', loss=total_loss, metrics=['accuracy'])

# model.summary()
# # print(f"This model summary:", model.summary)


# # ============= Model Training =================
# callbacks = [
#     EarlyStopping(
#     monitor="val_loss", 
#     patience=5, 
#     restore_best_weights=True
#     ),

# ModelCheckpoint(
#     model_filename, #"best_unet_model_v1.h5",
#     monitor="val_loss",
#     save_best_only=True,
#     verbose=1
# ), 

#     CSVLogger(csv_filename, append=False)
# ] 

# model_history = model.fit(x_train, y_train,
#                         batch_size=2,
#                         verbose=1,
#                         epochs=100, #10,
#                         validation_data=(x_test, y_test),
#                         shuffle=False,
#                         callbacks=callbacks
#                         )

# model = load_model(model_filename, #"best_unet_model_v1.h5",
#                    custom_objects={"dice_loss": dice_loss,
#                                    "total_loss": total_loss,
#                                    "jaccard_coef": jaccard_coef})


# # history_a = model_history

# # ========== History Loss ==================================
# # loss = history_a.history.history['loss']
# # val_loss = history_a.history.history['val_loss']

# def plot_history(model_history, png_filename):
#     plt.figure(figsize=(12,4))


#     loss = model_history.history['loss']          # Fixed double .history bug
#     val_loss = model_history.history['val_loss']  # Fixed double .history bug
#     epochs = range(1, len(loss) + 1)
#     plt.plot(epochs, loss, 'y', label='Training Loss')
#     plt.plot(epochs, val_loss, 'x', label='Validation Loss')
#     plt.title("Training Vs Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.show()



#     # =========== Jaccard_coef ======================
#     # jaccard_coef = history_a.history['jaccard_coef']
#     # val_jaccard_coef = history_a.history['val_jaccard_coef']

#     jaccard_coef_values = model_history.history['jaccard_coef']
#     val_jaccard_coef_values = model_history.history['val_jaccard_coef']

#     epochs = range(1, len(jaccard_coef) + 1)
#     plt.plot(epochs, jaccard_coef, 'y', label="Training IoU")
#     plt.plot(epochs, val_jaccard_coef_values, 'r', label="Validation IoU")
#     plt.title("Training Vs Validation IoU")
#     plt.xlabel("Epochs")
#     plt.ylabel("Iou")
#     plt.legend()
#     plt.show()


#     plt.tight_layout()
#     plt.savefig(png_filename)
#     plt.show()

# # Call after training
# plot_history(model_history, png_filename)

