import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, 
    Concatenate, BatchNormalization, Activation, Dropout
)
from tensorflow.keras.models import Model
from src.metrics import total_loss, jaccard_coef

def conv_block(inputs, num_filters, dropout_rate=0.0):
    """
    Standard Double Convolutional Block with Batch Normalization.
    Using modern Conv -> BN -> Activation order for training stability.
    """
    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    if dropout_rate > 0.0:
        x = Dropout(dropout_rate)(x)
        
    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(inputs, num_filters, dropout_rate=0.0):
    """Downsampling block: Extracts spatial features and reduces spatial dimensions."""
    x = conv_block(inputs, num_filters, dropout_rate)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """Upsampling block: Transposes convolutions and concatenates high-resolution skip features."""
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 3), num_classes=6):
    """
    Constructs a standard 4-level U-Net architecture for multi-class semantic segmentation.
    Outputs a probability distribution over the target classes via Softmax activation.
    """
    inputs = Input(input_shape)
    
    # ---  ENCODER (Downsampling Path) ---
    s1, p1 = encoder_block(inputs, 64, dropout_rate=0.1)
    s2, p2 = encoder_block(p1, 128, dropout_rate=0.1)
    s3, p3 = encoder_block(p2, 256, dropout_rate=0.2)
    s4, p4 = encoder_block(p3, 512, dropout_rate=0.2)
    
    # ---  BOTTLENECK ---
    b1 = conv_block(p4, 1024, dropout_rate=0.3)
    
    # ---  DECODER (Upsampling Path with Skip Connections) ---
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # ---  OUTPUT LAYER ---
    # 🟢 FIXED: Mapped explicitly to the dynamic num_classes variable
    outputs = Conv2D(num_classes, (1, 1), activation="softmax", name="segmentation_output")(d4)
    
    model = Model(inputs, outputs, name="UNet_Satellite_Segmentation")
    return model

def compile_production_model(input_shape=(256, 256, 3), num_classes=6, learning_rate=1e-4):
    """
    Instantiates and compiles the U-Net model with production loss architectures.
    Binds the custom focal+dice loss and tracking coefficients.
    """
    model = build_unet(input_shape=input_shape, num_classes=num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=total_loss,
        metrics=["accuracy", jaccard_coef]
    )
    
    return model

if __name__ == "__main__":
    # Smoke test to verify structural graph coherence
    test_model = build_unet()
    test_model.summary()
    print(f" U-Net graph initialized successfully. Output shape: {test_model.output_shape}")







# import tensorflow as tf
# from tf.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation, Dropout
# from tf.keras.models import Model
# from src.metrics import total_loss, jaccard_coef

# def conv_block(inputs, num_filters, dropout_rate=0.0):
#     """
#     Standard Double Convolutional Block with Batch Normalization.
#     Using modern Conv -> BN -> Activation order for training stability.
#     """
#     x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
    
#     if dropout_rate > 0.0:
#         x = Dropout(dropout_rate)(x)
        
#     x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#     return x

# def encoder_block(inputs, num_filters, dropout_rate=0.0):
#     """Downsampling block: Extracts spatial features and reduces spatial dimensions."""
#     x = conv_block(inputs, num_filters, dropout_rate)
#     p = MaxPooling2D((2, 2))(x)
#     return x, p

# def decoder_block(inputs, skip_features, num_filters):
#     """Upsampling block: Transposes convolutions and concatenates high-resolution skip features."""
#     x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(inputs)
#     x = Concatenate()([x, skip_features])
#     x = conv_block(x, num_filters)
#     return x

# def build_unet(input_shape=(256, 256, 3), num_classes=6):
#     """
#     Constructs a standard 4-level U-Net architecture for multi-class semantic segmentation.
#     Outputs a probability distribution over the target classes via Softmax activation.
#     """
#     inputs = Input(input_shape)
    
#     # ---  ENCODER (Downsampling Path) ---
#     s1, p1 = encoder_block(inputs, 64, dropout_rate=0.1)
#     s2, p2 = encoder_block(p1, 128, dropout_rate=0.1)
#     s3, p3 = encoder_block(p2, 256, dropout_rate=0.2)
#     s4, p4 = encoder_block(p3, 512, dropout_rate=0.2)
    
#     # ---  BOTTLENECK ---
#     b1 = conv_block(p4, 1024, dropout_rate=0.3)
    
#     # ---  DECODER (Upsampling Path with Skip Connections) ---
#     d1 = decoder_block(b1, s4, 512)
#     d2 = decoder_block(d1, s3, 256)
#     d3 = decoder_block(d2, s2, 128)
#     d4 = decoder_block(d3, s1, 64)
    
#     # ---  OUTPUT LAYER ---
#     # Multi-class segmentation requires Softmax activation
#     outputs = Conv2D(num_classes, (1, 1), activation="softmax", name="segmentation_output")(d4)
    
#     model = Model(inputs, outputs, name="UNet_Satellite_Segmentation")
#     return model

# def compile_production_model(input_shape=(256, 256, 3), num_classes=6, learning_rate=1e-4):
#     """
#     Instantiates and compiles the U-Net model with production loss architectures.
#     Binds the custom focal+dice loss and tracking coefficients.
#     """
#     model = build_unet(input_shape=input_shape, num_classes=num_classes)
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss=total_loss,
#         metrics=["accuracy", jaccard_coef]
#     )
    
#     return model

# if __name__ == "__main__":
#     # Smoke test to verify structural graph coherence
#     test_model = build_unet()
#     test_model.summary()
#     print(f" U-Net graph initialized successfully. Output shape: {test_model.output_shape}")