from tensorflow.keras import layers
import tensorflow.keras as keras


def unet_v2_get_model(img_size, num_classes, activation="softmax", filters=None):
    down_sampling_filters = [64, 128, 256]
    if filters is not None:
        down_sampling_filters = filters

    up_sampling_filters = down_sampling_filters.copy()
    up_sampling_filters.reverse()
    up_sampling_filters = up_sampling_filters + [32]

    inputs = keras.Input(shape=img_size + (3,))

    # [First half of the network: downsampling inputs]

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    print("down_sampling_filters:", down_sampling_filters)
    for filters in down_sampling_filters:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # [Second half of the network: upsampling inputs]

    print("up_sampling_filters:", up_sampling_filters)
    for filters in up_sampling_filters:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation=activation, padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model
