import tensorflow.keras as keras

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p


def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c


def build(img_size, model_sizes):
    inputs = keras.layers.Input(shape=img_size + (3,))
    p0 = inputs
    down_blocks_layers = [{
        "p": p0
    }]
    for i in range(len(model_sizes) - 1):
        c, p = down_block(down_blocks_layers[i]["p"], model_sizes[i])
        down_blocks_layers.append({
            "c": c,
            "p": p
        })

    bn = bottleneck(down_blocks_layers[len(down_blocks_layers) - 1]["p"], model_sizes[len(model_sizes) - 1])
    up_blocks_layers = [bn]

    for i in range(len(model_sizes) - 1):
        invert_i = len(model_sizes) - i - 1
        p = up_block(up_blocks_layers[i], down_blocks_layers[invert_i]["c"], model_sizes[invert_i])
        up_blocks_layers.append(p)

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(
        up_blocks_layers[len(up_blocks_layers) - 1])
    model = keras.models.Model(inputs, outputs)
    return model
