import math
import numpy as np
import matplotlib.pyplot as plt

# Requires TensorFlow >=2.11 for the GroupNormalization layer.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, 
                                     kernel_size=1, 
                                     kernel_initializer=kernel_init(1.0)
                                    )(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, 
                            kernel_initializer=kernel_init(1.0)
                           )(temb)[:, None, None, :]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, 
                          kernel_size=3, 
                          padding="same", 
                          kernel_initializer=kernel_init(1.0)
                         )(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(width, 
                          kernel_size=3, 
                          padding="same", kernel_initializer=kernel_init(0.0)
                         )(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(width, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_init(1.0),)(x)
        
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        
        return x

    return apply


class TimeEmbedding(layers.Layer):
    """
    one time point to embedding vector with dim. R^1--> R^dim
    """
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        
        return emb


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(units, activation=activation_fn, kernel_initializer=kernel_init(1.0))(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply



def build_unet_model_c2(img_size_H,
                     img_size_W,
                     img_channels,
                     widths,
                     has_attention,
                     num_res_blocks=2,
                     norm_groups=8,
                     first_conv_channels=64,
                     interpolation="nearest",
                     activation_fn=keras.activations.swish,
                     encoder=None
                    ):
    """
    define U-Net model
    """
    # image_input and time_input
    image_input = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    image_input_past1 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past1")
    image_input_past2 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past2")

    # ================= image past embedding =================
    image_input_past_embed1 = encoder(image_input_past1)
    image_input_past_embed1 = layers.Conv2D(first_conv_channels,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             kernel_initializer=kernel_init(1.0),
                                            )(image_input_past_embed1)
    print("image_input_past_embed1 shape:", image_input_past_embed1.shape)

    image_input_past_embed2 = encoder(image_input_past1)
    image_input_past_embed2 = layers.Conv2D(first_conv_channels,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             kernel_initializer=kernel_init(1.0),
                                            )(image_input_past_embed2)
    print("image_input_past_embed2 shape:", image_input_past_embed2.shape)


    image_input_past = layers.Concatenate(axis=-1)([image_input_past1, image_input_past2])
    image_input_past = layers.Conv2D(first_conv_channels,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     kernel_initializer=kernel_init(1.0),
                                    )(image_input_past)
    print("image_input_past shape:", image_input_past.shape)
    
    image_input_past_embed = layers.Reshape((32*64, first_conv_channels))(image_input_past)
    print("image_input_past shape:", image_input_past_embed.shape)


    
    # ================= image_embedding =================
    image_input_embed = layers.Conv2D(first_conv_channels,
                                      kernel_size=(3, 3),
                                      padding="same",
                                      kernel_initializer=kernel_init(1.0),
                                     )(image_input)
    image_input_embed = layers.Reshape((32*64, first_conv_channels))(image_input_embed)

    # ================= cross_attention =================
    cross_atte = layers.MultiHeadAttention(num_heads=1, key_dim=256)(image_input_past_embed, image_input_embed)
    
    
    x = layers.Add()([image_input_embed, image_input_past_embed])
    x = layers.Add()([x, cross_atte])
    x = layers.Reshape((32, 64, first_conv_channels))(x)
    
    # time_embedding
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    print("x.shape:", x.shape, "temb.shape:", temb.shape)
    
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(img_channels, (2, 2), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    return keras.Model([image_input, time_input,
                        image_input_past1, image_input_past2, 
                       ], x, name="unet")



def build_unet_model_c2_28deg(img_size_H,
                     img_size_W,
                     img_channels,
                     widths,
                     has_attention,
                     num_res_blocks=2,
                     norm_groups=8,
                     first_conv_channels=64,
                     interpolation="nearest",
                     activation_fn=keras.activations.swish,
                     encoder=None
                    ):
    """
    define U-Net model
    """
    # image_input and time_input
    image_input = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    image_input_past1 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past1")
    image_input_past2 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past2")

    # ================= image past embedding =================
    image_input_past_embed1 = encoder(image_input_past1)
    image_input_past_embed1 = layers.Conv2D(first_conv_channels,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             kernel_initializer=kernel_init(1.0),
                                            )(image_input_past_embed1)
    print("image_input_past_embed1 shape:", image_input_past_embed1.shape)

    image_input_past_embed2 = encoder(image_input_past1)
    image_input_past_embed2 = layers.Conv2D(first_conv_channels,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             kernel_initializer=kernel_init(1.0),
                                            )(image_input_past_embed2)
    print("image_input_past_embed2 shape:", image_input_past_embed2.shape)

    
    image_input_past = layers.Concatenate(axis=-1)([image_input_past_embed1, image_input_past_embed2])
    image_input_past = layers.Conv2D(first_conv_channels,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     kernel_initializer=kernel_init(1.0),
                                    )(image_input_past)
    print("image_input_past shape:", image_input_past.shape)
    
    image_input_past_embed = layers.Reshape((64*128, first_conv_channels))(image_input_past)
    print("image_input_past shape:", image_input_past_embed.shape)


    
    # ================= image_embedding =================
    image_input_embed = layers.Conv2D(first_conv_channels,
                                      kernel_size=(3, 3),
                                      padding="same",
                                      kernel_initializer=kernel_init(1.0),
                                     )(image_input)
    image_input_embed = layers.Reshape((64*128, first_conv_channels))(image_input_embed)

    # ================= cross_attention =================
    cross_atte = layers.MultiHeadAttention(num_heads=1, key_dim=256)(image_input_past_embed, image_input_embed)
    
    
    x = layers.Add()([image_input_embed, image_input_past_embed])
    x = layers.Add()([x, cross_atte])
    x = layers.Reshape((64, 128, first_conv_channels))(x)
    
    # time_embedding
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    print("x.shape:", x.shape, "temb.shape:", temb.shape)
    
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(img_channels, (2, 2), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    return keras.Model([image_input, time_input,
                        image_input_past1, image_input_past2, 
                       ], x, name="unet")




def build_unet_model_c2_14deg(img_size_H,
                     img_size_W,
                     img_channels,
                     widths,
                     has_attention,
                     num_res_blocks=2,
                     norm_groups=8,
                     first_conv_channels=64,
                     interpolation="nearest",
                     activation_fn=keras.activations.swish,
                     encoder=None
                    ):
    """
    define U-Net model
    """
    # image_input and time_input
    image_input = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    image_input_past1 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past1")
    image_input_past2 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past2")

    # ================= image past embedding =================
    image_input_past_embed1 = encoder(image_input_past1)
    image_input_past_embed1 = layers.Conv2D(first_conv_channels,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             kernel_initializer=kernel_init(1.0),
                                            )(image_input_past_embed1)
    print("image_input_past_embed1 shape:", image_input_past_embed1.shape)

    image_input_past_embed2 = encoder(image_input_past1)
    image_input_past_embed2 = layers.Conv2D(first_conv_channels,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             kernel_initializer=kernel_init(1.0),
                                            )(image_input_past_embed2)
    print("image_input_past_embed2 shape:", image_input_past_embed2.shape)

    
    image_input_past = layers.Concatenate(axis=-1)([image_input_past_embed1, image_input_past_embed2])
    image_input_past = layers.Conv2D(first_conv_channels,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     kernel_initializer=kernel_init(1.0),
                                    )(image_input_past)
    print("image_input_past shape:", image_input_past.shape)
    
    image_input_past_embed = layers.Reshape((128*256, first_conv_channels))(image_input_past)
    print("image_input_past shape:", image_input_past_embed.shape)


    
    # ================= image_embedding =================
    image_input_embed = layers.Conv2D(first_conv_channels,
                                      kernel_size=(3, 3),
                                      padding="same",
                                      kernel_initializer=kernel_init(1.0),
                                     )(image_input)
    image_input_embed = layers.Reshape((128*256, first_conv_channels))(image_input_embed)

    # ================= cross_attention =================
    cross_atte = layers.MultiHeadAttention(num_heads=1, key_dim=256)(image_input_past_embed, image_input_embed)
    
    
    x = layers.Add()([image_input_embed, image_input_past_embed])
    x = layers.Add()([x, cross_atte])
    x = layers.Reshape((128, 256, first_conv_channels))(x)
    
    # time_embedding
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    print("x.shape:", x.shape, "temb.shape:", temb.shape)
    
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(img_channels, (2, 2), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    return keras.Model([image_input, time_input,
                        image_input_past1, image_input_past2, 
                       ], x, name="unet")


def build_unet_model_c2_no_cross_attn(img_size_H,
                     img_size_W,
                     img_channels,
                     widths,
                     has_attention,
                     num_res_blocks=2,
                     norm_groups=8,
                     first_conv_channels=64,
                     interpolation="nearest",
                     activation_fn=keras.activations.swish,
                     encoder=None
                    ):
    """
    define U-Net model
    """
    # image_input and time_input
    image_input = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    image_input_past1 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past1")
    image_input_past2 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past2")

    # ================= image past embedding =================
    image_input_past_embed1 = encoder(image_input_past1)
    image_input_past_embed1 = layers.Conv2D(first_conv_channels,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             kernel_initializer=kernel_init(1.0),
                                            )(image_input_past_embed1)
    print("image_input_past_embed1 shape:", image_input_past_embed1.shape)

    image_input_past_embed2 = encoder(image_input_past1)
    image_input_past_embed2 = layers.Conv2D(first_conv_channels,
                                             kernel_size=(3, 3),
                                             padding="same",
                                             kernel_initializer=kernel_init(1.0),
                                            )(image_input_past_embed2)
    print("image_input_past_embed2 shape:", image_input_past_embed2.shape)

    
    image_input_past_embed = layers.Concatenate(axis=-1)([image_input_past_embed1, image_input_past_embed2])
    image_input_past_embed = layers.Conv2D(first_conv_channels,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     kernel_initializer=kernel_init(1.0),
                                    )(image_input_past_embed)
    print("image_input_past_embed shape:", image_input_past_embed.shape)


    
    # ================= image_embedding =================
    image_input_embed = layers.Conv2D(first_conv_channels,
                                      kernel_size=(3, 3),
                                      padding="same",
                                      kernel_initializer=kernel_init(1.0),
                                     )(image_input)
    

    # ================= image_embedding =================
    x = layers.Add()([image_input_embed, image_input_past_embed])

    
    # time_embedding
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    print("x.shape:", x.shape, "temb.shape:", temb.shape)
    
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(img_channels, (2, 2), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    return keras.Model([image_input, time_input,
                        image_input_past1, image_input_past2, 
                       ], x, name="unet")
    


def build_unet_model_c2_no_encoder(img_size_H,
                     img_size_W,
                     img_channels,
                     widths,
                     has_attention,
                     num_res_blocks=2,
                     norm_groups=8,
                     first_conv_channels=64,
                     interpolation="nearest",
                     activation_fn=keras.activations.swish,
                     encoder=None
                    ):
    """
    define U-Net model
    """
    # image_input and time_input
    image_input = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    image_input_past1 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past1")
    image_input_past2 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past2")

    # ================= image past =================    
    image_input_past = layers.Concatenate(axis=-1)([image_input_past1, image_input_past2])
    image_input_past = layers.Conv2D(first_conv_channels,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     kernel_initializer=kernel_init(1.0),
                                    )(image_input_past)
    print("image_input_past shape:", image_input_past.shape)
    
    image_input_past_embed = layers.Reshape((32*64, first_conv_channels))(image_input_past)
    print("image_input_past shape:", image_input_past_embed.shape)

    
    # ================= image_embedding =================
    image_input_embed = layers.Conv2D(first_conv_channels,
                                      kernel_size=(3, 3),
                                      padding="same",
                                      kernel_initializer=kernel_init(1.0),
                                     )(image_input)
    image_input_embed = layers.Reshape((32*64, first_conv_channels))(image_input_embed)

    
    # ================= cross_attention =================
    cross_atte = layers.MultiHeadAttention(num_heads=1, key_dim=256)(image_input_past_embed, image_input_embed)
    
    
    x = layers.Add()([image_input_embed, image_input_past_embed])
    x = layers.Add()([x, cross_atte])
    x = layers.Reshape((32, 64, first_conv_channels))(x)
    
    # time_embedding
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    print("x.shape:", x.shape, "temb.shape:", temb.shape)
    
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(img_channels, (2, 2), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    return keras.Model([image_input, time_input, image_input_past1, image_input_past2], x, name="unet")


def build_unet_model_c2_no_cross_attn_encoder(img_size_H,
                     img_size_W,
                     img_channels,
                     widths,
                     has_attention,
                     num_res_blocks=2,
                     norm_groups=8,
                     first_conv_channels=64,
                     interpolation="nearest",
                     activation_fn=keras.activations.swish,
                     encoder=None
                    ):
    """
    define U-Net model
    """
    # image_input and time_input
    image_input = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    image_input_past1 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past1")
    image_input_past2 = layers.Input(shape=(img_size_H, img_size_W, img_channels), name="image_input_past2")

    # ================= image past embedding =================    
    image_input_past = layers.Concatenate(axis=-1)([image_input_past1, image_input_past2])
    image_input_past = layers.Conv2D(first_conv_channels,
                                     kernel_size=(3, 3),
                                     padding="same",
                                     kernel_initializer=kernel_init(1.0),
                                    )(image_input_past)
    print("image_input_past shape:", image_input_past.shape)


    
    # ================= image_embedding =================
    image_input_embed = layers.Conv2D(first_conv_channels,
                                      kernel_size=(3, 3),
                                      padding="same",
                                      kernel_initializer=kernel_init(1.0),
                                     )(image_input)
    

    # ================= image_past + image_embedding =================
    x = layers.Add()([image_input_embed, image_input_past])

    
    # time_embedding
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    print("x.shape:", x.shape, "temb.shape:", temb.shape)
    
    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)

        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)

    # End block
    x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(img_channels, (2, 2), padding="same", kernel_initializer=kernel_init(1.0))(x)
    
    return keras.Model([image_input, time_input,
                        image_input_past1, image_input_past2, 
                        # image_input_past3, image_input_past4
                       ], x, name="unet")