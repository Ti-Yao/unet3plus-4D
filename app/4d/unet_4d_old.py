import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Conv1D, Conv2D, Conv3D,
    MaxPooling1D, MaxPooling2D,
    UpSampling1D, UpSampling2D,
    LeakyReLU, Concatenate, TimeDistributed,
    BatchNormalization, Lambda, Input, Activation
)

def Transpose(perm):
    return Lambda(lambda x: tf.transpose(x, perm))


def Multi_TimeDistributed(layer, iter):
    for _ in range(iter):
        layer = TimeDistributed(layer)
    return layer


class TimeDistributedConvBlock(layers.Layer):
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters
        self.conv3d_1 = TimeDistributed(Conv3D(num_filters, (3, 3, 3), padding="same"))
        self.conv1d_1 = Multi_TimeDistributed(Conv1D(num_filters, 3, padding="same"), iter=3)
        self.bn_1 = TimeDistributed(BatchNormalization())
        self.relu_1 = TimeDistributed(LeakyReLU())
        self.conv3d_2 = TimeDistributed(Conv3D(num_filters, (3, 3, 3), padding="same"))
        self.conv1d_2 = Multi_TimeDistributed(Conv1D(num_filters, 3, padding="same"), iter=3)
        self.bn_2 = TimeDistributed(BatchNormalization())
        self.relu_2 = TimeDistributed(LeakyReLU())

    def call(self, inputs):
        x = self.conv3d_1(inputs)
        x = Transpose((0, 2, 3, 4, 1, 5))(x)
        x = self.conv1d_1(x)
        x = Transpose((0, 4, 1, 2, 3, 5))(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv3d_2(x)
        x = Transpose((0, 2, 3, 4, 1, 5))(x)
        x = self.conv1d_2(x)
        x = Transpose((0, 4, 1, 2, 3, 5))(x)
        
        x = self.bn_2(x)
        x = self.relu_2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"num_filters": self.num_filters})
        return config


class TimeDistributedEncoderBlock(layers.Layer):
    def __init__(self, num_filters, temporal_maxpool=True, slice_maxpool=True):
        super().__init__()
        self.num_filters = num_filters
        self.temporal_maxpool = temporal_maxpool
        self.slice_maxpool = slice_maxpool

        self.conv_block = TimeDistributedConvBlock(num_filters)
        self.pool2d = Multi_TimeDistributed(MaxPooling2D((2, 2)), iter=2)
        self.pool1d = Multi_TimeDistributed(MaxPooling1D(2), iter=3)

    def call(self, inputs):
        x = self.conv_block(inputs)
        p = self.pool2d(x)

        if self.slice_maxpool:
            p = Transpose((0, 1, 3, 4, 2, 5))(p)
            p = self.pool1d(p)
            p = Transpose((0, 1, 4, 2, 3, 5))(p)

        if self.temporal_maxpool:
            p = Transpose((0, 2, 3, 4, 1, 5))(p)
            p = self.pool1d(p)
            p = Transpose((0, 4, 1, 2, 3, 5))(p)

        return x, p

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "temporal_maxpool": self.temporal_maxpool,
            "slice_maxpool": self.slice_maxpool,
        })
        return config


class TimeDistributedDecoderBlock(layers.Layer):
    def __init__(self, num_filters, temporal_upsamp=True, slice_upsamp=True):
        super().__init__()
        self.num_filters = num_filters
        self.temporal_upsamp = temporal_upsamp
        self.slice_upsamp = slice_upsamp

        self.upsample2d = Multi_TimeDistributed(UpSampling2D((2, 2)), iter=2)
        self.conv2d = Multi_TimeDistributed(Conv2D(num_filters, (3, 3), padding="same"), iter=2)

        self.upsample1d = Multi_TimeDistributed(UpSampling1D(2), iter=3)
        self.conv1d = Multi_TimeDistributed(Conv1D(num_filters, 3, padding="same"), iter=3)

        self.concat = Concatenate()
        self.conv_block = TimeDistributedConvBlock(num_filters)

    def call(self, inputs, skip):
        x = self.upsample2d(inputs)
        x = self.conv2d(x)

        if self.slice_upsamp:
            x = Transpose((0, 1, 3, 4, 2, 5))(x)
            x = self.upsample1d(x)
            x = self.conv1d(x)
            x = Transpose((0, 1, 4, 2, 3, 5))(x)

        if self.temporal_upsamp:
            x = Transpose((0, 2, 3, 4, 1, 5))(x)
            x = self.upsample1d(x)
            x = self.conv1d(x)
            x = Transpose((0, 4, 1, 2, 3, 5))(x)

        x = self.concat([x, skip])
        x = self.conv_block(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "temporal_upsamp": self.temporal_upsamp,
            "slice_upsamp": self.slice_upsamp,
        })
        return config


def build_4d_unet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    slice_pool = False
    temporal_pool = False

    s1, p1 = TimeDistributedEncoderBlock(16, slice_maxpool=slice_pool, temporal_maxpool=temporal_pool)(inputs)
    s2, p2 = TimeDistributedEncoderBlock(32, slice_maxpool=slice_pool, temporal_maxpool=temporal_pool)(p1)
    s3, p3 = TimeDistributedEncoderBlock(64, slice_maxpool=slice_pool, temporal_maxpool=temporal_pool)(p2)
    s4, p4 = TimeDistributedEncoderBlock(128, slice_maxpool=slice_pool, temporal_maxpool=temporal_pool)(p3)
    s5, p5 = TimeDistributedEncoderBlock(256, slice_maxpool=slice_pool, temporal_maxpool=temporal_pool)(p4)

    b1 = TimeDistributedConvBlock(512)(p5)  

    d1 = TimeDistributedDecoderBlock(256, slice_upsamp=slice_pool, temporal_upsamp=temporal_pool)(b1, s5)
    d2 = TimeDistributedDecoderBlock(128, slice_upsamp=slice_pool, temporal_upsamp=temporal_pool)(d1, s4)
    d3 = TimeDistributedDecoderBlock(64, slice_upsamp=slice_pool, temporal_upsamp=temporal_pool)(d2, s3)
    d4 = TimeDistributedDecoderBlock(32, slice_upsamp=slice_pool, temporal_upsamp=temporal_pool)(d3, s2)
    d5 = TimeDistributedDecoderBlock(16, slice_upsamp=slice_pool, temporal_upsamp=temporal_pool)(d4, s1)

    outputs = TimeDistributed(Conv3D(num_classes, (1, 1, 1)), name="output")(d5)
    outputs = Activation(activation = 'sigmoid' if num_classes == 1 else 'softmax', name = 'output')(outputs)

    model = models.Model(inputs, outputs, name="4D-UNet")
    return model
