from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv1D, Conv2D, Conv3D,
                                     MaxPooling1D, MaxPooling2D, MaxPooling3D,
                                     UpSampling1D, UpSampling2D, UpSampling3D,
                                     LeakyReLU, Concatenate, TimeDistributed,
                                     BatchNormalization, Lambda, Input, Activation)
import tensorflow as tf

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

        self.conv3d_2 = TimeDistributed(Conv3D(num_filters, (3, 3, 3), padding="same"))
        self.conv1d_2 = Multi_TimeDistributed(Conv1D(num_filters, 3, padding="same"), iter=3)

        self.bn_1 = BatchNormalization()
        self.relu_1 = LeakyReLU()
        self.bn_2 = BatchNormalization()
        self.relu_2 = LeakyReLU()

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
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters
        self.conv_block = TimeDistributedConvBlock(num_filters)
        self.pool3d = TimeDistributed(MaxPooling3D((1, 2, 2)))

    def call(self, inputs):
        x = self.conv_block(inputs)
        p = self.pool3d(x)
        return x, p


    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters
        })
        return config
    
class TimeDistributedDecoderBlock(layers.Layer):
    def __init__(self, num_filters):
        super().__init__()
        self.num_filters = num_filters

        self.upsample3d = TimeDistributed(UpSampling3D((1, 2, 2)))
        self.conv3d = TimeDistributed(Conv3D(num_filters, (3, 3, 3), padding="same"))

        self.concat = Concatenate()
        self.conv_block = TimeDistributedConvBlock(num_filters)

    def call(self, inputs, skip):
        x = self.upsample3d(inputs)
        x = self.concat([x, skip])
        x = self.conv_block(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters
        })
        return config


def build_4d_unet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    s1, p1 = TimeDistributedEncoderBlock(16)(inputs)
    s2, p2 = TimeDistributedEncoderBlock(32)(p1)
    s3, p3 = TimeDistributedEncoderBlock(64)(p2)
    s4, p4 = TimeDistributedEncoderBlock(128)(p3)
    s5, p5 = TimeDistributedEncoderBlock(256)(p4)

    # b1 = TimeDistributedConvBlock(512)(p5)  

    d1 = TimeDistributedDecoderBlock(256)(p5, s5)
    d2 = TimeDistributedDecoderBlock(128)(d1, s4)
    d3 = TimeDistributedDecoderBlock(64)(d2, s3)
    d4 = TimeDistributedDecoderBlock(32)(d3, s2)
    d5 = TimeDistributedDecoderBlock(16)(d4, s1)

    outputs = TimeDistributed(Conv3D(num_classes, (1, 1, 1)))(d5)
    outputs = Activation(activation = 'sigmoid' if num_classes == 1 else 'softmax', name = 'output')(outputs)

    model = models.Model(inputs, outputs, name="4D-UNet")

    return model

