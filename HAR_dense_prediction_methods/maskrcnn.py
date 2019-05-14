from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization


def Mask(
        subseq,
        INPUT_CHANNELS,
        filters,
        n_labels,
        kernel=3):
    inputs = Input((1, subseq, INPUT_CHANNELS))
    conv_0 = Convolution2D(filters,  (1, kernel), padding="same")(inputs)
    conv_0 = BatchNormalization()(conv_0)
    conv_0 = Activation("relu")(conv_0)
    pool_0 = MaxPooling2D(pool_size=(1, 2))(conv_0)

    conv_1 = Convolution2D(filters, (1, kernel), padding="same")(pool_0)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    conv_2 = Convolution2D(filters, (1, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    conv_3 = Convolution2D(filters, (1, kernel), padding="same")(conv_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    conv_4 = Convolution2D(filters, (1, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    deconv = Conv2DTranspose(filters, kernel_size=(1,2), strides=(1, 2))(conv_4)
    deconv = Activation("relu")(deconv)

    conv_5 = Convolution2D(n_labels, (1, 1), padding="same")(deconv)
    conv_5 = Activation('sigmoid')(conv_5)

    model = Model(inputs, conv_5, name="Mask_RCNN")

    return model

# def Mask(
#         subseq,
#         INPUT_CHANNELS,
#         filters,
#         n_labels,
#         kernel=3):
#     inputs = Input((1, subseq, INPUT_CHANNELS))
#     conv_0 = Convolution2D(filters,  (1, kernel), padding="same")(inputs)
#     conv_0 = BatchNormalization()(conv_0)
#     conv_0 = Activation("relu")(conv_0)
#     pool_0 = MaxPooling2D(pool_size=(1, 2))(conv_0)
#
#     conv_1 = Convolution2D(256, (1, kernel), padding="same")(pool_0)
#     conv_1 = BatchNormalization()(conv_1)
#     conv_1 = Activation("relu")(conv_1)
#
#     conv_2 = Convolution2D(256, (1, kernel), padding="same")(conv_1)
#     conv_2 = BatchNormalization()(conv_2)
#     conv_2 = Activation("relu")(conv_2)
#
#     conv_3 = Convolution2D(256, (1, kernel), padding="same")(conv_2)
#     conv_3 = BatchNormalization()(conv_3)
#     conv_3 = Activation("relu")(conv_3)
#
#     conv_4 = Convolution2D(256, (1, kernel), padding="same")(conv_3)
#     conv_4 = BatchNormalization()(conv_4)
#     conv_4 = Activation("relu")(conv_4)
#
#     deconv = Conv2DTranspose(256, kernel_size=(1,2), strides=(1, 2))(conv_4)
#     deconv = Activation("relu")(deconv)
#
#     conv_5 = Convolution2D(n_labels, (1, 1), padding="same")(deconv)
#     conv_5 = Activation('sigmoid')(conv_5)
#
#     model = Model(inputs, conv_5, name="Mask_RCNN")
#
#     return model