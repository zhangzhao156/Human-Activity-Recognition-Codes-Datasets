# coding: utf-8


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Convolution2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
from keras.layers.merge import concatenate


def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

	
def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (1, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (1, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def FCN(inputsize=512,deconv_output_size=512,INPUT_CHANNELS=3,num_classes=6,filters=32):

    if K.image_dim_ordering()== 'th':
        inputs=Input((INPUT_CHANNELS,1,inputsize))
        axis=1
    else:
        inputs=Input((1,inputsize,INPUT_CHANNELS))
        axis=3
    conv1=Convolution2D(filters, 1, 3, border_mode='same', activation='relu')(inputs)
    pool1=MaxPooling2D((1, 4), strides=(1, 2))(conv1)
    conv2=Convolution2D(filters, 1, 3, border_mode='same', activation='relu')(pool1)
    pool2=MaxPooling2D((1, 4), strides=(1, 2))(conv2)
    conv3=Convolution2D(filters, 1, 3, border_mode='same', activation='relu')(pool2)
    pool3=MaxPooling2D((1, 4), strides=(1, 2))(conv3)
    conv4=Convolution2D(filters, 1, 3, border_mode='same', activation='relu')(pool3)
    pool4=MaxPooling2D((1, 4), strides=(1, 2))(conv4)
    fc=Convolution2D(num_classes, 1, 1, border_mode='same', activation='relu')(pool4)
    dconv=Deconvolution2D(num_classes, 1, 48,output_shape=(None, 1, deconv_output_size,num_classes),subsample=(1, 16),border_mode='valid')(fc)
    conv_final = Activation('softmax')(dconv)
    model = Model(inputs, conv_final, name="FCN")
    return model


