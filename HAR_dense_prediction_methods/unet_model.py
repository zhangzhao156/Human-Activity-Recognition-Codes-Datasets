# coding: utf-8


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,Convolution2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
from keras.layers.merge import concatenate

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
# INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
# 修改成6维
# OUTPUT_MASK_CHANNELS = 6

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

# 上面5个函数没用上。。。用的keras自带的acc,loss

# 把conv2D参数改成（1，3），原来是（3,3）
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

# 网络改了输入维度（INPUT_CHANNELS, 1, 224），原来图片的输入（INPUT_CHANNELS, 224, 224）
# 2D池化的参数改成了pool_size=(1, 2)，之前是pool_size=(2, 2)
def ZF_UNET_224(subseq= 224,filters = 32,dropout_val=0.2, batch_norm=True, INPUT_CHANNELS=3, OUTPUT_MASK_CHANNELS=6):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 1, subseq))
        axis = 1
    else:
        inputs = Input((1, subseq, INPUT_CHANNELS))
        axis = 3
    # filters = 32

    conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(1, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, 0, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(1, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, 0, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(1, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters, 0, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(1, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters, 0, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(1, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters, 0, batch_norm)

    up_14 = concatenate([UpSampling2D(size=(1, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters, 0, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(1, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters, 0, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(1, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters, 0, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(1, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, 0, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(1, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224")
    return model

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


def ZF_UNET_224_4(subseq= 224,filters = 32,dropout_val=0.2, batch_norm=True, INPUT_CHANNELS=3, OUTPUT_MASK_CHANNELS=6):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 1, subseq))
        axis = 1
    else:
        inputs = Input((1, subseq, INPUT_CHANNELS))
        axis = 3
    # filters = 32

    conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(1, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, 0, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(1, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, 0, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(1, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters, 0, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(1, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters, 0, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(1, 2))(conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters, 0, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(1, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters, 0, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(1, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, 0, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(1, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224_4")
    return model

def ZF_UNET_224_3(subseq= 224,filters = 32,dropout_val=0.2, batch_norm=True, INPUT_CHANNELS=3, OUTPUT_MASK_CHANNELS=6):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 1, subseq))
        axis = 1
    else:
        inputs = Input((1, subseq, INPUT_CHANNELS))
        axis = 3
    # filters = 32

    conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(1, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, 0, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(1, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, 0, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(1, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters, 0, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(1, 2))(conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters, 0, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(1, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, 0, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(1, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224_3")
    return model

def ZF_UNET_224_2(subseq= 224,filters = 32,dropout_val=0.2, batch_norm=True, INPUT_CHANNELS=3, OUTPUT_MASK_CHANNELS=6):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 1, subseq))
        axis = 1
    else:
        inputs = Input((1, subseq, INPUT_CHANNELS))
        axis = 3
    # filters = 32

    conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(1, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, 0, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(1, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, 0, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(1, 2))(conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, 0, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(1, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224_2")
    return model

def ZF_UNET_224_6(subseq= 224,filters = 32,dropout_val=0.2, batch_norm=True, INPUT_CHANNELS=3, OUTPUT_MASK_CHANNELS=6):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 1, subseq))
        axis = 1
    else:
        inputs = Input((1, subseq, INPUT_CHANNELS))
        axis = 3
    # filters = 32

    conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(1, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, 0, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(1, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, 0, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(1, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters, 0, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(1, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters, 0, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(1, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters, 0, batch_norm)
    pool_3 = MaxPooling2D(pool_size=(1, 2))(conv_7)

    conv_3 = double_conv_layer(pool_3, 64*filters, 0, batch_norm)

    up_7 = concatenate([UpSampling2D(size=(1, 2))(conv_3), conv_7], axis=axis)
    up_conv_7 = double_conv_layer(up_7, 16 * filters, 0, batch_norm)

    up_14 = concatenate([UpSampling2D(size=(1, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters, 0, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(1, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters, 0, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(1, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters, 0, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(1, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, 0, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(1, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, dropout_val, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224_6")
    return model
