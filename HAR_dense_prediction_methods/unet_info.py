# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 14:11:51 2018

@author: zhangyu
"""

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from keras import backend as K
from keras import __version__

def set_gpu():
    # 指定第一块GPU可用 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    config = tf.ConfigProto()  
    config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    
    KTF.set_session(sess)

# if __name__ == '__main__':
def begin():
    set_gpu()
    print('GPU setting completed.')
    if K.backend() == 'tensorflow':
        try:
            from tensorflow import __version__ as __tensorflow_version__
            print('Tensorflow version: {}'.format(__tensorflow_version__))
        except:
            print('Tensorflow is unavailable...')
    else:
        try:
            from theano.version import version as __theano_version__
            print('Theano version: {}'.format(__theano_version__))
        except:
            print('Theano is unavailable...')
    print('Keras version {}'.format(__version__))
    print('Dim ordering:', K.image_dim_ordering())