# encoding=utf8
import numpy as np
# from keras.callbacks import CSVLogger
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# import unet_224_model
import unet_model
import unet_data_load
import unet_info
import common
import pandas as pd
import logging
import argparse
import time
import segnet
import maskrcnn
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, nargs='?', default='WISDMar', help="dataset name")
parser.add_argument('--subseq', type=int, nargs='?', default=224, help="loss name")
# different network length (block number)
parser.add_argument('--block', type=str, nargs='?', default='5', help="block number")
parser.add_argument('--net', type=str, nargs='?', default='unet', help="net name")

args = parser.parse_args()
subseq=args.subseq
# file_log = 'UNET_'+args.block+'_'+args.dataset+'_'+str(subseq)+'0501.log'
# file_log = 'MASK_'+args.dataset+'_'+str(subseq)+'0501.log'
file_log = args.net+'_'+args.dataset+'_'+str(subseq)+'0503.log'

logging.basicConfig(filename=file_log, level=logging.DEBUG)
unet_info.begin()

X, y, X_train, X_test, y_train, y_test, N_FEATURES, act_classes = unet_data_load.load_data(args.dataset,subseq=subseq)

#get dense prediction results with overlap(transformed win data label's ground truth)
# y_test_resh = y_test.reshape(y_test.shape[0], y_test.shape[2], -1)
# y_test_resh_argmax = np.argmax(y_test_resh, axis=2)
# labels_test_unary = y_test_resh_argmax.reshape(y_test_resh_argmax.size)
# file_labels_test_unary = 'labels_gd_'+args.dataset+'_'+str(subseq)+'0317.npy'
# np.save(file_labels_test_unary,labels_test_unary)

epochs = 50
batch_size = 32
optim_type = 'adam'
learning_rate = 0.001
sum_time = 0
if (args.net == 'unet')and(args.block == '5'):
    model = unet_model.ZF_UNET_224(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES, OUTPUT_MASK_CHANNELS=act_classes)
elif (args.net == 'unet')and(args.block == '4'):
    model = unet_model.ZF_UNET_224_4(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES, OUTPUT_MASK_CHANNELS=act_classes)
elif (args.net == 'unet')and(args.block == '3'):
    model = unet_model.ZF_UNET_224_3(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES,
                                     OUTPUT_MASK_CHANNELS=act_classes)
elif (args.net == 'unet')and(args.block == '2'):
    model = unet_model.ZF_UNET_224_2(subseq=subseq, filters=32, INPUT_CHANNELS=N_FEATURES,
                                     OUTPUT_MASK_CHANNELS=act_classes)
elif (args.net == 'segnet')and(args.block == '5'):
    model = segnet.segnet(subseq=subseq, INPUT_CHANNELS=N_FEATURES, filters=64, n_labels=act_classes, kernel=3,
                          pool_size=(1, 2))
elif args.net == 'fcn':
    model = unet_model.FCN(inputsize=subseq, deconv_output_size=subseq, INPUT_CHANNELS=N_FEATURES,
                           num_classes=act_classes)
elif args.net == 'maskrcnn':
    model = maskrcnn.Mask(subseq=28, INPUT_CHANNELS=N_FEATURES, filters=32, n_labels=act_classes, kernel=3)

# model = segnet.segnet(subseq=subseq, INPUT_CHANNELS=N_FEATURES,filters=64, n_labels = act_classes, kernel=3, pool_size=(1, 2))
# model = segnet.segnet4(subseq=subseq, INPUT_CHANNELS=N_FEATURES,filters=64, n_labels = act_classes, kernel=3, pool_size=(1, 2))
# model = segnet.segnet3(subseq=subseq, INPUT_CHANNELS=N_FEATURES,filters=64, n_labels = act_classes, kernel=3, pool_size=(1, 2))
# model = segnet.segnet2(subseq=subseq, INPUT_CHANNELS=N_FEATURES,filters=64, n_labels = act_classes, kernel=3, pool_size=(1, 2))

# model = maskrcnn.Mask(subseq=28, INPUT_CHANNELS=N_FEATURES, filters=32, n_labels=act_classes,kernel=3)

# model = unet_model.FCN(inputsize=subseq,deconv_output_size=subseq,INPUT_CHANNELS=N_FEATURES,num_classes=act_classes)
# model = unet_model.ZF_UNET_224_3(subseq=subseq,filters=32, INPUT_CHANNELS=N_FEATURES, OUTPUT_MASK_CHANNELS=act_classes)
if optim_type == 'SGD':
    optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
else:
    optim = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                            factor=np.sqrt(0.1),
                            cooldown=0,
                            patience=10, min_lr=1e-12)
callbacks = [lr_reducer]
# early_stopper = EarlyStopping(monitor='val_loss',
#                             patience=30)
#
# callbacks = [lr_reducer, early_stopper]

history=model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.3, callbacks=callbacks)
# history=model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), callbacks=callbacks) 
acc=np.array(history.history['acc'])
loss=np.array(history.history['loss'])
val_acc = np.array(history.history['val_acc'])
val_loss = np.array(history.history['val_loss'])
for i in range(acc.shape[0]):
    logging.info('Epoch: {} loss: {:.4f} accuracy{:.4f} val_loss: {:.4f} val_accuracy{:.4f}%\n'.format(i+1, loss[i], acc[i],val_loss[i],val_acc[i]))

start = time.clock()
model.evaluate(X_test, y_test, batch_size=batch_size)
end = time.clock()
sum_time += (end - start)
print(str(end - start))
logging.info('mean_time={}'.format(str(end - start)))
y_test_resh = y_test.reshape(y_test.shape[0], y_test.shape[2], -1)
y_test_resh_argmax = np.argmax(y_test_resh, axis=2)
labels_test_unary = y_test_resh_argmax.reshape(y_test_resh_argmax.size)
file_labels_test_unary = 'labels_gd_'+args.dataset+'_'+str(subseq)+'_'+args.net+args.block+'_0503.npy'
np.save(file_labels_test_unary,labels_test_unary)

y_pred_raw = model.predict(X_test, batch_size=batch_size)
y_pred_resh = y_pred_raw.reshape(y_pred_raw.shape[0], y_pred_raw.shape[2], -1)
y_pred_resh_argmax = np.argmax(y_pred_resh, axis=2)
y_pred = y_pred_resh_argmax.reshape(y_pred_resh_argmax.size)
y_pred_prob = y_pred_resh.reshape(y_pred_resh_argmax.size,y_pred_resh.shape[2])
print(y_pred_prob.shape)
file_y_pred = 'y_pred_'+args.dataset+'_'+str(subseq)+'_'+args.net+args.block+'_0503.npy'
np.save(file_y_pred,y_pred)
file_y_pred_prob = 'y_pred_prob_'+args.dataset+'_'+str(subseq)+'_'+args.net+args.block+'_0503.npy'
np.save(file_y_pred_prob,y_pred_prob)

label_index = list(range(1,act_classes+1))
accuracy,precision,recall,fscore,fw = common.checkAccuracy(labels_test_unary+1,y_pred+1,label_index)
print("testing confusionmatrix:")
print(common.createConfusionMatrix(labels_test_unary+1,y_pred+1,label_index))
logging.info("testing confusionmatrix:")
logging.info('testing confusionmatrix:{}'.format(common.createConfusionMatrix(labels_test_unary+1,y_pred+1,label_index)))
print('testing acc:{}'.format(accuracy))
logging.info('testing acc:{}'.format(accuracy))
print('testing fscore:{}'.format(fscore))
logging.info('testing fscore:{}'.format(fscore))
print('testing weighted fscore:{}'.format(fw))
logging.info('testing weighted fscore:{}'.format(fw))

