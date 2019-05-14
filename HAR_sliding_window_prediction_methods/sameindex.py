import numpy as np
import pandas as pd
import common
import logging
from scipy import stats
logging.basicConfig(filename='sameindex.log', level=logging.DEBUG)
logging.info('lstm_HATP224\n')
#win data prediction results transform to dense labels
y_pred_win = np.load("y_pred_UCI_HAPT_UNET_0318.npy")
list_y_pred = []
for i in range(y_pred_win.shape[0]):
    j = 0
    while j<96:
        list_y_pred.append(y_pred_win[i])
        j+=1
y_pred = np.array(list_y_pred)
#y_pred = np.load("y_pred_UCI_HAPT_224_UNET_0318.npy")
label_gd = np.load("labels_gd_UCI_HAPT_224_UNET_0318.npy")
print('y_pred shape=',y_pred.shape)
print('y_gd shape=',label_gd.shape)
# N = y_pred.shape[0]



# calculate F1-Score
label_index = list(range(1,np.max(label_gd)+2))
accuracy,precision,recall,fscore,fw = common.checkAccuracy(label_gd+1,y_pred+1,label_index)
print("testing confusionmatrix:")
print(common.createConfusionMatrix(label_gd+1,y_pred+1,label_index))
logging.info("testing confusionmatrix:")
logging.info('testing confusionmatrix:{}'.format(common.createConfusionMatrix(label_gd+1,y_pred+1,label_index)))
print('testing acc:{}'.format(accuracy))
logging.info('testing acc:{}'.format(accuracy))
print('testing fscore:{}'.format(fscore))
logging.info('testing fscore:{}'.format(fscore))
print('testing weighted fscore:{}'.format(fw))
logging.info('testing weighted fscore:{}'.format(fw))