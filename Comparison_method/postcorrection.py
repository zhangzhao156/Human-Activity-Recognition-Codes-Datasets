# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import common
import logging
from scipy import stats
logging.basicConfig(filename='postcorrection.log', level=logging.DEBUG)
logging.info('hapt freq\n')

# post correction
B = 0
F = 0
C = 0

# HAPT
# P = 20
# y_pred_prob = np.load("y_pred_prob_UCI_HAPT_224_UNET_0318.npy")
# y_pred = np.load("y_pred_UCI_HAPT_224_UNET_0318.npy")
# label_gd = np.load("labels_gd_UCI_HAPT_224_UNET_0318.npy")
# # y_pred_prob = np.load("y_pred_prob_UCI_HAPT_224_FCN_0318.npy")
# # y_pred = np.load("y_pred_UCI_HAPT_224_FCN_0318.npy")
# # label_gd = np.load("labels_gd_UCI_HAPT_224_FCN_0318.npy")
# print('y_pred shape=',y_pred.shape)
# print('y_gd shape=',label_gd.shape)
# N = y_pred.shape[0]

# OPP donnot need post correction unet p = 4
# P = 10
# # y_pred_prob = np.load("y_pred_prob_UCI_Opportunity_224_UNET_0318.npy")
# # y_pred = np.load("y_pred_UCI_Opportunity_224_UNET_0318.npy")
# # label_gd = np.load("labels_gd_UCI_Opportunity_224_UNET_0318.npy")
# y_pred_prob = np.load("y_pred_prob_UCI_Opportunity_224_FCN_0318.npy")
# y_pred = np.load("y_pred_UCI_Opportunity_224_FCN_0318.npy")
# label_gd = np.load("labels_gd_UCI_Opportunity_224_FCN_0318.npy")
# print('y_pred shape=',y_pred.shape)
# print('y_gd shape=',label_gd.shape)
# N = y_pred.shape[0]

# Sanitation FCN P = 25
P = 25
# y_pred_prob = np.load("y_pred_prob_Sanitation_224_UNET_0318.npy")
# y_pred = np.load("y_pred_Sanitation_224_UNET_0318.npy")
# label_gd = np.load("labels_gd_Sanitation_224_UNET_0318.npy")
y_pred_prob = np.load("y_pred_prob_Sanitation_224_FCN_0318.npy")
y_pred = np.load("y_pred_Sanitation_224_FCN_0318.npy")
label_gd = np.load("labels_gd_Sanitation_224_FCN_0318.npy")
print('y_pred shape=',y_pred.shape)
print('y_gd shape=',label_gd.shape)
N = y_pred.shape[0]

# WISDM
# P = 25
# # y_pred_prob = np.load("y_pred_prob_WISDMar_224_UNET_0318.npy")
# # y_pred = np.load("y_pred_WISDMar_224_UNET_0318.npy")
# # label_gd = np.load("labels_gd_WISDMar_224_UNET_0318.npy")
# y_pred_prob = np.load("y_pred_prob_WISDMar_224_FCN_0318.npy")
# y_pred = np.load("y_pred_WISDMar_224_FCN_0318.npy")
# label_gd = np.load("labels_gd_WISDMar_224_FCN_0318.npy")
# print('y_pred shape=',y_pred.shape)
# print('y_gd shape=',label_gd.shape)
# N = y_pred.shape[0]

corrects=0
for i in range(N):
    if y_pred[i]==label_gd[i]:
        corrects=corrects+1
print('accuracy={:.4f}'.format(corrects/N))


# calculate P
# freq = []
# same = []
# k = 0
# while k<N-1:
#     C = 0
#     same = []
#     while label_gd[k+1]==label_gd[k]:
#         C+=1
#         same.append(label_gd[k])
#         if k<=N-3:
#             k += 1
#         else:
#             break
#     # print(C)
#     # print(same)
#     freq.append(C)
#     if k < N-1:
#         k+=1
# nfreq= np.array(freq)
# p = np.unique(nfreq)
# print(p)
# a = pd.value_counts(nfreq, sort=False)
# print(a)
# logging.info('freq: {}'.format(a))


i = 0
while i<N-1 :
    B=i
    C = 0
    # print('out while i',i)
    while y_pred[i+1] == y_pred[i]:
        C+=1
        # i+=1
        # cnnlstm sanitation results
        if i<=N-3:
            i += 1
            # print('i', i)
        else:
            break

    F=i

    if C <= P:
        if y_pred[B-1]==y_pred[F+1]:
            j = B
            while j<=F:
                y_pred[j]=y_pred[B-1]
                j+=1
        else:
            cof_B = np.dot(y_pred_prob[B - 1], y_pred_prob[B])
            cof_F = np.dot(y_pred_prob[F + 1], y_pred_prob[B])
            if cof_B>cof_F:
                j = B
                while j<=F:
                    y_pred[j]=y_pred[B-1]
                    j+=1
            else:
                j = B
                while j <= F:
                    y_pred[j] = y_pred[F + 1]
                    j += 1
    if i < N-1:
        i+=1
        # print('out i',i)





corrects=0
for i in range(N):
    if y_pred[i]==label_gd[i]:
        corrects=corrects+1
print('post correction accuracy={:.4f}'.format(corrects/N))


B = 0
F = 0
C = 0
overfill = 0
of = []
underfill = 0
uf = []
fragmentation = 0
frag = []
substitution = 0
sub_long = []
sub_short = []
N = y_pred.shape[0]
i = 0
while i<N :
    B=i
    # print('out while i',i)
    while y_pred[i] != label_gd[i]:
        C+=1
        i+=1
        # cnnlstm sanitation results
        # if i<79967:
        #     i += 1
        # else:
        #     break
        # print('i',i)
    F=i
    if B != F:
        if (y_pred[B]!=y_pred[B-1])&(y_pred[B]!=y_pred[F]):
            con_flag = (np.sum(y_pred[B:F] == y_pred[B])==(F-B))
            if (y_pred[B-1]==y_pred[F])&(con_flag):
                fragmentation+=C
                frag.append(C)
                # print('gd start',label_gd[B-1],'gd end',label_gd[F])
                # print('start',y_pred[B-1],'end',y_pred[F],'modestart',stats.mode(y_pred[B-5:B-1])[0][0],'modeend',stats.mode(y_pred[F:F+5])[0][0])
                # print('label_gd',label_gd[B:F])
                # print('y_pred',y_pred[B:F])
                # print('frag=',C)
            else:
                substitution+=C
                sub_short.append(C)
                # print('sub=', C)
                # print('sub label_gd', label_gd[B-1:F+1])
                # print('sub y_pred', y_pred[B-1:F+1])
        elif (y_pred[B]==y_pred[B-1])&(y_pred[B]==y_pred[F]):
            substitution+=C
            sub_long.append(C)
            # print('substitution',C)
        elif y_pred[B]==y_pred[F]:
            overfill+=C
            of.append(C)
            # print('overfill',C)
        else:
            underfill+=C
            uf.append(C)
            # print('underfill',C)
        C = 0
    else:
        i += 1

print('overfill=',overfill,'rate=',overfill/N)
# print(of)
print('underfill=',underfill,'rate=',underfill/N)
# print(uf)
print('fragmentation=',fragmentation,'rate=',fragmentation/N)
# nfrag = np.array(frag)
# print(np.unique(nfrag))
# print(pd.value_counts(nfrag, sort=False))
print('substitution=',substitution,'rate=',substitution/N)
# nsubshort = np.array(sub_short)
# print(np.unique(nsubshort))
# print(pd.value_counts(nsubshort, sort=False))


