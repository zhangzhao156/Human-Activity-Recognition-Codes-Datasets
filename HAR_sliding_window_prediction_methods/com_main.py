# encoding=utf8
import numpy as np
import tensorflow as tf
from sklearn import metrics
import com_model
import win_data_load
import common
import pandas as pd
import logging
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, nargs='?', default='lstm', help="method name")
parser.add_argument('--dataset', type=str, nargs='?', default='WISDMar', help="dataset name")
parser.add_argument('--subseq', type=int, nargs='?', default=224, help="loss name")
args = parser.parse_args()
subseq=args.subseq
method = args.method
file_log = args.method + '_'+args.dataset+'_'+str(subseq)+'0320time.log'
logging.basicConfig(filename=file_log, level=logging.DEBUG)


X, y, X_train, X_test, y_train, y_test, N_FEATURES, act_classes = win_data_load.load_data(args.dataset,subseq=subseq)
N_EPOCHS = 2
BATCH_SIZE = 32
# labels_test_unary = (np.argmax(y_test, 1)).reshape((np.argmax(y_test, 1)).size)
# file_labels_test_unary = 'labels_gd_'+args.method+args.dataset+'_'+str(subseq)+'0312.npy'
# np.save(file_labels_test_unary,labels_test_unary)

if args.method == 'lstm':
    learning_rate = 0.0025
    n_steps = subseq
    n_input = N_FEATURES
    n_classes = act_classes
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')

    pred = com_model.lstm(_x=x, n_steps=subseq, n_input=N_FEATURES, n_classes=act_classes)

    loss = - tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

if args.method == 'cnn':
    learning_rate = 0.001
    n_steps = subseq
    n_input = N_FEATURES
    n_classes = act_classes
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')

    pred = com_model.cnn(X=x, num_labels=n_classes)

    loss = - tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

if args.method == 'cnnlstm':
    learning_rate = 0.0015
    n_steps = subseq
    n_input = N_FEATURES
    n_classes = act_classes
    x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')

    pred = com_model.cnnlstm(X=x, N_TIME_STEPS=subseq, N_CLASSES=n_classes)

    loss = - tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
train_count = len(X_train)
sum_time = 0
for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE),
                          range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
        sess.run(optimizer, feed_dict={x: X_train[start:end],
                                       y: y_train[start:end]})

    _, acc_train, loss_train = sess.run([pred, accuracy, loss], feed_dict={
        x: X_train, y: y_train})
    start = time.clock()
    _, acc_test, loss_test = sess.run([pred, accuracy, loss], feed_dict={
        x: X_test, y: y_test})
    end = time.clock()
    sum_time += (end - start)
    print(str(end - start))
    print(f'epoch: {i} train accuracy: {acc_train} train loss: {loss_train}')
    print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}%\n')
    logging.info(
        'Epoch: {} train_loss: {:.4f} train_accuracy{:.4f} test_loss: {:.4f} test_accuracy: {:.4f}%\n'.format(i,
                                                                                                              loss_train,
                                                                                                              acc_train,
                                                                                                              loss_test,
                                                                                                              acc_test))
print('mean_time=',str(sum_time/N_EPOCHS))
logging.info('mean_time={}'.format(str(sum_time/N_EPOCHS)))
predictions, acc_final, loss_final = sess.run([pred, accuracy, loss], feed_dict={x: X_test, y: y_test})
print()
print(f'final results: accuracy: {acc_final} loss: {loss_final}')
logging.info('Final_test_loss: {:.4f} test_accuracy: {:.4f}%\n'.format(loss_final,acc_final))
pred = sess.run(pred, feed_dict={x: X_test})
pred_y = np.argmax(pred, 1)
cm = metrics.confusion_matrix(np.argmax(y_test, 1), pred_y)
print(cm, '\n')
# cr=metrics.classification_report(np.argmax(y_test, 1), pred_y)
# print(cr, '\n')
print(metrics.precision_recall_fscore_support(np.argmax(y_test, 1), pred_y))
logging.info('testing confusionmatrix:{}'.format(cm))
# logging.info('testing report:{}'.format(cr))
logging.info('precision_recall_fscore_support: {}'.format(metrics.precision_recall_fscore_support(np.argmax(y_test, 1), pred_y)))


labels_test_unary = (np.argmax(y_test, 1)).reshape((np.argmax(y_test, 1)).size)
file_labels_test_unary = 'labels_gd_'+args.method+args.dataset+'_'+str(subseq)+'0320time.npy'
np.save(file_labels_test_unary,labels_test_unary)

y_pred = pred_y.reshape(pred_y.size)
y_pred_prob = pred.reshape(y_pred.size,pred.shape[1])
print(y_pred_prob.shape)
file_y_pred = 'y_pred_'+args.method+args.dataset+'_'+str(subseq)+'0320time.npy'
np.save(file_y_pred,y_pred)
file_y_pred_prob = 'y_pred_prob_'+args.method+args.dataset+'_'+str(subseq)+'0320time.npy'
np.save(file_y_pred_prob,y_pred_prob)



sess.close()

