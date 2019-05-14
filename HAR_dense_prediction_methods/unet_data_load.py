# encoding=utf8

'''
    - train "UNET_224" CNN with random images
'''

import numpy as np
import pandas as pd
import h5py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

def feature_normalization(X):
    X = X.astype(float)
    scaler = StandardScaler().fit(X)
    data = scaler.transform(X)
    return data

def load_data(data_name='WISDMar',subseq=224):
    if data_name == 'Sanitation':
        X, y, X_train, X_test, y_train, y_test, N_FEATURES, act_classes = load_Sanitation(subseq)
    elif data_name == 'UCI_HAR':
        X, y, X_train, X_test, y_train, y_test, N_FEATURES, act_classes = load_UCI_HAR(subseq)
    elif data_name == 'UCI_HAPT':
        X, y, X_train, X_test, y_train, y_test, N_FEATURES, act_classes = load_UCI_HAPT(subseq)
    elif data_name == 'UCI_Opportunity':
        X, y, X_train, X_test, y_train, y_test, N_FEATURES, act_classes = load_UCI_Opportunity(subseq)
    else:
        X, y, X_train, X_test, y_train, y_test, N_FEATURES, act_classes = load_WISDMar(subseq)
    return X, y, X_train, X_test, y_train, y_test, N_FEATURES, act_classes


def load_WISDMar(subseq):
    columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv('WISDM_ar_v1.1_raw.txt', header = None, names = columns)
    df = df.iloc[:,[1,3,4,5]]
    df = df.dropna()

    np_df = np.array(df.drop('activity',axis=1))
    norm_np_df = feature_normalization(np_df)
    print('first 3 columns of normalized data:')
    print(norm_np_df[:3])
    N_TIME_STEPS = subseq
    # N_TIME_STEPS = 224
    N_FEATURES = 3
    # step可以改小点
    # step = 224
    step = subseq
    segments = []   
    for i in range(0, len(df) - N_TIME_STEPS, step):
        xs = norm_np_df[i: i + N_TIME_STEPS, 0]
        ys = norm_np_df[i: i + N_TIME_STEPS, 1]
        zs = norm_np_df[i: i + N_TIME_STEPS, 2]
        segments.append([xs, ys, zs])
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 1,N_TIME_STEPS, N_FEATURES)
    print('\ndata shape')
    print(reshaped_segments.shape)
    
    label=df['activity'].values
    label_encoder = LabelEncoder()
    integer_encoded1 = label_encoder.fit_transform(label)
    integer_encoded2 = integer_encoded1.reshape(len(integer_encoded1), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded2)
    labels = []
    for i in range(0, len(df) - N_TIME_STEPS, step):
        li = onehot_encoded[i: i + N_TIME_STEPS]    
        labels.append(li)
    act_classes = len(np.unique(integer_encoded1))
    reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(-1, 1,N_TIME_STEPS, act_classes)
    print('label shape')
    print(reshaped_labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(
            reshaped_segments, reshaped_labels, test_size=0.3, random_state=RANDOM_SEED)  
    print('training data shape')
    print(X_train.shape)

    return reshaped_segments, reshaped_labels, X_train, X_test, y_train, y_test, N_FEATURES, act_classes

def load_Sanitation(subseq):
    df = pd.read_csv('sanitation.csv').iloc[:,[0,1,2,3]]
    df = df.dropna()
    
    np_df = np.array(df.drop('label',axis=1))
    norm_np_df = feature_normalization(np_df)
    print('first 3 columns of normalized data:')
    print(norm_np_df[:3])
    
    N_TIME_STEPS = subseq
    N_FEATURES = 3
    # step可以改小点
    step = subseq
    segments = []
    for i in range(0, len(df) - N_TIME_STEPS, step):
        xs = norm_np_df[i: i + N_TIME_STEPS, 0]
        ys = norm_np_df[i: i + N_TIME_STEPS, 1]
        zs = norm_np_df[i: i + N_TIME_STEPS, 2]
        segments.append([xs, ys, zs])
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 1,N_TIME_STEPS, N_FEATURES)
    print('\ndata shape')
    print(reshaped_segments.shape)
    
    label=df['label'].values
    label_encoder = LabelEncoder()
    integer_encoded1 = label_encoder.fit_transform(label)
    integer_encoded2 = integer_encoded1.reshape(len(integer_encoded1), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded2)
    labels = []
    for i in range(0, len(df) - N_TIME_STEPS, step):
        li = onehot_encoded[i: i + N_TIME_STEPS]    
        labels.append(li)
    act_classes = len(np.unique(integer_encoded1))
    reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(-1,1,N_TIME_STEPS, act_classes)
    print('label shape')
    print(reshaped_labels.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
            reshaped_segments, reshaped_labels, test_size=0.3, random_state=RANDOM_SEED)
    print('training data shape')
    print(X_train.shape)
    
    return reshaped_segments, reshaped_labels, X_train, X_test, y_train, y_test, N_FEATURES, act_classes

def load_UCI_HAR(subseq):
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_"
    ]

    DATASET_PATH = "../data/UCI/HAR/"
    TRAIN = "train/"
    TEST = "test/"
    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    
    def load_X(X_signals_paths):
        X_signals = []
        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()
        return np.transpose(np.array(X_signals), (1, 2, 0))

    X_train1 = load_X(X_train_signals_paths)[::2,:,:]
    X_test1 = load_X(X_test_signals_paths)[::2,:,:]
    X_train2 = X_train1.reshape(-1,X_train1.shape[2])
    X_test2 = X_test1.reshape(-1,X_test1.shape[2])
    np_df = np.concatenate((X_train2,X_test2),axis=0)
    
    def load_y(y_path):
        file = open(y_path, 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]], 
            dtype=np.int32
        )
        file.close()   
        # Substract 1 to each output class for friendly 0-based indexing 
        return y_ - 1
   
    y_train1 = load_y(y_train_path)[::2,:]
    y_test1 = load_y(y_test_path)[::2,:]
    
    init_seg_size = X_train1.shape[1]
    y_train2 = np.zeros(len(y_train1)*init_seg_size)
    for i in range(0, len(y_train1)):
        y_train2[init_seg_size*i:init_seg_size*(i+1)] = y_train1[i][0]   
    y_test2 = np.zeros(len(y_test1)*init_seg_size)
    for i in range(0, len(y_test1)):
        y_test2[init_seg_size*i:init_seg_size*(i+1)] = y_test1[i][0]
    
    label = np.concatenate((y_train2,y_test2),axis=0)
    
    norm_np_df = feature_normalization(np_df)
    print('first 3 columns of normalized data:')
    print(norm_np_df[:3])
    
    N_TIME_STEPS = subseq
    N_FEATURES = 6
    # step可以改小点
    step = subseq
    segments = []
    for i in range(0, len(np_df) - N_TIME_STEPS, step):
        tmp = norm_np_df[i: i + N_TIME_STEPS, :]
        segments.append(tmp)
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 1,N_TIME_STEPS, N_FEATURES)
    print('\ndata shape')
    print(reshaped_segments.shape)
    
    label_encoder = LabelEncoder()
    integer_encoded1 = label_encoder.fit_transform(label)
    integer_encoded2 = integer_encoded1.reshape(len(integer_encoded1), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded2)
    labels = []
    for i in range(0, len(np_df) - N_TIME_STEPS, step):
        li = onehot_encoded[i: i + N_TIME_STEPS]
        labels.append(li)
    act_classes = len(np.unique(integer_encoded1))
    reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(-1,1,N_TIME_STEPS, act_classes)
    print('label shape')
    print(reshaped_labels.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
            reshaped_segments, reshaped_labels, test_size=0.3, random_state=RANDOM_SEED)
    print('training data shape')
    print(X_train.shape)
    
    return reshaped_segments, reshaped_labels, X_train, X_test, y_train, y_test, N_FEATURES, act_classes

def load_UCI_HAPT(subseq):
    DATA_PATH = "hapt/"
    # Reading training data
    fa = open(DATA_PATH + "all_data.csv")
    data_train = np.loadtxt(fname = fa, delimiter = ',')
    fa.close() 
    # Reading test data
    fa = open(DATA_PATH + "all_data_test.csv")
    data_test = np.loadtxt(fname = fa, delimiter = ',')
    fa.close()   
    # Reading training labels
    fa = open(DATA_PATH + "answers.csv")
    labels_train = np.loadtxt(fname = fa, delimiter = ',')
    fa.close()  
    # Reading test labels
    fa = open(DATA_PATH + "answers_test.csv")
    labels_test = np.loadtxt(fname = fa, delimiter = ',')
    fa.close()
    print(data_train.shape)
    print(labels_train.shape)
    print(data_test.shape)
    print(labels_test.shape)
    # X_train1 = data_train.reshape(-1, 6, 128)
    # X_test1 = data_test.reshape(-1, 6, 128)
    # X_train2 = np.transpose(np.array(X_train1), (0, 2, 1))
    # X_test2 = np.transpose(np.array(X_test1), (0, 2, 1))
    # X_train3 = X_train2[::2,:,:]
    # X_test3 = X_test2[::2,:,:]
    # X_train4 = X_train3.reshape(-1, 6)
    # X_test4 = X_test3.reshape(-1, 6)
    # np_df = np.concatenate((X_train4,X_test4),axis=0)
    np_df = np.concatenate((data_train, data_test), axis=0)
    # y_train1 = np.argmax(labels_train, axis=1)
    # y_test1 = np.argmax(labels_test, axis=1)
    # y_train2 = y_train1[::2]
    # y_test2 = y_test1[::2]
    # init_seg_size = 128
    # y_train3 = np.zeros(len(y_train2)*init_seg_size)
    # for i in range(0, len(y_train2)):
    #     y_train3[init_seg_size*i:init_seg_size*(i+1)] = y_train2[i]
    # y_test3 = np.zeros(len(y_test2)*init_seg_size)
    # for i in range(0, len(y_test2)):
    #     y_test3[init_seg_size*i:init_seg_size*(i+1)] = y_test2[i]
    # label = np.concatenate((y_train3,y_test3),axis=0)
    label = np.concatenate((labels_train, labels_test), axis=0)
       
    norm_np_df = feature_normalization(np_df)
    print('first 3 columns of normalized data:')
    print(norm_np_df[:3])
    
    N_TIME_STEPS = subseq
    N_FEATURES = 6
    # step可以改小点
    step = subseq
    segments = []
    for i in range(0, len(np_df) - N_TIME_STEPS, step):
        tmp = norm_np_df[i: i + N_TIME_STEPS, :]
        segments.append(tmp)
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 1,N_TIME_STEPS, N_FEATURES)
    print('\ndata shape')
    print(reshaped_segments.shape)
    
    label_encoder = LabelEncoder()
    integer_encoded1 = label_encoder.fit_transform(label)
    integer_encoded2 = integer_encoded1.reshape(len(integer_encoded1), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded2)
    labels = []
    for i in range(0, len(np_df) - N_TIME_STEPS, step):
        li = onehot_encoded[i: i + N_TIME_STEPS]
        labels.append(li)
    act_classes = len(np.unique(integer_encoded1))
    reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(-1,1,N_TIME_STEPS, act_classes)
    print('label shape')
    print(reshaped_labels.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
            reshaped_segments, reshaped_labels, test_size=0.3, random_state=RANDOM_SEED)
    print('training data shape')
    print(X_train.shape)
    
    return reshaped_segments, reshaped_labels, X_train, X_test, y_train, y_test, N_FEATURES, act_classes

def load_UCI_Opportunity(subseq):
    f = h5py.File('opportunity.h5', 'r')
    X_train1 = f['training']['inputs'].value
    y_train1 = f['training']['targets'].value
    X_test1 = f['test']['inputs'].value
    y_test1 = f['test']['targets'].value
    
    np_df = np.concatenate((X_train1,X_test1),axis=0)
    label = np.concatenate((y_train1,y_test1),axis=0)
    
    norm_np_df = feature_normalization(np_df)
    print('first 3 columns of normalized data:')
    print(norm_np_df[:3])
    
    N_TIME_STEPS = subseq
    N_FEATURES = 77
    # step可以改小点
    step = subseq
    segments = []
    for i in range(0, len(np_df) - N_TIME_STEPS, step):
        tmp = norm_np_df[i: i + N_TIME_STEPS, :]
        segments.append(tmp)
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, 1,N_TIME_STEPS, N_FEATURES)
    print('\ndata shape')
    print(reshaped_segments.shape)
    
    label_encoder = LabelEncoder()
    integer_encoded1 = label_encoder.fit_transform(label)
    integer_encoded2 = integer_encoded1.reshape(len(integer_encoded1), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded2)
    labels = []
    for i in range(0, len(np_df) - N_TIME_STEPS, step):
        li = onehot_encoded[i: i + N_TIME_STEPS]
        labels.append(li)
    act_classes = len(np.unique(integer_encoded1))
    reshaped_labels = np.asarray(labels, dtype= np.float32).reshape(-1,1,N_TIME_STEPS, act_classes)
    print('label shape')
    print(reshaped_labels.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(
            reshaped_segments, reshaped_labels, test_size=0.3, random_state=RANDOM_SEED)
    print('training data shape')
    print(X_train.shape)
    
    return reshaped_segments, reshaped_labels, X_train, X_test, y_train, y_test, N_FEATURES, act_classes