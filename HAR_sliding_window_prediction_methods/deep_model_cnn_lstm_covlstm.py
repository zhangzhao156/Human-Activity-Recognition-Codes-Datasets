import tensorflow as tf
import numpy as np


def lstm(_x, n_steps, n_input, n_classes):
    n_hidden = 32
    _x = tf.transpose(_x, [1, 0, 2])  # (128, ?, 9)
    print("# transpose shape: ", _x.shape)    # transpose shape:  (128, ?, 9)
    _x = tf.reshape(_x, [-1, n_input])  # (n_step*batch_size, n_input)
    print("# reshape shape: ", _x.shape)    # reshape shape:  (?, 9)
    _x = tf.layers.dense(
        inputs=_x,
        units=n_hidden,
        activation=tf.nn.relu,
    )
    print("# relu shape: ", _x.shape)     # relu shape:  (?, 32)
    _x = tf.split(_x, n_steps, 0)  # n_steps * (batch_size, n_hidden)
    # spilt makes _x.type from array --> list for static_rnn()
    print("# list shape: ", np.array(_x).shape)    # list shape:  (128,)
    print("# list unit shape: ", np.array(_x)[0].shape)    # list unit shape:  (?, 32)

    lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_1_drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_1, output_keep_prob=0.5)
    print("# cell_1 shape: ", lstm_cell_1.state_size)   # cell_1 shape:  LSTMStateTuple(c=32, h=32)

    lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2_drop = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_2, output_keep_prob=0.5)
    print("# cell_2 shape: ", lstm_cell_2.state_size)   # cell_2 shape:  LSTMStateTuple(c=32, h=32)

    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1_drop, lstm_cell_2_drop], state_is_tuple=True)
    print("# multi cells shape: ", lstm_cells.state_size)
    # multi cells shape:  (LSTMStateTuple(c=32, h=32), LSTMStateTuple(c=32, h=32))

    outputs, states = tf.nn.static_rnn(cell=lstm_cells, inputs=_x, dtype=tf.float32)
    print("# outputs & states shape: ", np.array(outputs).shape, np.array(states).shape)
    # outputs & states shape:  (128,) (2, 2)

    lstm_last_output = outputs[-1]  # N to 1
    print("# last output shape: ", lstm_last_output.shape)  # last output shape:  (?, 32)

    lstm_last_output = tf.layers.dense(
        inputs=lstm_last_output,
        units=n_hidden,
        activation=tf.nn.relu
    )
    print("# fully connected shape: ", lstm_last_output.shape)  # fully connected shape:  (?, 32)

    prediction = tf.layers.dense(
        inputs=lstm_last_output,
        units=n_classes,
        activation=tf.nn.softmax
    )
    print("# prediction shape: ", prediction.shape)  # prediction shape:  (?, 6)
    return prediction


def cnn(X, num_labels):
    # CNN
    # convolution layer 1
    conv1 = tf.layers.conv1d(
        inputs=X,
        filters=64,
        kernel_size=2,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    print("### convolution layer 1 shape: ", conv1.shape, " ###")

    # pooling layer 1
    pool1 = tf.layers.max_pooling1d(
        inputs=conv1,
        pool_size=4,
        strides=2,
        padding='same'
    )
    print("### pooling layer 1 shape: ", pool1.shape, " ###")

    # convolution layer 2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=128,
        kernel_size=2,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    print("### convolution layer 2 shape: ", conv2.shape, " ###")

    # pooling layer 2
    pool2 = tf.layers.max_pooling1d(
        inputs=conv2,
        pool_size=4,
        strides=2,
        padding='same'
    )
    print("### pooling layer 2 shape: ", pool2.shape, " ###")

    # convolution layer 3
    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=256,
        kernel_size=2,
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    print("### convolution layer 3 shape: ", conv3.shape, " ###")

    # pooling layer 3
    pool3 = tf.layers.max_pooling1d(
        inputs=conv3,
        pool_size=4,
        strides=2,
        padding='same'
    )
    print("### pooling layer 3 shape: ", pool3.shape, " ###")

    # flat output
    l_op = pool3
    shape = l_op.get_shape().as_list()
    flat = tf.reshape(l_op, [-1, shape[1] * shape[2]])
    print("### flat shape: ", flat.shape, " ###")

    # fully connected layer 1
    fc1 = tf.layers.dense(
        inputs=flat,
        units=100,
        activation=tf.nn.tanh
    )
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
    print("### fully connected layer 1 shape: ", fc1.shape, " ###")
    # bn_fc1 = tf.layers.batch_normalization(fc1, training=training)
    # bn_fc1_act = tf.nn.relu(bn_fc1)

    # fully connected layer 1
    fc2 = tf.layers.dense(
        inputs=fc1,
        units=100,
        activation=tf.nn.tanh
    )
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)
    print("### fully connected layer 2 shape: ", fc2.shape, " ###")
    # bn_fc2 = tf.layers.batch_normalization(fc2, training=training)
    # bn_fc2_act = tf.nn.relu(bn_fc2)

    # fully connected layer 3
    fc3 = tf.layers.dense(
        inputs=fc2,
        units=num_labels,
        activation=tf.nn.softmax
    )
    print("### fully connected layer 3 shape: ", fc3.shape, " ###")

    # prediction
    # y_ = tf.layers.batch_normalization(fc3, training=training)
    y_ = fc3
    print("### prediction shape: ", y_.get_shape(), " ###")
    return y_


def cnnlstm(X, N_TIME_STEPS, N_CLASSES):
    N_HIDDEN_UNITS = 32
    conv1 = tf.layers.conv1d(inputs=X, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv1d(inputs=conv1, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    # conv3 = tf.layers.conv1d(inputs=conv2, filters=32, kernel_size=5, strides=1, padding='same', activation = tf.nn.relu)
    n_ch = 32
    lstm_in = tf.transpose(conv2, [1, 0, 2])  # reshape into (seq_len, batch, channels)
    lstm_in = tf.reshape(lstm_in, [-1, n_ch])  # Now (seq_len*batch, n_channels)
    # To cells
    lstm_in = tf.layers.dense(lstm_in, N_HIDDEN_UNITS, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?

    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, N_TIME_STEPS, 0)

    # Add LSTM layers
    lstm = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    cell = tf.contrib.rnn.MultiRNNCell(lstm)
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32)

    # We only need the last output tensor to pass into a classifier
    pred = tf.layers.dense(outputs[-1], units=N_CLASSES,activation=tf.nn.softmax)
    return pred