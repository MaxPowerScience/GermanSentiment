import tensorflow as tf
import tflearn

def create_convolutional(max_sequence_length, dict_size):
    net = tflearn.input_data([None, max_sequence_length])
    net = tflearn.embedding(net, input_dim=dict_size+1, output_dim=128)
    branch1 = tflearn.conv_1d(net, 128, 3, padding='valid', activation='relu', regularizer='L2')
    branch2 = tflearn.conv_1d(net, 128, 4, padding='valid', activation='relu', regularizer='L2')
    branch3 = tflearn.conv_1d(net, 128, 5, padding='valid', activation='relu', regularizer='L2')
    net = tflearn.merge([branch1, branch2, branch3], mode='concat', axis=1)
    net = tf.expand_dims(net, 2)
    net = tflearn.layers.conv.global_max_pool(net)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='../tensorboard/tensorboard_conv')