import tflearn

def create_lstm(max_sequence_length, dict_size):
    net = tflearn.input_data([None, max_sequence_length])
    net = tflearn.embedding(net, input_dim=dict_size+1, output_dim=128)
    net = tflearn.lstm(net, 256, dropout=0.8)
    net = tflearn.fully_connected(net, 3, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='../tensorboard/tensorboard_lstm')