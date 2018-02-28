import os
import tflearn
import datetime
import sklearn
import numpy as np

def create_perceptron(max_sequence_length, dict_size):

    net = tflearn.input_data([None, max_sequence_length])
    net = tflearn.embedding(net, input_dim=dict_size+1, output_dim=128)
    net = tflearn.fully_connected(net, 100, activation='relu')
    net = tflearn.fully_connected(net, 20, activation='relu')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1,
                             loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='../tensorboard/tensorboard_fully')

def main():
    create_perceptron()

if __name__ == "__main__":
    main()