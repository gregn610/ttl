from keras.models import Sequential
from keras.layers.core import Dense, Activation, TimeDistributedDense, Masking
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from math import ceil
import numpy as np
import uuid
import os
from Consts import CONST_EMPTY
from TTCModelData import TTCModelData


class ModelAbstract(object):

    def __init__(self):
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
        os.environ['CUDA_HOME'] = '/usr/local/cuda'

        self.model = None
        self.training_history = []

    def train(self, X_train, y_train, X_validation, y_validation, epochs, verbose=0):
        assert self.model is not None

        self.training_history.append(
            self.model.fit(
                X_train,
                y_train,
                nb_epoch=epochs,
    #                batch_size=batch_size,
                shuffle=False,
                validation_data=(X_validation, y_validation),
                verbose= verbose,
            )
        )
        return self.training_history[-1]


    def load(self, savefile):
        self.model = TTCModelData()
        self.modelData


    def save(self, savefile):
        return self.model.save(savefile)

    def  buildModel(self, batch_size, timesteps, input_dim, in_neurons, hidden_layers, hidden_neurons, out_neurons,
                        rnn_activation, dense_activation):
        raise NotImplemented
