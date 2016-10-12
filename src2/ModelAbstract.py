import h5py
import keras
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
        self.complete_features = None
        self.max_timesteps     = None
        self.xscaler_params    = None
        self.training_history  = []

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


    def _load_keras_model(self, model_file):
        self.model = keras.models.load_model(model_file)
        self.sample_handler = TTCModelData()
        self.sample_handler.unstash_xscaler(model_file)


    def load(self, model_file):
        self._load_keras_model(model_file)


    def predict(self, model_file, sample_file, verbose=0, **X_pd_kwargs):
        self._load_keras_model()
        X = self.sample_handler.load_prediction_files([sample_file], **X_pd_kwargs)

        return self.model.predict(X, self.sample_handler.max_timesteps, verbose)


    def save(self, modelData, save_file):
        self.model.save(save_file)
        modelData.stash_xscaler(save_file)


    def  buildModel(self, batch_size, timesteps, input_dim, in_neurons, hidden_layers, hidden_neurons, out_neurons,
                         rnn_activation, dense_activation):
        raise NotImplemented
