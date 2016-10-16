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
from DebugBatchSample import DebugBatchSample
from TTCModelData import TTCModelData


class ModelAbstract(object):

    def __init__(self):
        self.model = None
        self.complete_features = None
        self.max_timesteps     = None
        self.xscaler_params    = None
        self.training_history  = []

    def train(self, X_train, y_train, X_validation, y_validation, batch_size, epochs, verbose=0):
        assert self.model is not None

        self.training_history.append(
            self.model.fit(
                X_train,
                y_train,
                nb_epoch=epochs,
                batch_size=batch_size,
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


    def load_ml_model(self, model_file):
        self._load_keras_model(model_file)


    def evaluate(self, model_file, sample_file, verbose=0, **X_pd_kwargs):
        self._load_keras_model(model_file)
        # this takes a list but, for now, only ever send in 1 sample_file
        X, y = self.sample_handler.load_prediction_files([sample_file], **X_pd_kwargs)

        # Convert to a DebugBatchSample class
        debugBatchSample = self.sample_handler.prediction_samples[0]
        debugBatchSample.__class__ = DebugBatchSample
        debugBatchSample._conversion_from_BatchSample()

        predictions = []

        # Maybe should predict on batchs Xpop[0:idx,:,:]
        # and mode.reset() for each new Xpop ???

        for sliced in X:
            p = self.model.predict(sliced[np.newaxis, :, :])
            # print('p: %s' % str(p))
            predictions.append(
                debugBatchSample.regularizedToDateTime(debugBatchSample.event_time_col, (p[0, 0]))
            )

        idx = 0
        return debugBatchSample, predictions, idx







    def predict(self, model_file, sample_file, verbose=0, **X_pd_kwargs):
        self._load_keras_model(model_file)
        # this takes a list but, for now, only ever send in 1 sample_file
        X, y = self.sample_handler.load_prediction_files([sample_file], **X_pd_kwargs)

        raw_predictions = self.model.predict(X, self.sample_handler.max_timesteps, verbose)
        predictions = []
        for idx_bs, bs in enumerate(self.sample_handler.prediction_samples):
            predictions.append([])
            for rpred in raw_predictions:
                predictions[-1].append(bs.regularizedToDateTime(bs.event_time_col, rpred[0]) )

        return predictions



    def save_ml_model(self, modelData, save_file):
        self.model.save(save_file)
        modelData.stash_xscaler(save_file)


    def  buildModel(self, batch_size, timesteps, input_dim, in_neurons, hidden_layers, hidden_neurons, out_neurons,
                   rnn_activation, dense_activation,
                   nb_epoch, X_train, y_train, X_validation, y_validation, verbose=0):
        raise NotImplemented
