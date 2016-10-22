import re

import h5py
import keras
import numpy as np

from DebugBatchSample import DebugBatchSample
from TTCModelData import TTCModelData


class ModelAbstract(object):

    def __init__(self):
        self.model = None
        self.training_history  = []  # List of dicts. from keras.callbacks.History objects which each have a history dict

    def train(self, X_train, y_train, X_validation, y_validation, batch_size, epochs, verbose=0):
        assert self.model is not None # Either 1) self.buildModel()
                                      #  or    2) self.load_ml_model() & self.set_sample_handler() have been called.
        th = self.model.fit(
                X_train,
                y_train,
                nb_epoch=epochs,
                batch_size=batch_size,
                shuffle=False,
                validation_data=(X_validation, y_validation),
                verbose= verbose,
            )
        self.training_history.append( th.history )
        return self.training_history[-1]


    def _load_keras_model(self, model_file):
        self.model = keras.models.load_model(model_file)
        self._unstash_learning_history(model_file)
        self.sample_handler = TTCModelData()
        self.sample_handler.unstash_xscaler(model_file)


    def load_ml_model(self, model_file):
        self._load_keras_model(model_file)


    def evaluate(self, sample_file, verbose=0, **X_pd_kwargs):
        assert self.model is not None # Either 1) self.buildModel()
                                      #  or    2) self.load_ml_model() & self.set_sample_handler() have been called.

        # this takes a list but, for now, only ever send in 1 sample_file
        self.sample_handler.load_prediction_files([sample_file], **X_pd_kwargs)
        X, y = self.sample_handler.convert_to_numpy(self.sample_handler.prediction_samples, 'Predictions')

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







    def predict(self, sample_handler, verbose=0, **X_pd_kwargs):
        assert self.model is not None # Either 1) self.buildModel()
                                      #  or    2) self.load_ml_model() have been called.
        predictions = []
        Xs, ys = sample_handler.convert_to_numpy(sample_handler.prediction_samples, 'Predictions')

        for idx, X in enumerate(np.split(Xs, Xs.shape[0] / sample_handler.max_timesteps)):
            bs = sample_handler.prediction_samples[idx]
            pcount = bs.dfX.shape[0]
            # X is _padded to self.max_timesteps but don't want predictions off of padding samples
            raw_predictions = self.model.predict(X[:pcount,:,:], pcount, verbose)
            reg_predictions = [bs.regularizedToDateTime(bs.event_time_col, rpred[0]) for rpred in raw_predictions]
            predictions.append(reg_predictions)

        return predictions



    def save_ml_model(self, modelData, save_file):
        self.model.save(save_file)
        modelData.stash_xscaler(save_file)
        self._stash_learning_history(save_file)


    def  buildModel(self, batch_size, timesteps, input_dim, in_neurons, hidden_layers, hidden_neurons, out_neurons,
                   rnn_activation, dense_activation):
        raise NotImplemented


    def _stash_learning_history(self, filename):
        with h5py.File(filename, 'r+') as h5f:
            for idx in range(len(self.training_history)):
                h5f.create_dataset('misc/training_history/th%05d/acc' % idx,
                                   data=self.training_history[idx]['acc']
                                   )
                h5f.create_dataset('misc/training_history/th%05d/loss' % idx,
                                   data=self.training_history[idx]['loss']
                                   )
                h5f.create_dataset('misc/training_history/th%05d/val_acc' % idx,
                                   data=self.training_history[idx]['val_acc']
                                   )
                h5f.create_dataset('misc/training_history/th%05d/val_loss' % idx,
                                   data=self.training_history[idx]['val_loss']
                                   )

    def _unstash_learning_history(self, filename):
        with h5py.File(filename, 'r+') as h5f:
            self.model.training_history = []
            groups = []
            h5f.visititems(lambda name, obj: groups.append(name))

            for idx, val in enumerate([k for k in groups if re.match('^misc/training_history/th\d+$', k)]):
                th = {}
                th['acc']      = h5f['misc/training_history/th%05d/acc'      % idx][:]
                th['loss']     = h5f['misc/training_history/th%05d/loss'     % idx][:]
                th['val_acc']  = h5f['misc/training_history/th%05d/val_acc'  % idx][:]
                th['val_loss'] = h5f['misc/training_history/th%05d/val_loss' % idx][:]
                self.training_history.append(th)
