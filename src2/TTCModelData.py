import os

import h5py
import numpy as np
import re
import pandas as pd
from sklearn import preprocessing
import math

from BatchSample import BatchSample
from Consts import CONST_EMPTY


class TTCModelData(object):
    """
    Home for the numpy and keras stuff
    """

    def __init__(self):

        self.training_files = []
        self.validation_files = []
        self.testing_files = []

        self.training_samples = []
        self.validation_samples = []
        self.testing_samples = []

        self.X_train = None
        self.y_train = None
        self.X_validation = None
        self.y_validation = None
        self.X_test = None
        self.y_test = None

        self.complete_features = []
        self.max_timesteps = None

        self.Xscaler = None

    def load_raw_sample_files(self, filelist, **X_pd_kwargs):
        self.training_files,   \
        self.validation_files, \
        self.testing_files = self.split_population(filelist)

        self.training_samples   = self._preprocess_sample_files(self.training_files,   **X_pd_kwargs)
        self.validation_samples = self._preprocess_sample_files(self.validation_files, **X_pd_kwargs)
        self.testing_samples    = self._preprocess_sample_files(self.testing_files,    **X_pd_kwargs)

        self._uniform_features()
        self._fit_x_scaler()

        self._load_numpy_data()


    def split_population(self, filelist):
        """
        Split a list of logfile names into 3 lists of 75%, 20%, 5%
        """

        sample_size        = math.floor(len(filelist) * 0.75)
        training_indices   = np.random.choice(range(len(filelist)), sample_size, replace=False)
        sample_size        = math.floor(len(filelist) * 0.2)
        validation_indices = np.random.choice(list(set(range(len(filelist))) - set(training_indices)),
                                              sample_size, replace=False)
        testing_indices = set(range(len(filelist))) - set(np.concatenate((training_indices, validation_indices)))

        return(
            [filelist[idx] for idx in training_indices],
            [filelist[idx] for idx in validation_indices],
            [filelist[idx] for idx in testing_indices],
        )

    def _preprocess_sample_files(self, filelist, **X_pd_kwargs):
        """
        Turn a list of files into a list of BatchSamples
        :type filelist: List<BatchSample>
        """

        bsList = [BatchSample() for _ in filelist]
        for idx, fname in enumerate(filelist):
            bsList[idx].process_file(fname, 0, 1, **X_pd_kwargs)

        return bsList


    def _uniform_features(self):
        # Gather whole population data (NB: with testing_samples)
        self.max_timesteps = 0
        for bs in (self.training_samples + self.validation_samples + self.testing_samples):
            self.max_timesteps = max([self.max_timesteps, bs.dfX.shape[0]])
            for ft in bs.get_dfX_feature_cols():
                if ft not in self.complete_features:
                    self.complete_features.append(ft)

        # Now go and do it
        for bs in (self.training_samples + self.validation_samples + self.testing_samples):
            bs.pad_feature_columns(self.complete_features)

    def _fit_x_scaler(self):
        """
        Must be called after self._uniform_features()
        """
        # copy=True. don't want to touch the cache's numpy array in bs
        self.Xscaler = preprocessing.MinMaxScaler((-0.99999999, 1),copy=True)
        for bs in (self.training_samples + self.validation_samples):
            self.Xscaler.partial_fit(bs.get_dfI_values())


    def _load_numpy_data(self):
        """
        Set up the numpy X_train, y_train, X_validation etc ..
        """
        self.X_train, self.y_train           = self.convert_to_numpy(self.training_samples)
        self.X_validation, self.y_validation = self.convert_to_numpy(self.validation_samples)
        self.X_test, self.y_test             = self.convert_to_numpy(self.testing_samples)


    def convert_to_numpy(self, batchSamples):
        """
        BatchSample deals in pandas.DataFrames
        TTCModelData deali in numpy.adarrays
        """
        X_pop = np.concatenate([self.get_shaped_features_X(bs) for bs in batchSamples])
        y_pop = np.concatenate([self.get_shaped_y(bs)          for bs in batchSamples])

        assert type(X_pop) == np.ndarray
        assert type(y_pop) == np.ndarray
        assert len(X_pop.shape) == 3  # (nb_samples, timesteps, features )
        assert len(y_pop.shape) == 1  # (nb_samples)

        return X_pop, y_pop


    def get_shaped_features_X(self, batchSample):
        """
        NB!!!.
        ToDo: Add all other (bool, datetime, timedelta & numeric) features back in.
        """

        # Xscaler is copy=True
        scaled = self.Xscaler.transform(batchSample.get_dfI_values())

        # If using keras batch normalization ...
        # scaled = batchSample.get_dfI_values()

        # Fall the 2D wall forward
        npRotated = scaled[np.newaxis, :, :]

        # npPaddedTimeSteps = np.ones((npRotated.shape[0], self.max_timesteps, npRotated.shape[2])) * CONST_EMPTY
        #
        # npPaddedTimeSteps[0:npRotated.shape[0],
        #       0:npRotated.shape[1],
        #       0:npRotated.shape[2]] = npRotated[:,:,:]
        #
        # return npPaddedTimeSteps

        # Explode each sample into timestep samples
        npPaddedTimeSteps = np.ones((self.max_timesteps, self.max_timesteps, npRotated.shape[2])) * CONST_EMPTY
        # then grab off bites, more sandwich per timestep
        for idx in range(npRotated.shape[1]):
            # self._shaped_features_X[idx,0:idx+1,:] = (np.flipud(npRotated[0,0:idx+1,:]))
            npPaddedTimeSteps[idx:idx + 1,
            0:idx + 1,
            0:npRotated.shape[2]] = npRotated[0:idx + 1, 0:idx + 1, :]

        assert type(npPaddedTimeSteps) == np.ndarray
        assert len(npPaddedTimeSteps.shape) == 3  # (nb_samples, timesteps, features )

        return npPaddedTimeSteps

    def get_shaped_y(self, batchSample):
        return np.repeat(batchSample.dfy['finish_watershedded'].values, self.max_timesteps)

    def _save_batch_samples(self, filename, batchSamples, samples_path):
        """
        Write out all the BatchSample's pd.DataFrame stuff
        :param filename:
        :param batchSamples:
        :return:
        """

        with pd.HDFStore(filename, mode='r+') as store:
            for idx, bs in enumerate(batchSamples):
                # Groups
                store['dataframes/training_samples/%d/dfX' % idx] = bs.dfX
                store['dataframes/training_samples/%d/dfy' % idx] = bs.dfy

        with h5py.File(filename, 'r+') as h5f:  # mode='r+' append
            for idx, bs in enumerate(batchSamples):
                # Dummy group to hold BatchSample attributes
                h5f.create_dataset('members/%s/%d' % (samples_path, idx), (0,), dtype='i')
                #Attribs
                h5f['members/%s/%d' % (samples_path, idx)].attrs[ "CONST_COLNAME_PREFIX"]     = bs.CONST_COLNAME_PREFIX
                h5f['members/%s/%d' % (samples_path, idx)].attrs[ "event_time_col"]           = bs.event_time_col
                h5f['members/%s/%d' % (samples_path, idx)].attrs[ "event_label_col"]          = bs.event_label_col
                h5f['members/%s/%d' % (samples_path, idx)].attrs[ "_feature_padding_columns"] = bs._feature_padding_columns
                h5f['members/%s/%d' % (samples_path, idx)].attrs[ "filepath_or_buffer"]       = bs.filepath_or_buffer
                h5f['members/%s/%d' % (samples_path, idx)].attrs[ "source_was_buffer"]        = bs.source_was_buffer

    def save_np_data_file(self, filename):
        with h5py.File(filename, 'w') as h5f:  # mode='w' overwrite
            h5f.create_dataset('numpy/X_train',      data=self.X_train)
            h5f.create_dataset('numpy/y_train',      data=self.y_train)
            h5f.create_dataset('numpy/X_validation', data=self.X_validation)
            h5f.create_dataset('numpy/y_validation', data=self.y_validation)
            h5f.create_dataset('numpy/X_test',       data=self.X_test)
            h5f.create_dataset('numpy/y_test',       data=self.y_test)

        self._save_batch_samples(filename, self.training_samples  , "training_samples")
        self._save_batch_samples(filename, self.validation_samples, "validation_samples")
        self._save_batch_samples(filename, self.testing_samples   , "testing_samples")


    def _load_batch_samples(self, filename, batchSamples, samples_path):
        """
        Handle loading the pd.DataFrames of a BatchSample()

        :param filename:
        :param batchSamples:
        :param samples_path:
        :return:
        """
        with pd.HDFStore(filename) as store:
            # Init
            batchSamples = [BatchSample() for _ in store.keys() if re.match('^/dataframes/training_samples/\d+/dfX', _)]

            for idx, val in enumerate([ _ for _ in store.keys() if re.match( '^/dataframes/%s/\d+/dfX' % samples_path, _)]):
                # Groups
                batchSamples[idx].dfX = store[ 'dataframes/%s/%d/dfX' % (samples_path, idx) ]
                batchSamples[idx].dfy = store[ 'dataframes/%s/%d/dfy' % (samples_path, idx) ]

            for idx, val in enumerate([ _ for _ in store.keys() if re.match('members/%s' % samples_path, _ )]):
                # Attributes
                batchSamples[idx].CONST_COLNAME_PREFIX     = store['members/%s/%d' % (samples_path, idx)].attrs["CONST_COLNAME_PREFIX"]
                batchSamples[idx].event_time_col           = store['members/%s/%d' % (samples_path, idx)].attrs["event_time_col"]
                batchSamples[idx].event_label_col          = store['members/%s/%d' % (samples_path, idx)].attrs["event_label_col"]
                batchSamples[idx]._feature_padding_columns = store['members/%s/%d' % (samples_path, idx)].attrs["_feature_padding_columns"]
                batchSamples[idx].filepath_or_buffer       = store['members/%s/%d' % (samples_path, idx)].attrs["filepath_or_buffer"]
                batchSamples[idx].source_was_buffer        = store['members/%s/%d' % (samples_path, idx)].attrs["source_was_buffer"]


    def load_np_data_file(self, filename):
        """

        """
        with h5py.File(filename, 'r') as h5f:
            self.X_train      = h5f['numpy/X_train'     ][:]
            self.y_train      = h5f['numpy/y_train'     ][:]
            self.X_validation = h5f['numpy/X_validation'][:]
            self.y_validation = h5f['numpy/y_validation'][:]
            self.X_test       = h5f['numpy/X_test'      ][:]
            self.y_test       = h5f['numpy/y_test'      ][:]

            self._load_batch_samples(filename, self.training_samples,   "training_samples")
            self._load_batch_samples(filename, self.validation_samples, "validation_samples")
            self._load_batch_samples(filename, self.testing_samples,    "testing_samples")

            self._uniform_features()

    def import_file(self, savefile):
        pass