import os

import h5py
import numpy as np
import re
import pandas as pd
from sklearn import preprocessing
import math
from tqdm import tqdm



from BatchSample import BatchSample
from Consts import CONST_EMPTY, CONST_XSCALER_MAX, CONST_XSCALER_MIN


class TTCModelData(object):
    """
    Home for the numpy and keras stuff
    """

    def __init__(self):

        self.training_files   = []
        self.validation_files = []
        self.testing_files    = []

        self.training_samples   = []
        self.validation_samples = []
        self.testing_samples    = []
        self.prediction_samples = []

        self.X_train      = None
        self.y_train      = None
        self.X_validation = None
        self.y_validation = None
        self.X_test       = None
        self.y_test       = None

        self.complete_features = []
        self.max_timesteps     = None
        self.xscaler           = None
        self.xscaler_mins      = None
        self.xscaler_maxs      = None

    def load_raw_sample_files(self, filelist, **X_pd_kwargs):
        self.training_files,   \
        self.validation_files, \
        self.testing_files = self.split_population(filelist)

        self.training_samples   = self._preprocess_sample_files(self.training_files,   "training files",   **X_pd_kwargs)
        self.validation_samples = self._preprocess_sample_files(self.validation_files, "validation files", **X_pd_kwargs)
        self.testing_samples    = self._preprocess_sample_files(self.testing_files,    "testing files",    **X_pd_kwargs)

        self._homogenize_features()
        self._fit_x_scaler()

        self._load_numpy_data()

    def load_prediction_files(self, filelist, **X_pd_kwargs):
        self.prediction_samples = self._preprocess_sample_files(filelist, 'prediction file', **X_pd_kwargs)

        for bs in self.prediction_samples:
            bs.pad_feature_columns(self.complete_features)

        X = np.concatenate([self.get_shaped_features_X(bs) for bs in self.prediction_samples])
        # ToDo: test is X the right shape?
        return X


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


    def _preprocess_sample_files(self, filelist, descr=None, **X_pd_kwargs):
        """
        Turn a list of files into a list of BatchSamples
        :type filelist: List<BatchSample>
        """

        bsList = [BatchSample() for _ in filelist]
        for idx, fname in enumerate(tqdm(filelist, desc=descr)):
            bsList[idx].process_file(fname, 0, 1, **X_pd_kwargs)

        return bsList


    def _homogenize_features(self):
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
        Must be called after self._homogenize_features()
        """
        # copy=True. don't want to touch the cache's numpy array in bs
        self.xscaler = preprocessing.MinMaxScaler((CONST_XSCALER_MIN, CONST_XSCALER_MAX), copy=True)
        for bs in (self.training_samples + self.validation_samples):
            self.xscaler.partial_fit(bs.get_dfI_values())


    def _load_numpy_data(self):
        """
        Set up the numpy X_train, y_train, X_validation etc ..
        """
        self.X_train,      self.y_train      = self.convert_to_numpy(self.training_samples,   "training reshaped")
        self.X_validation, self.y_validation = self.convert_to_numpy(self.validation_samples, "validation reshaped")
        self.X_test,       self.y_test       = self.convert_to_numpy(self.testing_samples,    "testing reshaped")

        X_pop = np.concatenate((self.X_train, self.X_validation))
        self.xscaler_mins = np.nanmin( np.nanmin( X_pop, axis=0), axis=0)
        self.xscaler_maxs = np.nanmax( np.nanmax( X_pop, axis=0), axis=0)


    def convert_to_numpy(self, batch_samples, descr=None):
        """
        BatchSample deals in pandas.DataFrames
        TTCModelData deals in numpy.nparrays
        """

        X_pop = np.concatenate([self.get_shaped_features_X(bs) for bs in batch_samples])
        y_pop = np.concatenate([self.get_shaped_y(bs)          for bs in batch_samples])

        assert type(X_pop) == np.ndarray
        assert type(y_pop) == np.ndarray
        assert len(X_pop.shape) == 3  # (nb_samples, timesteps, features )
        assert len(y_pop.shape) == 1  # (nb_samples)

        return X_pop, y_pop


    def get_shaped_features_X(self, batch_sample):
        """
        NB!!!.
        ToDo: Add all other (bool, datetime, timedelta & numeric) features back in.
        """

        # xscaler is copy=True
        scaled = self.xscaler.transform(batch_sample.get_dfI_values())

        # If using keras batch normalization ...
        # scaled = batch_sample.get_dfI_values()

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

    def get_shaped_y(self, batch_sample):
        return np.repeat(batch_sample.dfy['finish_watershedded'].values, self.max_timesteps)


    def _save_batch_samples(self, filename, batch_samples, samples_path):
        """
        Write out all the BatchSample's pd.DataFrame stuff
        :param filename:
        :param batch_samples:
        :return:
        """
        with pd.HDFStore(filename, mode='r+') as store:
            for idx, bs in enumerate(batch_samples):
                # Groups
                store['batch_samples/%s/bs%05d/dfX' % (samples_path, idx)] = bs.dfX
                store['batch_samples/%s/bs%05d/dfy' % (samples_path, idx)] = bs.dfy

        with h5py.File(filename, 'r+') as h5f:  # mode='r+' append
            for idx, bs in enumerate(batch_samples):
                #Attribs
                h5f['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs[ "CONST_COLNAME_PREFIX"]     = bs.CONST_COLNAME_PREFIX
                h5f['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs[ "event_time_col"]           = bs.event_time_col
                h5f['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs[ "event_label_col"]          = bs.event_label_col
                h5f['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs[ "_feature_padding_columns"] = bs._feature_padding_columns
                h5f['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs[ "filepath_or_buffer"]       = bs.filepath_or_buffer
                h5f['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs[ "source_was_buffer"]        = int(bs.source_was_buffer)

    def stash_xscaler(self, filename):
        with h5py.File(filename, 'r+') as h5f:
            #dt = h5py.special_dtype(vlen=np.unicode)
            #h5f.create_dataset('misc/complete_features',  (len(self.complete_features), ), data=self.complete_features, dtype=dt)

            # thx: http://stackoverflow.com/a/23223417
            ascii_features = [n.encode("ascii", "ignore") for n in self.complete_features]
            h5f.create_dataset('misc/complete_features', (len(ascii_features), ),
                               'S%d' % len(max(ascii_features, key=len)), ascii_features)

            h5f.create_dataset('misc/max_timesteps'      , data=self.max_timesteps )
            h5f.create_dataset('misc/xscaler/min_'       , data=self.xscaler.min_ )
            h5f.create_dataset('misc/xscaler/scale_'     , data=self.xscaler.scale_)
            h5f.create_dataset('misc/xscaler/data_min_'  , data=self.xscaler.data_min_)
            h5f.create_dataset('misc/xscaler/data_max_'  , data=self.xscaler.data_max_)
            h5f.create_dataset('misc/xscaler/data_range_', data=self.xscaler.data_range_)

    def save_np_data_file(self, filename):
        with h5py.File(filename, 'w') as h5f:  # mode='w' overwrite
            h5f.create_dataset('numpy/X_train',      data=self.X_train)
            h5f.create_dataset('numpy/y_train',      data=self.y_train)
            h5f.create_dataset('numpy/X_validation', data=self.X_validation)
            h5f.create_dataset('numpy/y_validation', data=self.y_validation)
            h5f.create_dataset('numpy/X_test',       data=self.X_test)
            h5f.create_dataset('numpy/y_test',       data=self.y_test)

        self.stash_xscaler(filename)

        self._save_batch_samples(filename, self.training_samples  , "training_samples")
        self._save_batch_samples(filename, self.validation_samples, "validation_samples")
        self._save_batch_samples(filename, self.testing_samples   , "testing_samples")


    def _load_batch_samples(self, filename, samples_path, verbose=1):
        """
        Handle loading the pd.DataFrames of a BatchSample()

        :param filename:
        :param samples_path:
        :return:
        """
        disable_progbar = verbose < 1
        with pd.HDFStore(filename) as store:
            # Init
            batch_samples = [BatchSample() for _ in store.keys() if re.match('^/dataframes/%s/bs\d+/dfX' %(samples_path), _)]

            for idx, val in enumerate(tqdm([ _ for _ in store.keys() if re.match( '^/dataframes/%s/bs\d+/dfX' % samples_path, _)],
                                           desc=samples_path,
                                           disable=disable_progbar)):
                # Groups
                batch_samples[idx].dfX = store[ 'batch_samples/%s/bs%05d/dfX' % (samples_path, idx) ]
                batch_samples[idx].dfy = store[ 'batch_samples/%s/bs%05d/dfy' % (samples_path, idx) ]

            for idx, val in enumerate([ _ for _ in store.keys() if re.match('batch_samples/%s' % samples_path, _ )]):
                # Attributes
                batch_samples[idx].CONST_COLNAME_PREFIX     = store['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs["CONST_COLNAME_PREFIX"]
                batch_samples[idx].event_time_col           = store['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs["event_time_col"]
                batch_samples[idx].event_label_col          = store['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs["event_label_col"]
                batch_samples[idx]._feature_padding_columns = store['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs["_feature_padding_columns"]
                batch_samples[idx].filepath_or_buffer       = store['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs["filepath_or_buffer"]
                batch_samples[idx].source_was_buffer        = bool(store['batch_samples/%s/bs%05d' % (samples_path, idx)].attrs["source_was_buffer"])

            return batch_samples

    def unstash_xscaler(self, filename):
        with h5py.File(filename, 'r+') as h5f:
            # ascii_features = [n.encode("ascii", "ignore") for n in self.complete_features]
            # h5f.create_dataset('misc/complete_features', (len(ascii_features), 1), 'S512', ascii_features)
            self.max_timesteps     = h5f['misc/max_timesteps'][...]
            self.complete_features = [n.decode('utf-8') for n in h5f['misc/complete_features'][:]]

            self.xscaler = preprocessing.MinMaxScaler((CONST_XSCALER_MIN, CONST_XSCALER_MAX), copy=True)
            self.xscaler.min_        = h5f['misc/xscaler/min_'       ][:]
            self.xscaler.scale_      = h5f['misc/xscaler/scale_'     ][:]
            self.xscaler.data_min_   = h5f['misc/xscaler/data_min_'  ][:]
            self.xscaler.data_max_   = h5f['misc/xscaler/data_max_'  ][:]
            self.xscaler.data_range_ = h5f['misc/xscaler/data_range_'][:]

    def load_np_data_file(self, filename, verbose=1):
        """

        """
        with h5py.File(filename, 'r') as h5f:
            self.X_train      = h5f['numpy/X_train'     ][:]
            self.y_train      = h5f['numpy/y_train'     ][:]
            self.X_validation = h5f['numpy/X_validation'][:]
            self.y_validation = h5f['numpy/y_validation'][:]
            self.X_test       = h5f['numpy/X_test'      ][:]
            self.y_test       = h5f['numpy/y_test'      ][:]

        self.unstash_xscaler(filename)

        self.training_samples   = self._load_batch_samples(filename, "training_samples", verbose)
        self.validation_samples = self._load_batch_samples(filename, "validation_samples", verbose)
        self.testing_samples    = self._load_batch_samples(filename, "testing_samples", verbose)


