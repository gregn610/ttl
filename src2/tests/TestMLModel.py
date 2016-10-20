import unittest
import numpy as np
import pandas as pd
import os
import tempfile

class TestMLModel(unittest.TestCase):
    def setUp(self):
        import os
        from ModelLSTM import ModelLSTM
        from TTCModelData import TTCModelData

        # Thanks: http://stackoverflow.com/a/4060259
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.SAMPLE_FILES_PATH = os.path.join(__location__, '../../data/population_v2.1/')
        self.SAMPLE_FILES = list(map(lambda fn: os.path.join(self.SAMPLE_FILES_PATH, fn), [
            "2004-01-01.log",
            "2004-01-02.log",
            "2004-01-03.log",
            "2004-01-04.log",
            "2004-01-05.log",
            "2004-01-06.log",
            "2004-01-07.log",
            "2004-01-08.log",
            "2004-01-09.log",
            "2004-01-10.log",
            "2004-01-11.log",
            "2004-01-12.log",
            "2004-01-13.log",
            "2004-01-14.log",
            "2004-01-15.log",
            "2004-01-16.log",
            "2004-01-17.log",
            "2004-01-18.log",
            "2004-01-19.log",
            "2004-01-20.log",
            "2004-01-21.log",
            "2004-01-22.log",
            "2004-01-23.log",
            "2004-01-24.log",
            "2004-01-25.log",
            "2004-01-26.log",
            "2004-01-27.log",
            "2004-01-28.log",
            "2004-01-29.log",
            "2004-01-30.log",
            "2004-01-31.log",
            "2004-02-01.log",
            "2004-02-02.log",
            "2004-02-03.log",
            "2004-02-04.log",
            "2004-02-05.log",
            "2004-02-06.log",
            "2004-02-07.log",
            "2004-02-08.log",
            "2004-02-09.log",
            "2004-02-10.log",
            "2004-02-11.log",
            "2004-02-12.log",
            "2004-02-13.log",
            "2004-02-14.log",
            "2004-02-15.log",
            "2004-02-16.log",
            "2004-02-17.log",
            "2004-02-18.log",
            "2004-02-19.log",
            "2004-02-20.log",
            "2004-02-21.log",
            "2004-02-22.log",
            "2004-02-23.log",
            "2004-02-24.log",
            "2004-02-25.log",
            "2004-02-26.log",
            "2004-02-27.log",
            "2004-02-28.log",
            "2004-02-29.log",
        ]))
        self.PREDICTION_FILES = list(map(lambda fn: os.path.join(self.SAMPLE_FILES_PATH, fn), [ "2004-01-01.log", ]))

        self.tmp_model_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        print("Temp model file name : %s" % self.tmp_model_file.name)

        self.tmp_data_file = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        print("Temp data file name : %s" % self.tmp_data_file.name)

        self.sample_handler = TTCModelData()
        self.sample_handler.load_raw_sample_files(self.SAMPLE_FILES)
        # ToDo: Think about population needed, performance and fixed test_model.h5 file implications
        self.mlModel = ModelLSTM()

        # Don't care about accuracy, just speed & basically working
        self.mlModel.buildModel(batch_size=self.sample_handler.X_train.shape[1],
                                timesteps=self.sample_handler.X_train.shape[1],
                                input_dim=self.sample_handler.X_train.shape[2],
                                in_neurons=9,
                                hidden_layers=1,
                                hidden_neurons=9,
                                out_neurons=9,
                                rnn_activation='tanh',
                                dense_activation='linear'
                                )
        self.mlModel.train(self.sample_handler.X_train, self.sample_handler.y_train,
                           self.sample_handler.X_validation, self.sample_handler.y_validation,
                           batch_size=self.sample_handler.X_train.shape[1],
                           epochs=1,
                           verbose=0
                           )

    def tearDown(self):
        os.unlink(self.tmp_model_file.name)
        os.unlink(self.tmp_data_file.name)

    def test_save_and_load(self):
        from ModelLSTM import ModelLSTM

        self.mlModel.save_ml_model(self.sample_handler, self.tmp_model_file.name)

        tmpModel = ModelLSTM()
        tmpModel.load_ml_model(self.tmp_model_file.name)

        # hmmm what to assert ???
        self.assertTrue(True)


    def test_load_ml_model(self):
        """Is the model and all its' stash_xscaler stuff reloaded properly
        """
        pass


    def test_save_ml_model(self):
        """Is the model and all its' stash_xscaler stuff saved properly
        """
        pass


    def test_prediction_full_sample(self):
        """Are predictions coming out as regularized DateTimes?
        """
        self.sample_handler.load_prediction_files(self.PREDICTION_FILES)
        Xs, ys = self.sample_handler.convert_to_numpy(self.sample_handler.prediction_samples, 'Test Predictions')
        predictions = self.mlModel.predict(self.sample_handler)

        self.assertTrue(True)


    def test_prediction_partial_samples(self):
        """Predictions run on log files that are still being written, do handle np dims
        """
        tmp_sample_file_22 = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        tmp_sample_file_44 = tempfile.NamedTemporaryFile(mode='w+b', delete=False)

        with open(self.PREDICTION_FILES[0]) as infile, \
            open(tmp_sample_file_22.name,'w') as of22, \
                open(tmp_sample_file_44.name, 'w') as of44:

            for idx in range(22):
                ln = infile.readline()
                of22.write(ln)
                of44.write(ln)

            for idx in range(22):
                ln = infile.readline()
                of44.write(ln)

        self.sample_handler.load_prediction_files([tmp_sample_file_22, tmp_sample_file_44])
        Xs, ys = self.sample_handler.convert_to_numpy(self.sample_handler.prediction_samples, 'Test Predictions')
        predictions = self.mlModel.predict(self.sample_handler)

        self.assertTrue(True)

        os.unlink(tmp_sample_file_22)
        os.unlink(tmp_sample_file_44)





    def test_stash_unstash_learning_history(self):
        """"""
        import copy
        self.test_save_and_load()
        hist_before = copy.deepcopy(self.mlModel.training_history) # will just be the dicts
        self.mlModel.training_history= [] # because train appends by default

        import h5py
        import re
        with h5py.File(self.tmp_model_file.name, 'r+') as h5f:
            groups = []
            h5f.visititems(lambda name, obj: groups.append(name))
            self.assertListEqual(['misc/training_history/th00000/acc',
                                  'misc/training_history/th00000/loss',
                                  'misc/training_history/th00000/val_acc',
                                  'misc/training_history/th00000/val_loss'],
                                 [k for k in groups if re.match('^misc/training_history/th\d+/.+$', k)]
                                 )

        self.mlModel._unstash_learning_history(self.tmp_model_file.name)
        for idx in range(len(self.mlModel.training_history)):
            self.assertDictEqual( hist_before[idx], self.mlModel.training_history[idx] )



