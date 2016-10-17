import unittest
import numpy as np
import pandas as pd


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

        import tempfile
        self.tmpfile = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        print("Temp file name : %s" % self.tmpfile.name)

        self.modelData = TTCModelData()
        self.modelData.load_raw_sample_files(self.SAMPLE_FILES)
        # ToDo: Think about population needed, performance and fixed test_model.h5 file implications
        self.mlModel = ModelLSTM()


    def test_build_train_and_save(self):
        # Don't care about accuracy, just speed & basically working
        self.mlModel.buildModel(batch_size       = self.modelData.X_train.shape[1],
                               timesteps         = self.modelData.X_train.shape[1],
                               input_dim         = self.modelData.X_train.shape[2],
                               in_neurons        = 9,
                               hidden_layers     = 1,
                               hidden_neurons    = 9,
                               out_neurons       = 9,
                               rnn_activation    = 'tanh',
                               dense_activation  = 'linear'
                               )
        self.mlModel.train(self.modelData.X_train, self.modelData.y_train,
                           self.modelData.X_validation, self.modelData.y_validation,
                      batch_size=self.modelData.X_train.shape[1],
                      epochs=1,
                      verbose=0
                      )

        print('Saving trained model to: %s' % self.tmpfile.name)
        self.mlModel.save_ml_model(self.modelData, self.tmpfile.name)

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


    def test_prediction(self):
        """Are predictions coming out as regularized DateTimes?
        """
        pass


    def test_stash_unstash_learning_history(self):
        """"""
        import copy
        self.test_build_train_and_save()
        hist_before = copy.deepcopy(self.mlModel.training_history) # will just be the dicts
        self.mlModel.training_history= [] # because train appends by default

        import h5py
        import re
        with h5py.File(self.tmpfile.name, 'r+') as h5f:
            groups = []
            h5f.visititems(lambda name, obj: groups.append(name))
            self.assertListEqual(['misc/training_history/th00000/acc',
                                  'misc/training_history/th00000/loss',
                                  'misc/training_history/th00000/val_acc',
                                  'misc/training_history/th00000/val_loss'],
                                 [k for k in groups if re.match('^misc/training_history/th\d+/.+$', k)]
                                 )

        self.mlModel._unstash_learning_history(self.tmpfile.name)
        for idx in range(len(self.mlModel.training_history)):
            self.assertDictEqual( hist_before[idx], self.mlModel.training_history[idx] )



