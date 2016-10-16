import unittest
import numpy as np
import pandas as pd


class TestMLModel(unittest.TestCase):
    def setUp(self):
        from ModelLSTM import ModelLSTM

        # ToDo: Think about population needed, performance and fixed test_model.h5 file implications

        self.model = ModelSimpleRNN()




    def test_train(self):
        pass


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
