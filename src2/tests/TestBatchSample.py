import unittest
import numpy as np
import pandas as pd


from Consts import CONST_EMPTY


class TestBatchSample(unittest.TestCase):
    """

    """

    def setUp(self):
        from BatchSample import BatchSample
        from calendar import day_abbr

        self.FNAME_LOG = '/Users/gregn/Projects/Software/TimeTillComplete.com/data/population_v2.1/2005-11-11.log'
        self.X_pd_kwargs = {}

        self.batchSample = BatchSample()
        self.batchSample.process_file(self.FNAME_LOG, 0, 1, **self.X_pd_kwargs)


    def test_init(self):
        self.assertTrue(self.batchSample is not None)
        self.assertEqual(self.FNAME_LOG, self.batchSample.filepath_or_buffer)

    def test_init_dfX_columns(self):
        """
        The columns from the input file should be there (with headings for now)
        Each datetime column should be expanded into a one-hot day of the week mask
        """
        self.assertEqual([  # This should be just the original columns ???
            '0000__C0',
            '0000__C1',
            '0000__C2',
            '0000__C3',
            '0000__label',
            '0000__C0_watershedded'],
            list(self.batchSample.dfX.columns.values))

    def test_init_dfX_feature_columns(self):
        self.assertEqual(['0000__C2', '0000__C3', '0000__C0_watershedded']
                         , self.batchSample.get_dfX_feature_cols())

    def test_init_dfX_non_feature_columns(self):
        self.assertEqual(['0000__C0',
                          '0000__C1',
                          '0000__label'],
                         self.batchSample.get_non_feature_cols())

    def test_init_dfy_columns(self):
        """
        Unrelated to the input file column headings
        """
        self.assertEqual(['predicted_time', 'finish_watershedded']
                         , list(self.batchSample.dfy.columns.values))



    def test_get_raw_non_features_X(self):
        """
        """
        self.assertEqual((55, 3), self.batchSample.get_raw_non_features_X().shape)
        # some spot checks
        self.assertEqual(pd.Timestamp('2005-11-11 00:00:00'), self.batchSample.get_raw_non_features_X()[0, 0])
        self.assertEqual(pd.Timestamp('2005-11-11 02:51:45'), self.batchSample.get_raw_non_features_X()[22, 0])
        self.assertEqual(' Critical path event 44', self.batchSample.get_raw_non_features_X()[54, 1])

    def test_get_raw_y(self):
        """
        one row, two columns: predicted_time & finish_watershedded
        2005-11-11 07:22:17           442.283333
        """
        self.assertEqual((1, 2), self.batchSample.get_raw_y().shape)
        # spot checks
        self.assertEqual(pd.Timestamp('2005-11-11 07:22:17'), self.batchSample.get_raw_y()[0, 0])
        self.assertAlmostEqual(442.283333, self.batchSample.get_raw_y()[0, 1], places=5)

    def test_regularizedToDateTime(self):
        """
        AssertionError: numpy.datetime64('2005-01-03T20:32:34.068000') !=

        """
        self.assertEqual(pd.Timestamp('2005-11-11 20:34:34.068000'),
                         self.batchSample.regularizedToDateTime(self.batchSample.event_time_col, 1234.5678))

    def test_get_dfI_values(self):
        """
        ToDo: Needs work
        """
        bs = self.batchSample
        X = bs.get_dfI_values()
        self.assertEqual(55, X.shape[0])
        self.assertEqual(52, X.shape[1])

        # first row of dfX features is at slice X[0,:]
        self.assertEqual([
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 82.0, 9.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0],
            X[0, :].tolist())

        # second row of dfX features is at slice X[1,:]
        self.assertEqual([
            7.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 76.0, 3.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 7.3, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0],
            X[1, :].tolist())

        # third row of dfX features is at slice X[2,:]
        self.assertTrue(np.allclose([
            442.28333333 , 1.           ,  0.          ,   0.         , 0.         ,   0. ,
            0.           , 0.           ,  70.         ,   4.         , 134.3      ,
            289.41666667 , 17.85        , 427.56666667 , 171.75       , 83.63333333,
            336.3        , 365.41666667 , 442.28333333 ,  79.96666667 , 0.         , 259. ,
            257.58333333 , 239.4        , 357.65       , 191.76666667 , 69.55      ,
            126.43333333 , 340.33333333 , 123.38333333 ,  50.91666667 , 250.68333333,
            350.38333333 , 117.68333333 ,   7.3        , 345.33333333 , 66.66666667,
            385.66666667 ,  8.91666667  , 272.38333333 , 366.06666667 , 427.93333333,
            118.91666667 , 152.4        ,  78.58333333 ,  36.56666667 , 143.46666667,
            134.55       , 280.5        , 393.18333333 , 186.1        , 377.35      ],
            X[-1, :].tolist()) )
        



    def test_get_dfI_values_valid_A(self):
        """
        'Critical path event 9' is the first event that occurs multiple times.
        Check that they get stacked into timesteps correctly
        """
        bs = self.batchSample
        X = bs.get_dfI_values()

        # confirm inputs.
        # Unfortunately the original text has been discarded for the hash by now so have to recreate that here.
        # Maybe go back to dfX ???
        import hashlib
        event9 = hashlib.sha1(' Critical path event 9'.encode('UTF-8')).hexdigest()
        noise3 = hashlib.sha1(' Misc. noise event 3'.encode('UTF-8')).hexdigest()
        event19 = hashlib.sha1(' Critical path event 19'.encode('UTF-8')).hexdigest()

        # Don't forget zero based and header lines
        self.assertEqual(event9, bs.dfX[bs.event_label_col].iloc[11])
        self.assertEqual(event9, bs.dfX[bs.event_label_col].iloc[12])
        self.assertEqual(event9, bs.dfX[bs.event_label_col].iloc[13])

        self.assertEqual(event9, bs.dfX[bs.event_label_col].iloc[15])

        self.assertEqual(noise3, bs.dfX[bs.event_label_col].iloc[14])

        #
        self.assertEqual(event19, bs.dfX[bs.event_label_col].iloc[25])
        self.assertEqual(event19, bs.dfX[bs.event_label_col].iloc[26])
        self.assertEqual(event19, bs.dfX[bs.event_label_col].iloc[27])
        self.assertEqual(event19, bs.dfX[bs.event_label_col].iloc[28])
        self.assertEqual(event19, bs.dfX[bs.event_label_col].iloc[49])
        self.assertEqual(event19, bs.dfX[bs.event_label_col].iloc[50])
        self.assertEqual(event19, bs.dfX[bs.event_label_col].iloc[51])
        self.assertEqual(event19, bs.dfX[bs.event_label_col].iloc[53])





