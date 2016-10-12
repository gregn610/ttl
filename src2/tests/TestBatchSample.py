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

        """
        bs = self.batchSample
        X = bs.get_dfI_values()
        self.assertEqual(55, X.shape[0])
        self.assertEqual(52, X.shape[1])

        # The batchSamples change because of the population randomization so can't do this
        # first row of dfX features is at slice X[0,:]
        # second row of dfX features is at slice X[1,:]
        # third row of dfX features is at slice X[2,:]




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





