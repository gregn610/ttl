import unittest
import numpy as np
import pandas as pd
import os

"""
ToDo: refactor to support
ttc.py preprocess [--pandas-reader=(csv|excel|json|table)] <modelData.npz> LOGFILES ...
"""
from Consts import CONST_EMPTY

class TestTTCModelData(unittest.TestCase):


    def setUp(self):

        from TTCModelData import TTCModelData

        # Thanks: http://stackoverflow.com/a/4060259
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.SAMPLE_FILES_PATH      = os.path.join(__location__, '../../data/population_v2.1/')
        self.SAMPLE_FILES           = list(map(lambda fn: os.path.join(self.SAMPLE_FILES_PATH, fn), [
"2004-01-01.log", "2004-04-02.log", "2004-07-03.log", "2004-10-03.log",
"2004-01-02.log", "2004-04-03.log", "2004-07-04.log", "2004-10-04.log",
"2004-01-03.log", "2004-04-04.log", "2004-07-05.log", "2004-10-05.log",
"2004-01-04.log", "2004-04-05.log", "2004-07-06.log", "2004-10-06.log",
"2004-01-05.log", "2004-04-06.log", "2004-07-07.log", "2004-10-07.log",
"2004-01-06.log", "2004-04-07.log", "2004-07-08.log", "2004-10-08.log",
"2004-01-07.log", "2004-04-08.log", "2004-07-09.log", "2004-10-09.log",
"2004-01-08.log", "2004-04-09.log", "2004-07-10.log", "2004-10-10.log",
"2004-01-09.log", "2004-04-10.log", "2004-07-11.log", "2004-10-11.log",
"2004-01-10.log", "2004-04-11.log", "2004-07-12.log", "2004-10-12.log",
"2004-01-11.log", "2004-04-12.log", "2004-07-13.log", "2004-10-13.log",
"2004-01-12.log", "2004-04-13.log", "2004-07-14.log", "2004-10-14.log",
"2004-01-13.log", "2004-04-14.log", "2004-07-15.log", "2004-10-15.log",
"2004-01-14.log", "2004-04-15.log", "2004-07-16.log", "2004-10-16.log",
"2004-01-15.log", "2004-04-16.log", "2004-07-17.log", "2004-10-17.log",
"2004-01-16.log", "2004-04-17.log", "2004-07-18.log", "2004-10-18.log",
"2004-01-17.log", "2004-04-18.log", "2004-07-19.log", "2004-10-19.log",
"2004-01-18.log", "2004-04-19.log", "2004-07-20.log", "2004-10-20.log",
"2004-01-19.log", "2004-04-20.log", "2004-07-21.log", "2004-10-21.log",
"2004-01-20.log", "2004-04-21.log", "2004-07-22.log", "2004-10-22.log",
"2004-01-21.log", "2004-04-22.log", "2004-07-23.log", "2004-10-23.log",
"2004-01-22.log", "2004-04-23.log", "2004-07-24.log", "2004-10-24.log",
"2004-01-23.log", "2004-04-24.log", "2004-07-25.log", "2004-10-25.log",
"2004-01-24.log", "2004-04-25.log", "2004-07-26.log", "2004-10-26.log",
"2004-01-25.log", "2004-04-26.log", "2004-07-27.log", "2004-10-27.log",
"2004-01-26.log", "2004-04-27.log", "2004-07-28.log", "2004-10-28.log",
"2004-01-27.log", "2004-04-28.log", "2004-07-29.log", "2004-10-29.log",
"2004-01-28.log", "2004-04-29.log", "2004-07-30.log", "2004-10-30.log",
"2004-01-29.log", "2004-04-30.log", "2004-07-31.log", "2004-10-31.log",
"2004-01-30.log", "2004-05-01.log", "2004-08-01.log", "2004-11-01.log",
"2004-01-31.log", "2004-05-02.log", "2004-08-02.log", "2004-11-02.log",
"2004-02-01.log", "2004-05-03.log", "2004-08-03.log", "2004-11-03.log",
"2004-02-02.log", "2004-05-04.log", "2004-08-04.log", "2004-11-04.log",
"2004-02-03.log", "2004-05-05.log", "2004-08-05.log", "2004-11-05.log",
"2004-02-04.log", "2004-05-06.log", "2004-08-06.log", "2004-11-06.log",
"2004-02-05.log", "2004-05-07.log", "2004-08-07.log", "2004-11-07.log",
"2004-02-06.log", "2004-05-08.log", "2004-08-08.log", "2004-11-08.log",
"2004-02-07.log", "2004-05-09.log", "2004-08-09.log", "2004-11-09.log",
"2004-02-08.log", "2004-05-10.log", "2004-08-10.log", "2004-11-10.log",
"2004-02-09.log", "2004-05-11.log", "2004-08-11.log", "2004-11-11.log",
"2004-02-10.log", "2004-05-12.log", "2004-08-12.log", "2004-11-12.log",
"2004-02-11.log", "2004-05-13.log", "2004-08-13.log", "2004-11-13.log",
"2004-02-12.log", "2004-05-14.log", "2004-08-14.log", "2004-11-14.log",
"2004-02-13.log", "2004-05-15.log", "2004-08-15.log", "2004-11-15.log",
"2004-02-14.log", "2004-05-16.log", "2004-08-16.log", "2004-11-16.log",
"2004-02-15.log", "2004-05-17.log", "2004-08-17.log", "2004-11-17.log",
"2004-02-16.log", "2004-05-18.log", "2004-08-18.log", "2004-11-18.log",
"2004-02-17.log", "2004-05-19.log", "2004-08-19.log", "2004-11-19.log",
"2004-02-18.log", "2004-05-20.log", "2004-08-20.log", "2004-11-20.log",
"2004-02-19.log", "2004-05-21.log", "2004-08-21.log", "2004-11-21.log",
"2004-02-20.log", "2004-05-22.log", "2004-08-22.log", "2004-11-22.log",
"2004-02-21.log", "2004-05-23.log", "2004-08-23.log", "2004-11-23.log",
"2004-02-22.log", "2004-05-24.log", "2004-08-24.log", "2004-11-24.log",
"2004-02-23.log", "2004-05-25.log", "2004-08-25.log", "2004-11-25.log",
"2004-02-24.log", "2004-05-26.log", "2004-08-26.log", "2004-11-26.log",
"2004-02-25.log", "2004-05-27.log", "2004-08-27.log", "2004-11-27.log",
"2004-02-26.log", "2004-05-28.log", "2004-08-28.log", "2004-11-28.log",
"2004-02-27.log", "2004-05-29.log", "2004-08-29.log", "2004-11-29.log",
"2004-02-28.log", "2004-05-30.log", "2004-08-30.log", "2004-11-30.log",
"2004-02-29.log", "2004-05-31.log", "2004-08-31.log", "2004-12-01.log",
"2004-03-01.log", "2004-06-01.log", "2004-09-01.log", "2004-12-02.log",
"2004-03-02.log", "2004-06-02.log", "2004-09-02.log", "2004-12-03.log",
"2004-03-03.log", "2004-06-03.log", "2004-09-03.log", "2004-12-04.log",
"2004-03-04.log", "2004-06-04.log", "2004-09-04.log", "2004-12-05.log",
"2004-03-05.log", "2004-06-05.log", "2004-09-05.log", "2004-12-06.log",
"2004-03-06.log", "2004-06-06.log", "2004-09-06.log", "2004-12-07.log",
"2004-03-07.log", "2004-06-07.log", "2004-09-07.log", "2004-12-08.log",
"2004-03-08.log", "2004-06-08.log", "2004-09-08.log", "2004-12-09.log",
"2004-03-09.log", "2004-06-09.log", "2004-09-09.log", "2004-12-10.log",
"2004-03-10.log", "2004-06-10.log", "2004-09-10.log", "2004-12-11.log",
"2004-03-11.log", "2004-06-11.log", "2004-09-11.log", "2004-12-12.log",
"2004-03-12.log", "2004-06-12.log", "2004-09-12.log", "2004-12-13.log",
"2004-03-13.log", "2004-06-13.log", "2004-09-13.log", "2004-12-14.log",
"2004-03-14.log", "2004-06-14.log", "2004-09-14.log", "2004-12-15.log",
"2004-03-15.log", "2004-06-15.log", "2004-09-15.log", "2004-12-16.log",
"2004-03-16.log", "2004-06-16.log", "2004-09-16.log", "2004-12-17.log",
"2004-03-17.log", "2004-06-17.log", "2004-09-17.log", "2004-12-18.log",
"2004-03-18.log", "2004-06-18.log", "2004-09-18.log", "2004-12-19.log",
"2004-03-19.log", "2004-06-19.log", "2004-09-19.log", "2004-12-20.log",
"2004-03-20.log", "2004-06-20.log", "2004-09-20.log", "2004-12-21.log",
"2004-03-21.log", "2004-06-21.log", "2004-09-21.log", "2004-12-22.log",
"2004-03-22.log", "2004-06-22.log", "2004-09-22.log", "2004-12-23.log",
"2004-03-23.log", "2004-06-23.log", "2004-09-23.log", "2004-12-24.log",
"2004-03-24.log", "2004-06-24.log", "2004-09-24.log", "2004-12-25.log",
"2004-03-25.log", "2004-06-25.log", "2004-09-25.log", "2004-12-26.log",
"2004-03-26.log", "2004-06-26.log", "2004-09-26.log", "2004-12-27.log",
"2004-03-27.log", "2004-06-27.log", "2004-09-27.log", "2004-12-28.log",
"2004-03-28.log", "2004-06-28.log", "2004-09-28.log", "2004-12-29.log",
"2004-03-29.log", "2004-06-29.log", "2004-09-29.log", "2004-12-30.log",
"2004-03-30.log", "2004-06-30.log", "2004-09-30.log", "2004-12-31.log",
"2004-03-31.log", "2004-07-01.log", "2004-10-01.log",
"2004-04-01.log", "2004-07-02.log", "2004-10-02.log",
        ]))
        self.MODELDATA_SAVE_FILE    = "modeldata.npz"
        self.MODEL_SAVE_FILE        = "testing_WIP_08.model.npz"

        self.modelData              = TTCModelData()

    def test_split_population(self):
        self.modelData.training_files, \
        self.modelData.validation_files, \
        self.modelData.testing_files = self.modelData.split_population(self.SAMPLE_FILES)

        # 2004 leap year
        self.assertEqual(366, len(self.modelData.training_files) +
                         len(self.modelData.validation_files) +
                         len(self.modelData.testing_files)
                         )
        self.assertTrue(len(self.modelData.training_files)   > len(self.modelData.validation_files))
        self.assertTrue(len(self.modelData.validation_files) > len(self.modelData.testing_files))
        self.assertTrue(len(self.modelData.testing_files)    > 0)


    def test_preprocess_sample_files(self):
        from BatchSample import BatchSample

        self.modelData.training_files, \
        self.modelData.validation_files, \
        self.modelData.testing_files = self.modelData.split_population(self.SAMPLE_FILES)

        X_pd_kwargs = {}
        self.modelData.training_samples   = self.modelData._preprocess_sample_files(
            self.modelData.training_files, "training samples",     **X_pd_kwargs
        )
        self.modelData.validation_samples = self.modelData._preprocess_sample_files(
            self.modelData.validation_files, "validation samples", **X_pd_kwargs
        )
        self.modelData.testing_samples    = self.modelData._preprocess_sample_files(
            self.modelData.testing_files, "testing samples",       **X_pd_kwargs
        )

        self.assertEqual(len(self.modelData.training_files), len(self.modelData.training_samples))
        self.assertEqual(len(self.modelData.validation_files), len(self.modelData.validation_samples))

        for idx, bs in enumerate(self.modelData.training_samples[:3]):  # remove the slice later
            self.assertIsInstance(bs, BatchSample)

            # This is only set right at the end of BatchSample.process_file(), so it's a proxy for
            # the BatchSample loading successfully.
            self.assertEqual(self.modelData.training_files[idx], bs.filepath_or_buffer)

        for idx, bs in enumerate(self.modelData.validation_samples[:3]):  # remove the slice later
            self.assertIsInstance(bs, BatchSample)

            # filepath_or_buffer is only set right at the end of BatchSample.process_file(), so it's a proxy for
            # the BatchSample loading successfully.
            self.assertEqual(self.modelData.validation_files[idx], bs.filepath_or_buffer)

    def test_convert_to_numpy(self):
        """
        BatchSample deals in pandas.DataFrames
        TTCModelData deali in numpy.adarrays
        """

        self.modelData.training_files, \
        self.modelData.validation_files, \
        self.modelData.testing_files = self.modelData.split_population(self.SAMPLE_FILES)

        X_pd_kwargs = {}
        # validation_samples is big enough population for unit tests
        self.modelData.validation_samples = self.modelData._preprocess_sample_files(self.modelData.validation_files, "validation samples", **X_pd_kwargs)

        self.modelData._uniform_features()
        self.modelData._fit_x_scaler()

        X_pop, y_pop = self.modelData.convert_to_numpy(self.modelData.validation_samples)

        self.assertIsInstance(X_pop, np.ndarray)
        self.assertIsInstance(y_pop, np.ndarray)
        self.assertEqual((4015, 55, 52), X_pop.shape)  # (nb_samples, timesteps, features )
        self.assertEqual((4015,),        y_pop.shape)  # (nb_samples)

        #ToDo: More tests to make sure the array is np.floats, no nested lists / arrays/ objects



    def test_load_raw_sample_files(self):
        self.modelData.load_raw_sample_files(self.SAMPLE_FILES)

        from BatchSample import BatchSample
        self.assertIsInstance(self.modelData.training_samples[0],   BatchSample)
        self.assertIsInstance(self.modelData.validation_samples[0], BatchSample)
        self.assertIsInstance(self.modelData.testing_samples[0],    BatchSample)

        # Make sure training_samples[0] is not the same thing as training_samples[-1]
        self.assertTrue(len(self.modelData.training_samples)   > 1 )
        self.assertTrue(len(self.modelData.validation_samples) > 1)
        # Make sure padding has worked
        self.assertEqual(1,1)



    def test_save_np_data_file(self):
        import h5py
        import tempfile
        import os

        tmpfile = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        print("Temp file name : %s" % tmpfile.name)

        self.modelData.load_raw_sample_files(self.SAMPLE_FILES)
        self.modelData.save_np_data_file(tmpfile.name)

        with h5py.File(tmpfile.name, 'r') as store:
            self.assertEqual( ['dataframes', 'members', 'numpy'], list(store.keys()))

            self.assertEqual( ['X_test',
                              'X_train',
                              'X_validation',
                              'y_test',
                              'y_train',
                              'y_validation'], list(store['numpy'].keys())
                              )
        #os.unlink(tmpfile.name)

    def test_load_np_data_file(self):
        import h5py
        import tempfile
        import os

        tmpfile = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        print("Temp file name : %s" % tmpfile.name)

        self.modelData.load_raw_sample_files(self.SAMPLE_FILES)
        self.modelData.save_np_data_file(tmpfile.name)

        from TTCModelData import TTCModelData
        testingModelData = TTCModelData()
        testingModelData.load_np_data_file(tmpfile.name)

        # Can't do this as the populations contain randomised BatchSamples each load#
        #self.assertTrue(np.allclose( testingModelData.X_train,      self.modelData.X_train, ), 'X_train[-1,-1,:]: %s and %s' % \
        #                (str(testingModelData.X_train[-1,1,:]),   str(self.modelData.X_train[-1,-1,:])))
        #self.assertTrue(np.allclose( testingModelData.y_train,      self.modelData.y_train, ))
        # ...
        
        # Going with min,max&sum of the combined sample populations

        leftData  = np.concatenate((testingModelData.X_train,  testingModelData.X_validation, testingModelData.X_test))
        rightData = np.concatenate((self.modelData.X_train,   self.modelData.X_validation, self.modelData.X_test))

        self.assertTrue(np.allclose(np.nanmin(leftData), np.nanmin(rightData)), "nanmin() differences %s and %s" % (
            np.nanmin(leftData), np.nanmin(rightData))
        )

        self.assertTrue(np.allclose(np.nanmax(leftData), np.nanmax(rightData)), "nanmax() differences %s and %s" % (
            np.nanmax(leftData), np.nanmax(rightData))
        )

        self.assertTrue(np.allclose(np.nansum(leftData), np.nansum(rightData)), "nansum() differences %s and %s" % (
            np.nansum(leftData), np.nansum(rightData))
        )


        # Same for Y
        leftData  = np.concatenate((testingModelData.y_train,  testingModelData.y_validation, testingModelData.y_test))
        rightData = np.concatenate((self.modelData.y_train,   self.modelData.y_validation, self.modelData.y_test))

        self.assertTrue(np.allclose(np.nanmin(leftData), np.nanmin(rightData)), "nanmin() differences %s and %s" % (
            np.nanmin(leftData), np.nanmin(rightData))
        )

        self.assertTrue(np.allclose(np.nanmax(leftData), np.nanmax(rightData)), "nanmax() differences %s and %s" % (
            np.nanmax(leftData), np.nanmax(rightData))
        )

        self.assertTrue(np.allclose(np.nansum(leftData), np.nansum(rightData)), "nansum() differences %s and %s" % (
            np.nansum(leftData), np.nansum(rightData))
        )



        #os.unlink(tmpfile.name)



    def test_get_shaped_features_X(self):
        pass


    def test_get_shaped_y(self):
        pass
