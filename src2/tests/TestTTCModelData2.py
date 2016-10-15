import unittest
import numpy as np
import pandas as pd
import os

"""
These are TTCModelData tests that recycle the same X_pop & y_pop
"""
from Consts import CONST_EMPTY


class TestTTCModelData2(unittest.TestCase):
    def setUp(self):

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

        self.modelData = TTCModelData()
        self.modelData.load_raw_sample_files(self.SAMPLE_FILES)

    def test_unstash_xscaler(self):
        """Does the same come out of unstash() as went in to stash()"""
        import tempfile

        tmpfile = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        print("Temp file name : %s" % tmpfile.name)

        self.modelData.save_np_data_file(tmpfile.name)

        from TTCModelData import TTCModelData

        modelData2 = TTCModelData()
        modelData2.unstash_xscaler(tmpfile.name)

        self.assertEqual( self.modelData.max_timesteps,     modelData2.max_timesteps     )
        self.assertEqual( self.modelData.complete_features, modelData2.complete_features )

        self.assertTrue( np.allclose(self.modelData.xscaler.min_        , modelData2.xscaler.min_        ))
        self.assertTrue( np.allclose(self.modelData.xscaler.scale_      , modelData2.xscaler.scale_      ))
        self.assertTrue( np.allclose(self.modelData.xscaler.data_min_   , modelData2.xscaler.data_min_   ))
        self.assertTrue( np.allclose(self.modelData.xscaler.data_max_   , modelData2.xscaler.data_max_   ))
        self.assertTrue( np.allclose(self.modelData.xscaler.data_range_ , modelData2.xscaler.data_range_ ))



    def test_xscaler_prediction(self):
        """Does the xscaler unpacked for predictions give the same results as the xscaler used
        for the original X_pop
        """
        from TTCModelData import TTCModelData
        import tempfile

        tmpfile = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
        print("Temp file name : %s" % tmpfile.name)

        self.modelData.save_np_data_file(tmpfile.name)

        modelData2 = TTCModelData()
        modelData2.unstash_xscaler(tmpfile.name)

        bs = self.modelData.training_samples[0]

        lleft  = self.modelData.xscaler.transform(bs.get_dfI_values())
        rright = modelData2.xscaler.transform(bs.get_dfI_values())

        # os.unlink(tmpfile.name)

        self.assertTrue(np.allclose(lleft, rright))


    def test_get_shaped_y(self):
        """If a batch_sample is smaller than self.max_timesteps ,then the Y should be padded with CONST_EMPTY"""

        from BatchSample import BatchSample

        bs2 = BatchSample()
        bs2.process_file(self.SAMPLE_FILES[0], 0, 1)
        bs2.dfX.drop(bs2.dfX.index[-20:], inplace=True)

        # this needs modelData.max_timesteps which is only set after some other files have been _homogenized()
        # Thats why it's in here and not the other TestTTCModelData
        y2 = self.modelData.get_shaped_y(bs2)
        lleft  =  np.ones(( 20 )) * CONST_EMPTY
        rright = y2[-20:]
        self.assertTrue(np.allclose(lleft, rright ), "Left:%s\nRight:%s" %(str(lleft), str(rright)))

    def test_scaler_roundtrip(self):
        """Make sure that values round-trip through the xscaler without changing
        Needs love, not a strong test as written here, would be better inverse_transforming() X_training[] back
         to bs.dfX"""

        bs = self.modelData.training_samples[0]
        scaled = self.modelData.xscaler.transform(bs.get_dfI_values())

        self.assertTrue(np.allclose(bs.get_dfI_values(), self.modelData.xscaler.inverse_transform( scaled)))

    def test_something(self):
        """There's something not cool with X_training. Make sure it's right"""
        for idx, bs in enumerate(self.modelData.training_samples):
            _dbg_lleft_file = bs.filepath_or_buffer
            lleft = bs.get_dfI_values()
            left_dfx_series = bs.dfX[bs.event_time_col]

            # should be the last exploded nb_sample for that batch_sample
            x_sliced = self.modelData.X_train[(idx+1)*self.modelData.max_timesteps -1,:,:]
            sliced_unscaled = self.modelData.xscaler.inverse_transform(x_sliced)

            self.assertTrue(np.allclose(lleft, sliced_unscaled))
            self.assertTrue(np.allclose(lleft[-1,-1], sliced_unscaled[-1,-1]),'lleft:%s\nsliced_unscaled:%s'%(
                    str(lleft[-1,-1]), str(sliced_unscaled[-1,-1])
                )
            )
            reg_left  = bs.regularizedToDateTime(bs.event_time_col, lleft[-1,-1])
            reg_right = bs.regularizedToDateTime(bs.event_time_col, sliced_unscaled[-1,-1])
            print('reg_left: %s\nreg_right:%s'% (str(reg_left), str(reg_right)))
            self.assertEqual(reg_left, reg_right)
            #self.assertEqual(left_dfx_series[-1], reg_right)
            # ToDo: also compare to y


    def test_homogenize_features(self):
        for idx in (range(1, len(self.modelData.training_samples))):
            first = self.modelData.training_samples[idx - 1]._get_dfI()
            second = self.modelData.training_samples[idx]._get_dfI()
            assert (np.array_equal(first.columns.values, second.columns.values))
        print('All asserts passed')