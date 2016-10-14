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


