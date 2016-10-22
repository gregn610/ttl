import unittest
import numpy as np
import pandas as pd

from BatchSample import BatchSample
from xmlunittest import XmlTestCase

class TestOutServiceXml(XmlTestCase):

    def setUp(self):
        from OutServiceXml import OutServiceXml

        self.xml_services = OutServiceXml()


    def test_print_predictions_xml(self):
        import io
        from DebugBatchSample import DebugBatchSample
        from contextlib import redirect_stdout
        import xml.etree.ElementTree as ET

        predictions = [ pd.date_range('2011-11-11', periods=7, freq='30min').tolist(), ]


        lleft = """<?xml version='1.0' encoding='utf-8'?>
<ttc>
  <predictions ml_model="unit_testing.ttc">
    <sample file="../../data/population_v2.1/2007-11-11.log">
      <pr __dbg_realtime_finish="2007-11-11T04:56:56.000000000">2011-11-11 00:00:00</pr>
      <pr __dbg_realtime_finish="2007-11-11T04:56:56.000000000">2011-11-11 00:30:00</pr>
      <pr __dbg_realtime_finish="2007-11-11T04:56:56.000000000">2011-11-11 01:00:00</pr>
      <pr __dbg_realtime_finish="2007-11-11T04:56:56.000000000">2011-11-11 01:30:00</pr>
      <pr __dbg_realtime_finish="2007-11-11T05:01:42.000000000">2011-11-11 02:00:00</pr>
      <pr __dbg_realtime_finish="2007-11-11T05:01:42.000000000">2011-11-11 02:30:00</pr>
      <pr __dbg_realtime_finish="2007-11-11T05:01:42.000000000">2011-11-11 03:00:00</pr>
    </sample>
  </predictions>
</ttc>

"""

        # thx: http://stackoverflow.com/a/22434594/266387
        with io.StringIO() as buf, redirect_stdout(buf):

            dbs = BatchSample()
            dbs.process_file('../../data/population_v2.1/2007-11-11.log', 0, 1)
            dbs.__class__ = DebugBatchSample
            dbs._conversion_from_BatchSample()


            self.xml_services.printPredictions(batch_samples = [dbs, ],
                                               predictions   = predictions,
                                               model_descr   = 'unit_testing.ttc')
            rright_byte_str = buf.getvalue()
            #rright = ET.fromstring(rright_byte_str)

        self.assertEqual(lleft, rright_byte_str)
#            root = self.assertXmlDocument(rright_byte_str)


