import unittest
import numpy as np
import pandas as pd


class TestOutServiceXml(unittest.TestCase):

    def setUp(self):
        from OutServiceXml import OutServiceXml

        self.xml_services = OutServiceXml()


    def test_print_predictions(self):
        import io
        from contextlib import redirect_stdout

        predictions = pd.date_range('2011-11-11', periods=7, freq='30min')
        lleft = """
<ttc encoding="UTF-8">
  <predictions ml_model="test_model.ttc" sample_file="test123.log">
    <p>2011-11-11T00:00:00</p>
    <p>2011-11-11T00:30:00</p>
    <p>2011-11-11T01:00:00</p>
    <p>2011-11-11T01:30:00</p>
    <p>2011-11-11T02:00:00</p>
    <p>2011-11-11T02:30:00</p>
    <p>2011-11-11T03:00:00</p>
  </predictions>
</ttc>

"""

        # thx: http://stackoverflow.com/a/22434594/266387
        with io.StringIO() as buf, redirect_stdout(buf):
            self.xml_services.printPredictions(predictions, ml_model='test_model.ttc', sample_file='test123.log' )
            rright = '\n' + buf.getvalue()

        self.assertEqual(lleft, rright)


