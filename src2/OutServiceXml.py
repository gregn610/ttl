import sys
from lxml import etree
import pandas as pd

from DebugBatchSample import DebugBatchSample
import OutService

class OutServiceXml(OutService.OutService):

    def printPredictions(self, batch_samples, predictions, model_descr, *print_args, **print_kwargs):
        """

        :param batch_samples:
        :param predictions:
        :param model_descr:
        :param print_args:
        :param print_kwargs:
        :return:
        """
        assert len(predictions) == len(batch_samples)

        e_root  = etree.Element("ttc")
        e_predictions = etree.SubElement(e_root, "predictions")
        if model_descr:
            e_predictions.set("ml_model", model_descr)

        for idx, prd in enumerate(predictions):
            e_sample = etree.SubElement(e_predictions, "sample")
            e_sample.set("file", batch_samples[idx].filepath_or_buffer)

            if isinstance(batch_samples[idx], DebugBatchSample):
                for idx_p, pred in enumerate(prd):
                    e_pr = etree.SubElement(e_sample, "pr")
                    e_pr.text = "%s" % pred.strftime("%Y-%m-%d %H:%M:%S")
                    e_pr.set("__dbg_realtime_finish", str(
                        batch_samples[idx].debug_dfy['__dbg_realtime_finish'].values[idx_p]
                        )
                    )
            else:
                for pred in predictions[idx]:
                    e_pr = etree.SubElement(e_sample, "pr")
                    e_pr.text = "%s" % pred.strftime("%Y-%m-%d %H:%M:%S")


        print(etree.tostring(e_root,
                             pretty_print    = True,
                             method          = 'xml',
                             encoding        = 'utf-8',
                             xml_declaration = True,
                             ).decode("utf-8") ,
              *print_args, **print_kwargs)

        #sys.stdout.buffer.write(etree.tostring(e_root,
        #                         pretty_print=True,
        #                         xml_declaration=True)
        #                        )