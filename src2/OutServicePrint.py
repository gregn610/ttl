from DebugBatchSample import DebugBatchSample
from OutService import OutService

class OutServicePrint(OutService):

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

        if model_descr:
            print("ml_model: %s" % model_descr, *print_args, **print_kwargs)

        for idx, prd in enumerate(predictions):
            print("sample_file: %s" % batch_samples[idx].filepath_or_buffer, *print_args, **print_kwargs)
            if isinstance(batch_samples[idx], DebugBatchSample):
                for idx_p, pred in enumerate(predictions):
                    print("%05d: %s,     --- Debug: %s" % (
                        idx,
                        pred.isoformat(),
                        str(batch_samples[idx].debug_dfy['__dbg_realtime_finish'].values[idx_p])),
                          *print_args, **print_kwargs)
            else:
                for idx_p, pred in enumerate(prd):
                    print("%05d: %s" % (idx_p, pred.isoformat()), *print_args, **print_kwargs)

        print("\n", *print_args, **print_kwargs)
