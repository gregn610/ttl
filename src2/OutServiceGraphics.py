import math

from DebugBatchSample import DebugBatchSample
from OutService import OutService
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class OutServiceGraphics(OutService):

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
        plt.close('all')

        f, axarr = plt.subplots(len(batch_samples), figsize=(12, 6), sharex=True, squeeze=False)
        for idx, pr in enumerate(predictions):
            axarr[idx, 0].set_title(batch_samples[idx].filepath_or_buffer)

            if idx == 0:
                axarr[idx, 0].set_ylabel('Time')
                axarr[idx, 0].set_xlabel('Log Event')
                axarr[idx, 0].legend()


            axarr[idx, 0].plot(batch_samples[idx].dfX[batch_samples[idx].event_time_col].values,
                            label='Log Event',
                            marker='o'
                            )
            axarr[idx, 0].plot(predictions[idx],
                            label='Predictions'
                            )

            if isinstance(batch_samples[idx], DebugBatchSample):
                axarr[idx, 0].plot(batch_samples[idx].debug_dfy['__dbg_realtime_finish'].values,
                                label='y realtime finish'
                                )

        plt.tight_layout()

        #if (data_filename is not None):
        #    plt.savefig(data_filename)
        #    plt.close()
        #else:
        plt.show()





    def printEvaluation(self, batch_sample, predictions, data_filename=None):
        plt.figure(figsize=(12, 6))

        plt.ylabel('Time')
        plt.xlabel('Log Event')
        plt.legend()
        plt.title('Evaluate Prediction')

        dfX = batch_sample.dfX
        dfy = batch_sample.dfy
        debug_dfy = batch_sample.debug_dfy
        debug_dfX = batch_sample.debug_dfX


        plt.plot(dfX[batch_sample.event_time_col].values,  label='X Log Event')
        plt.plot(debug_dfy['__dbg_realtime_finish'].values,label='y realtime finish')
        plt.plot(predictions,                              label='Predictions')

        if (data_filename is not None):
            plt.savefig(data_filename)
            plt.close()
        else:
            plt.show()
