import math

from OutService import OutService
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class OutServiceGraphics(OutService):

    def printPredictions(self, predictions, ml_model="", sample_file="", *print_args, **print_kwargs ):
        """

        :param predictions: a list of pandas Timestamps
        :param print_args: as for normal print()
        :param print_kwargs: as for normal print()
        :return:
        """
        raise NotImplementedError

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
