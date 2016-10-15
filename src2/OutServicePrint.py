from OutService import OutService

class OutServicePrint(OutService):

    def printPredictions(self, predictions, ml_model="", sample_file="", *print_args, **print_kwargs ):
        """

        :param predictions: a list of pandas Timestamps
        :param print_args: as for normal print()
        :param print_kwargs: as for normal print()
        :return:
        """

        print(str(predictions), *print_args, **print_kwargs)