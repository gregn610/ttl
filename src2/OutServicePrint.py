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
            for pred in predictions[0]:
                print("%05d: %s" % (idx, pred.isoformat()), *print_args, **print_kwargs)

        print("\n", *print_args, **print_kwargs)
