# Python 3.4+
from abc import ABC, abstractmethod


class OutService(ABC):

    @abstractmethod
    def printPredictions(self, batch_samples, predictions, model_descr, *print_args, **print_kwargs):
        pass