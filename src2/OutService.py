# Python 3.4+
from abc import ABC, abstractmethod


class OutService(ABC):

    @abstractmethod
    def printPredictions(self, predictions, ml_model="", sample_file="", *print_args, **print_kwargs):
        pass