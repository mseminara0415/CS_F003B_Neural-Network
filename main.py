from enum import Enum
import unittest
import numpy as np


class DataMismatchError(Exception):
    def __init__(self, message):
        self.message = message


class NNData:
    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        if percentage > 1:
            return 1
        elif percentage < 0:
            return 0
        else:
            return percentage

    def __init__(self,
                 features: list = None,
                 labels: list = None,
                 train_factor: float = .9):

        if features is None:
            self._features = []
        else:
            self._features = features

        if labels is None:
            self._labels = []
        else:
            self._labels = labels

        self._train_factor = self.percentage_limiter(train_factor)

    def load_data(self,
                  features=None,
                  labels=None):
        """
        Load features and labels
        :param features:
        :param labels:
        :return:
        """

        if features is None:
            self._features = None
            self._labels = None
            return

        # Make sure features and labels match lengthwise
        length_features = len(features)
        length_labels = len(labels)
        if length_features != length_labels:
            self._features = None
            self._labels = None
            raise DataMismatchError("Features and Labels must be of "
                                    "the same length.")

        else:
            try:
                self._features = np.array(features, dtype=float)
                self._labels = np.array(labels, dtype=float)
            except ValueError:
                self._features = None
                self._labels = None
                raise ValueError


def load_xor():
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]

    test_object = NNData(features=features,
                         labels=labels,
                         train_factor=1)


def unit_test():
    features1 = [[0, 0], [1, 0], [0, "CAT"], [1, "1"]]
    bad_features = [[0, 0], [1, 0]]
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]

    test = NNData()
    test.load_data(features=features, labels=labels)


if __name__ == '__main__':
    unit_test()
