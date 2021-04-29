"""
Build a NNData class that will help us better manage our training and
testing data.
"""
from collections import deque
from enum import Enum
import numpy as np
import random
from math import floor


class DataMismatchError(Exception):
    """
    This custom exception is raised if our features and labels do not match
    in length.
    """

    def __init__(self, message):
        self.message = message


class NNData:
    class Order(Enum):
        """
        Define whether the training data is presented in the same order to NN
        each time, or in random order.
        """
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """
        This enum will help us identify whether we are requesting training set
        data or testing set data.
        """
        TRAIN = 0
        TEST = 1

    def __init__(self,
                 features: list = None,
                 labels: list = None,
                 train_factor: float = .9):
        """
        Class built to help us efficiently manage our training and testing
        data.
        :param features:
        :param labels:
        :param train_factor:
        """

        if features is None:
            self._features = []
        else:
            try:
                self.load_data(features=features, labels=labels)
            except (DataMismatchError, ValueError):
                self._features = None
                self._labels = None

        if labels is None:
            self._labels = []
        else:
            try:
                self.load_data(features=features, labels=labels)
            except (DataMismatchError, ValueError):
                self._features = None
                self._labels = None

        # Set Training Factor
        self._train_factor = self.percentage_limiter(train_factor)

        # Set Train and Test indices
        self._train_indices = []
        self._test_indices = []

        self._train_pool = deque()
        self._test_pool = deque()
        self.split_set()

    @staticmethod
    def percentage_limiter(percentage: float):
        """
        Limits floats to a range between 0 and 1.
        :param percentage:
        :return:
        """
        if percentage > 1:
            return 1
        elif percentage < 0:
            return 0
        else:
            return percentage

    def split_set(self, new_train_factor: float = None):
        """
        Splits features and labels based on training factor. training
        factor when creating the instance is the default, otherwise it
        is replaced by the new one given.
        :param new_train_factor:
        :return:
        """
        if self._features is None:
            self._train_indices = []
            self._test_indices = []
            return

        if new_train_factor is not None:
            self._train_factor = self.percentage_limiter(new_train_factor)

        size = len(self._features)
        available_indices = [i for i in range(0, size)]
        number_of_training_examples = floor(size * self._train_factor)
        indirect_train = []
        while len(indirect_train) < number_of_training_examples:
            r = random.randint(available_indices[0], available_indices[-1])
            if r in indirect_train:
                continue
            else:
                indirect_train.append(r)

        indirect_test = []
        for i in available_indices:
            if i in indirect_train:
                pass
            else:
                indirect_test.append(i)

        indirect_train.sort()
        indirect_test.sort()

        self._train_indices = indirect_train
        self._test_indices = indirect_test

    def prime_data(self, target_set=None, order=None):
        """
        Load one or both deques to be used as indirect indices.
        :param target_set:
        :param order:
        :return:
        """

        # If target set is set to train
        if target_set == NNData.Set.TRAIN:
            if order == NNData.Order.RANDOM:
                self._train_pool.extend(self._train_indices)
                random.shuffle(self._train_pool)
            else:
                self._train_pool.extend(self._train_indices)

        # If target set is test
        elif target_set == NNData.Set.TEST:
            if order == NNData.Order.RANDOM:
                self._test_pool.extend(self._test_indices)
                random.shuffle(self._test_pool)
            else:
                self._test_pool.extend(self._test_indices)

        # If target set to none, but order is random
        elif target_set is None and order == NNData.Order.RANDOM:
            self._train_pool.extend(self._train_indices)
            self._test_pool.extend(self._test_indices)
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)

        # If both target set and order are set to None
        else:
            self._train_pool.extend(self._train_indices)
            self._test_pool.extend(self._test_indices)

    def get_one_item(self, target_set=None):
        """
        This method will return exactly one feature/label pair as a tuple.
        :param target_set:
        :return:
        """
        try:
            if target_set is None or target_set == NNData.Set.TRAIN:
                train_pool_index = self._train_pool.popleft()
                pair = (self._features[train_pool_index],
                        self._labels[train_pool_index])
                return pair
            else:
                test_pool_index = self._test_pool.popleft()
                pair = (self._features[test_pool_index],
                        self._labels[test_pool_index])
                return pair
        except IndexError:
            print("No more indices left to chose from target_set")
            return None

    def number_of_samples(self, target_set=None):
        """
        Returns total number of testing or training examples
        based on the target_set parameter. If target_set is None, then the
        method returns the total number of combined examples (training + test).
        :param target_set:
        :return:
        """

        if target_set == NNData.Set.TRAIN:
            return len(self._train_indices)
        elif target_set == NNData.Set.TEST:
            return len(self._test_indices)
        else:
            return len(self._train_indices) + len(self._test_indices)

    def pool_is_empty(self, target_set=None):
        """
        Returns True if specified target_set is empty, otherwise returns false.
        When target_set is None, then defaults to evaluating the training pool.
        :param target_set:
        :return:
        """
        if target_set == NNData.Set.TRAIN:
            if len(self._train_pool) == 0:
                return True
            else:
                return False
        elif target_set == NNData.Set.TEST:
            if len(self._test_pool) == 0:
                return True
            else:
                return False
        else:
            if len(self._train_pool) == 0:
                return True
            else:
                return False

    def load_data(self,
                  features=None,
                  labels=None):
        """
        Load features and labels into class instance.
        :param features:
        :param labels:
        :return:
        """

        if features is None:
            self._features = None
            self._labels = None
            return

        # Make sure features and labels are the same size.
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


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode:
    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}

        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [],
                           MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        pass

    def _process_new_neighbor(self, node, side: Enum):
        pass

    def reset_neighbors(self, nodes: list, side: Enum):
        pass


class Neurode(MultiLinkNode):
    def __init__(self, node_type: LayerType, learning_rate: float =.05):
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}






def load_xor():
    """
    Defines an XOR feature and label set and
    creates a test object of our class with these sets.
    :return:
    """
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]

    test_xor_object = NNData(features=features,
                             labels=labels,
                             train_factor=1)


def unit_test_assignment1():
    """
    Test constructor and methods from NNData class to make sure that they are
    up to spec.
    :return:
    """

    # Define good and bad feature/label sets
    bad_features_lengths = [[0, 0], [1, 0]]
    bad_features_values = [[0, 0], [1, "Cat"], [0, 1], [1, "Green"]]
    bad_label_value = [[0], [1], ["Cat"], [0]]
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]

    # Test DataMismatchError when calling load_data
    try:
        test1 = NNData()
        test1.load_data(features=bad_features_lengths, labels=labels)
        print("FAIL: NNData.load_data() did not raise DataMismatchError.")
    except DataMismatchError:
        print("SUCCESS: NNData.load_data() raises DataMismatchError when "
              "features and labels have different lengths.")

    # Test ValueError when non-float values called during method 'load_data'.
    try:
        test2 = NNData()
        test2.load_data(features=bad_features_values, labels=labels)
        print("FAIL: NNData.load_data() did not raise ValueError.")
    except ValueError:
        print("SUCCESS: NNData.load_data() raises ValueError when "
              "either features or labels include non-float values.")

    # Verify that if invalid data are passed to the constructor that labels and
    # features are set to None.
    test3 = NNData(bad_features_values, labels, .8)
    if test3._labels is None and test3._features is None:
        print("SUCCESS: invalid feature data passed to the constructor sets "
              "features and labels to None")
    else:
        print("FAIL: features or labels not set to None after invalid data "
              "passed to the constructor")

    test4 = NNData(features, bad_label_value, .8)
    if test4._labels is None and test4._features is None:
        print("SUCCESS: invalid label data passed to the constructor sets "
              "features and labels to None")
    else:
        print("FAIL: features or labels not set to None after invalid data "
              "passed to the constructor")

    # Verify that training factor is limited to range between 0 and 1
    test5 = NNData(features, labels, -5)
    if test5._train_factor == 0:
        print("SUCCESS: training factor limited to zero when negative value "
              "passed")
    else:
        print("FAIL: training factor not limited to zero when negative value "
              "passed")

    test6 = NNData(features, labels, 5)
    if test6._train_factor == 1:
        print("SUCCESS: training factor limited to one when a value greater "
              " than 1 was passed")
    else:
        print("FAIL: training factor not limited to 1 when a value greater "
              " than 1 passed")


def unit_test():
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = [[i] for i in range(10)]
        y = x
        our_data_0 = NNData(x, y)
        x = [[i] for i in range(100)]
        y = x
        our_big_data = NNData(x, y, .5)

        # Try loading lists of different sizes
        y = [[1]]
        try:
            our_bad_data = NNData()
            our_bad_data.load_data(x, y)
            raise Exception
        except DataMismatchError:
            pass
        except:
            raise Exception

        # Create a dataset that can be used to make sure the
        # features and labels are not confused
        x = [[1.0], [2.0], [3.0], [4.0]]
        y = [[.1], [.2], [.3], [.4]]
        our_data_1 = NNData(x, y, .5)

    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of examples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        print(f"Train Indices:{our_data_0._train_indices}")
        print(f"Test Indices:{our_data_0._test_indices}")

        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True

    # Make sure prime_data sets up the deques correctly, whether
    # sequential or random.
    try:
        our_data_0.prime_data(order=NNData.Order.SEQUENTIAL)
        assert len(our_data_0._train_pool) == 3
        assert len(our_data_0._test_pool) == 7
        assert our_data_0._train_indices == list(our_data_0._train_pool)
        assert our_data_0._test_indices == list(our_data_0._test_pool)
        our_big_data.prime_data(order=NNData.Order.RANDOM)
        assert our_big_data._train_indices != list(our_big_data._train_pool)
        assert our_big_data._test_indices != list(our_big_data._test_pool)
    except:
        print("There are errors that likely come from prime_data")
        errors = True

    # Make sure get_one_item is returning the correct values, and
    # that pool_is_empty functions correctly.
    try:
        our_data_1.prime_data(order=NNData.Order.SEQUENTIAL)
        my_x_list = []
        my_y_list = []
        while not our_data_1.pool_is_empty():
            example = our_data_1.get_one_item()
            my_x_list.append(example[0])
            my_y_list.append(example[1])
        assert len(my_x_list) == 2
        assert my_x_list != my_y_list
        my_matched_x_list = [i * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        while not our_data_1.pool_is_empty(our_data_1.Set.TEST):
            example = our_data_1.get_one_item(our_data_1.Set.TEST)
            my_x_list.append(example[0])
            my_y_list.append(example[1])
        assert my_x_list != my_y_list
        my_matched_x_list = [i * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        assert set(i[0] for i in my_x_list) == set(i[0] for i in x)
        assert set(i[0] for i in my_y_list) == set(i[0] for i in y)
    except:
        print("There are errors that may come from prime_data, but could "
              "be from another method")
        errors = True

    # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


if __name__ == '__main__':
    unit_test()
