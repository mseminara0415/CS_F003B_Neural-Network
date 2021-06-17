"""
Build a NNData class that will help us better manage our training and
testing data.
"""
from abc import ABC, abstractmethod
import collections
from enum import Enum
import numpy as np
import random
from math import floor
import math
import matplotlib.pyplot as plt


class DataMismatchError(Exception):
    """ Label and example lists have different lengths"""


class NNData:

    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(factor):
        return min(1, max(factor, 0))

    def __init__(self, features=None, labels=None, train_factor=.9):
        self._train_factor = NNData.percentage_limiter(train_factor)

        self._train_indices = []
        self._test_indices = []
        self._train_pool = collections.deque()
        self._test_pool = collections.deque()
        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            self._features = None
            self._labels = None
            return
        self.split_set()

    def load_data(self, features=None, labels=None):
        if features is None:
            features = []
            labels = []

        if len(features) != len(labels):
            raise DataMismatchError("Label and example lists have "
                                    "different lengths")

        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = []
            self._labels = []
            raise ValueError("Label and example lists must be homogeneous "
                             "and numeric lists of lists")

    def split_set(self, new_train_factor=None):
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        total_set_size = len(self._features)
        train_set_size = math.floor(total_set_size * self._train_factor)
        self._train_indices = random.sample(range(total_set_size),
                                            train_set_size)
        self._test_indices = list(set(range(total_set_size)) -
                                  set(self._train_indices))
        self._train_indices.sort()
        self._test_indices.sort()

    def get_one_item(self, target_set=None):
        try:
            if target_set == NNData.Set.TEST:
                index = self._test_pool.popleft()
            else:
                index = self._train_pool.popleft()
            return self._features[index], self._labels[index]
        except IndexError:
            return None

    def number_of_samples(self, target_set=None):
        if target_set is NNData.Set.TEST:
            return len(self._test_indices)
        elif target_set is NNData.Set.TRAIN:
            return len(self._train_indices)
        else:
            return len(self._features)

    def pool_is_empty(self, target_set=None):
        if target_set is NNData.Set.TEST:
            return len(self._test_pool) == 0
        else:
            return len(self._train_pool) == 0

    def prime_data(self, my_set=None, order=None):
        if order is None:
            order = NNData.Order.SEQUENTIAL
        if my_set is not NNData.Set.TRAIN:
            test_indices_temp = list(self._test_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(test_indices_temp)
            self._test_pool = collections.deque(test_indices_temp)
        if my_set is not NNData.Set.TEST:
            train_indices_temp = list(self._train_indices)
            if order == NNData.Order.RANDOM:
                random.shuffle(train_indices_temp)
            self._train_pool = collections.deque(train_indices_temp)


class LayerType(Enum):
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2


class MultiLinkNode(ABC):

    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {side: 0 for side in self.Side}
        self._reference_value = {side: 0 for side in self.Side}
        self._neighbors = {side: [] for side in self.Side}

    def __str__(self):
        ret_str = "-->Node " + str(id(self)) + "\n"
        ret_str = ret_str + "   Input Nodes:\n"
        for key in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            ret_str = ret_str + "   " + str(id(key)) + "\n"
        ret_str = ret_str + "   Output Nodes\n"
        for key in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            ret_str = ret_str + "   " + str(id(key)) + "\n"
        return ret_str

    @abstractmethod
    def _process_new_neighbor(self, node, side: Side):
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = (1 << len(nodes)) - 1


class Neurode(MultiLinkNode):

    def __init__(self, node_type, learning_rate=.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    def _process_new_neighbor(self, node, side: MultiLinkNode.Side):
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side: MultiLinkNode.Side):
        node_number = self._neighbors[side].index(node)
        self._reporting_nodes[side] =\
            self._reporting_nodes[side] | 1 << node_number
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._learning_rate = learning_rate

    def get_weight(self, node):
        return self._weights[node]


class FFNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        input_sum = 0
        for node, weight in self._weights.items():
            input_sum += node.value * weight
        self._value = self._sigmoid(input_sum)

    def _fire_downstream(self):
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, from_node):
        if self._check_in(from_node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value: float):
        self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class BPNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self):
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1.0 - value)

    def _calculate_delta(self, expected_value=None):
        if self._node_type == LayerType.OUTPUT:
            error = expected_value - self.value
            self._delta = error * self._sigmoid_derivative(self.value)
        else:
            self._delta = 0
            for neurode in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._delta += neurode.get_weight(self) * neurode.delta
            self._delta *= self._sigmoid_derivative(self.value)

    def set_expected(self, expected_value: float):
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def data_ready_downstream(self, from_node):
        if self._check_in(from_node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def adjust_weights(self, node, adjustment):
        self._weights[node] += adjustment

    def _update_weights(self):
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = node.learning_rate * node.delta * self.value
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class DLLNode:
    """ Node class for a DoublyLinkedList - not designed for
        general clients, so no accessors or exception raising """

    def __init__(self, data=None):
        self.prev = None
        self.next = None
        self.data = data


class DoublyLinkedList:

    class EmptyListError(Exception):
        pass

    def __init__(self):
        self._head = None
        self._tail = None
        self._current = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._current and self._current.next:
            ret_val = self._current.data
            self._current = self._current.next
            return ret_val
        raise StopIteration

    def move_forward(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current.next:
            self._current = self._current.next
        else:
            raise IndexError

    def move_back(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current.prev:
            self._current = self._current.prev
        else:
            raise IndexError

    def add_to_head(self, data):
        new_node = DLLNode(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        self._head = new_node
        if self._tail is None:
            self._tail = new_node
        self.reset_to_head()

    def remove_from_head(self):
        if not self._head:
            raise DoublyLinkedList.EmptyListError

        ret_val = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        else:
            self._tail = None
        self.reset_to_head()
        return ret_val

    def add_after_cur(self, data):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        new_node = DLLNode(data)
        new_node.prev = self._current
        new_node.next = self._current.next
        if self._current.next:
            self._current.next.prev = new_node
        self._current.next = new_node
        if self._tail == self._current:
            self._tail = new_node

    def remove_after_cur(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        if self._current == self._tail:
            raise IndexError
        ret_val = self._current.next.data
        if self._current.next == self._tail:
            self._tail = self._current
            self._current.next = None
        else:
            self._current.next = self._current.next.next
            self._current.next.prev = self._current
        return ret_val

    def reset_to_head(self):
        if not self._head:
            raise DoublyLinkedList.EmptyListError
        self._current = self._head

    def reset_to_tail(self):
        if not self._tail:
            raise DoublyLinkedList.EmptyListError
        self._current = self._tail

    def get_current_data(self):
        if not self._current:
            raise DoublyLinkedList.EmptyListError
        return self._current.data


class LayerList(DoublyLinkedList):

    def __init__(self, inputs, outputs):
        super().__init__()
        if inputs < 1 or outputs < 1:
            raise ValueError
        input_layer = [FFBPNeurode(LayerType.INPUT) for _ in range(inputs)]
        output_layer = [FFBPNeurode(LayerType.OUTPUT) for _ in range(outputs)]
        self.add_to_head(input_layer)
        self.add_after_cur(output_layer)
        self._link_with_next()

    def _link_with_next(self):
        for node in self._current.data:
            node.reset_neighbors(self._current.next.data, FFBPNeurode.Side.DOWNSTREAM)
        for node in self._current.next.data:
            node.reset_neighbors(self._current.data, FFBPNeurode.Side.UPSTREAM)

    def add_layer(self, num_nodes):
        if self._current == self._tail:
            raise IndexError
        hidden_layer = [FFBPNeurode(LayerType.HIDDEN) for _ in range(num_nodes)]
        self.add_after_cur(hidden_layer)
        self._link_with_next()
        self.move_forward()
        self._link_with_next()
        self.move_back()

    def remove_layer(self):
        if self._current == self._tail or self._current.next == self._tail:
            raise IndexError
        self.remove_after_cur()
        self._link_with_next()

    @property
    def input_nodes(self):
        return self._head.data

    @property
    def output_nodes(self):
        return self._tail.data


class FFBPNetwork:

    class EmptySetException(Exception):
        """
        Error if set is empty.
        """

    def __init__(self, num_inputs: int, num_outputs: int):
        self.network = LayerList(num_inputs, num_outputs)
        self.inputs = num_inputs
        self.outputs = num_outputs

    def add_hidden_layer(self, num_nodes: int, position=0):
        """
        Add hidden layer of Neurodes at specified position.
        :param num_nodes:
        :param position:
        :return:
        """

        self.network.reset_to_head()
        for _ in range(position):
            self.network.move_forward()
        self.network.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=3001, verbosity=2, order=NNData.Order.RANDOM):
        """
        Train the Neural Network on our training dataset.
        :param data_set:
        :param epochs:
        :param verbosity:
        :param order:
        :return:
        """

        # Check if training set is empty
        if data_set.number_of_samples(data_set.Set.TRAIN) == 0:
            raise self.EmptySetException
        else:
            for epoch in range(epochs):
                data_set.prime_data(data_set.Set.TRAIN, order=order)
                output_node_errors = []
                while not data_set.pool_is_empty(data_set):
                    feature_label_pair = data_set.get_one_item(data_set.Set.TRAIN)
                    features = feature_label_pair[0]
                    labels = feature_label_pair[1]

                    # Get list of Input/Output Neurodes
                    inputs = self.network.input_nodes
                    outputs = self.network.output_nodes
                    sample_output = []

                    # Set Inputs
                    for i, feature in enumerate(features):
                        inputs[i].set_input(features[i])

                    # Set Expected Values
                    for i, label in enumerate(labels):
                        outputs[i].set_expected(label)
                        error = outputs[i].value - label
                        output_node_errors.append(error)
                        sample_output.append(outputs[i].value)

                    if verbosity > 1:
                        if epoch % 1000 == 0:
                            print(f"Sample {features} Expected {labels} Produced {sample_output}")

                squared_errors = [output_error ** 2 for output_error in
                                  output_node_errors]
                sum_squared_errors = sum(squared_errors)
                mean_sum_squared_errors = sum_squared_errors/len(output_node_errors)
                epoch_rmse = np.sqrt(mean_sum_squared_errors)

                # Report epoch RMSE
                if verbosity > 0:
                    if epoch % 100 == 0:
                        print(f"Epoch {epoch}: RMSE = {epoch_rmse}")

                # Reset RMSE
                epoch_rmse = 0

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """
        After training our NN on the training dataset, train the
         NN on our unseen testing dataset.
        :param data_set:
        :param order:
        :return:
        """

        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        else:
            data_set.prime_data(order=order)
            sum_error = 0
            produced_outputs = []
            while not data_set.pool_is_empty(NNData.Set.TEST):
                x, y = data_set.get_one_item(NNData.Set.TEST)
                for j, node in enumerate(self.network.input_nodes):
                    node.set_input(x[j])
                produced = []
                for j, node in enumerate(self.network.output_nodes):
                    sum_error += (node.value - y[j]) ** 2 / self.outputs
                    produced.append(node.value)
                    produced_outputs.append(node.value)

                print(f"Sample {x} Expected {y}, Produced {produced}")

        print("Final Test RMSE = ", math.sqrt(
            sum_error / data_set.number_of_samples(NNData.Set.TEST)))
        return produced_outputs


def load_xor():
    """
    Defines an XOR feature and label set and
    creates a test object of our class with these sets.
    :return:
    """
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]

    #[1,0],[1]

    test_xor_object = NNData(features=features,
                             labels=labels,
                             train_factor=1)

    return test_xor_object


def run_iris():
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3], [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5], [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2], [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]

    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]

    data = NNData(Iris_X, Iris_Y, .7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2],
             [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33],
             [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46],
             [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59],
             [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72],
             [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85],
             [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98],
             [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24],
             [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37],
             [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5],
             [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331],
             [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446],
             [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175],
             [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599],
             [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501],
             [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135],
             [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114],
             [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868],
             [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962],
             [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957],
             [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068],
             [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737],
             [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883],
             [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035],
             [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392],
             [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968],
             [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145],
             [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041],
             [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963],
             [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505],
             [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995],
             [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998],
             [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015],
             [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017],
             [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487],
             [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826],
             [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136],
             [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068],
             [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516],
             [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193],
             [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319],
             [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236],
             [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178],
             [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764],
             [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476],
             [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 10001, order=NNData.Order.RANDOM,verbosity=1)
    network.test(data)


def run_XOR():
    data = load_xor()
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(4)
    network.train(data, epochs=100000, verbosity=1)


def plotting():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2],
             [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33],
             [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46],
             [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59],
             [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72],
             [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85],
             [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98],
             [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24],
             [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37],
             [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5],
             [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331],
             [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446],
             [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175],
             [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599],
             [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501],
             [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135],
             [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114],
             [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868],
             [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962],
             [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957],
             [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068],
             [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737],
             [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883],
             [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035],
             [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392],
             [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968],
             [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145],
             [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041],
             [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963],
             [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505],
             [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995],
             [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998],
             [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015],
             [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017],
             [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487],
             [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826],
             [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136],
             [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068],
             [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516],
             [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193],
             [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319],
             [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236],
             [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178],
             [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764],
             [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476],
             [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 5000, order=NNData.Order.RANDOM, verbosity=1)
    produced_outputs = network.test(data)
    produced_output_array = np.asarray(produced_outputs)

    # sin graph between 0 and pi/2
    x = np.arange(0, np.pi/2, 0.01)
    y = np.sin(x)
    plt.plot(x, y, "r-")

    # sin graph of test output
    x2 = np.arange(0, produced_output_array[-1], .01)
    y2 = np.sin(x2)
    plt.plot(x2, y2, "b-")
    plt.show()


def main():
    run_iris()
    run_sin()


if __name__ == '__main__':
    plotting()




