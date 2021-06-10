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

        # Default to position right after head node
        if position == 0:
            self.network.reset_to_head()
            self.network.add_layer(num_nodes=num_nodes)

        # Add node in after specified position
        if position > 0:
            # Reset to head node
            self.network.reset_to_head()

            # Move forward i positions
            for i in range(position-1):
                self.network.move_forward()

            # Add hidden layer
            self.network.add_layer(num_nodes=num_nodes)

            # Reset to head
            self.network.reset_to_head()

    def train(self, data_set: NNData, epochs=3001, verbosity=2, order=NNData.Order.RANDOM):

        # Prime the data
        data_set.prime_data(data_set.Set.TRAIN, order=order)

        # Check if training set is empty
        if data_set.pool_is_empty(data_set):
            raise self.EmptySetException
        else:
            final_rmse = 0
            for epoch in range(epochs):
                data_set.prime_data(data_set.Set.TRAIN, order=order)
                while not data_set.pool_is_empty(data_set):
                    feature_label_pair = data_set.get_one_item(data_set.Set.TRAIN)
                    features = feature_label_pair[0]
                    labels = feature_label_pair[1]

                    # Get list of Input/Output Neurodes
                    inputs = self.network.input_nodes
                    outputs = self.network.output_nodes
                    output_node_errors = []

                    for i, feature in enumerate(features):
                        inputs[i].set_input(features[i])
                    for i, label in enumerate(labels):
                        outputs[i].set_expected(labels[i])
                        error = outputs[i].delta
                        output_node_errors.append(error)

                    squared_errors = [output_error ** 2 for output_error in output_node_errors]
                    sum_squared_errors = sum(squared_errors)/len(outputs)
                    epoch_rsme = np.sqrt(sum_squared_errors)

                    if verbosity > 0:
                        if epoch % 100 == 0:
                            print(f"Epoch {epoch}: RMSE = {epoch_rsme}")
                    if verbosity > 1:
                        if epoch % 100 == 0:
                            print(f"Sample {features} expected {labels} produced {output_node_errors}")
                        if epoch % 1000 == 0:
                            print(f"Epoch {epoch} RMSE = {epoch_rsme}")

                    final_rmse = epoch_rsme
                    epoch_rsme = 0

            print(f"Final RMSE: {final_rmse} ")




        # Get one item
        # Feed it to the input layer
        # See what comes out in the output layer
        # update error
        # Feed the answer back into the network (ground truth - expected value) using set expected
        # repeat

        # mean squared error for items in epoch
        # add up each epochs value and take ro
        # Between epochs reset test data

    def test(self, data_set:NNData,order=NNData.Order.SEQUENTIAL):
        pass

def unit_test():
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.2]]

    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ]]
    network = FFBPNetwork(4,3)
    network.add_hidden_layer(3)
    data = NNData(Iris_X, Iris_Y, train_factor=.7)
    network.train(data, verbosity=2)


if __name__ == '__main__':
    unit_test()