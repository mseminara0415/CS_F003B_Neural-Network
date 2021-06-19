"""
Neural Network
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

        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException
        for epoch in range(0, epochs):
            data_set.prime_data(order=order)
            sum_error = 0
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                x, y = data_set.get_one_item(NNData.Set.TRAIN)
                for j, node in enumerate(self.network.input_nodes):
                    node.set_input(x[j])
                produced = []
                for j, node in enumerate(self.network.output_nodes):
                    node.set_expected(y[j])
                    sum_error += (node.value - y[
                        j]) ** 2 / self.outputs
                    produced.append(node.value)

                if epoch % 1000 == 0 and verbosity > 1:
                    print("Sample", x, "expected", y, "produced",
                          produced)
            if epoch % 100 == 0 and verbosity > 0:
                print("Epoch", epoch, "RMSE = ", math.sqrt(
                    sum_error / data_set.number_of_samples(
                        NNData.Set.TRAIN)))
        print("Final Epoch RMSE = ", math.sqrt(
            sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))

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
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_XOR():
    data = load_xor()
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(1)
    network.train(data, epochs=10001)


def main():
    run_iris()
    run_sin()
    run_XOR()


if __name__ == '__main__':
    main()

"""
========= Sample Run =========
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.9995390940466543, 0.9999677959677977, 0.9962744442902222]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.9999003956387701, 0.9999938730704687, 0.998888017358306]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.9987654945100697, 0.9999130396194397, 0.9929883486615305]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.9850651074612392, 0.9997408314445448, 0.981738933523911]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.9889861731492544, 0.9997332738400468, 0.9831372920913526]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.9976396914792538, 0.9998040159809063, 0.9905103398671452]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.9996461961730617, 0.9999789000428512, 0.9972618782821653]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.9874763647878316, 0.9997159346496747, 0.9800912352873205]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9939128426681149, 0.999953557288923, 0.9908814751365266]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.9926608045148945, 0.9997597211294214, 0.9834850031477649]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.9996693589705898, 0.9999666355265752, 0.9967830116274606]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.9998934492481154, 0.9999947083362364, 0.999051986234074]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.999587272812145, 0.9999821926565985, 0.9962923483320757]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.9996677666408339, 0.9999775319017511, 0.9976862578500215]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.9989144925529998, 0.9999417893704783, 0.9941033568639198]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.9999085385776186, 0.9999927708579612, 0.9982070089619137]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.9996777508566951, 0.9999740930445093, 0.9966946697331205]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.9904245226101493, 0.9998440596267183, 0.9832245671628904]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.999324788337588, 0.9999584222370039, 0.9950981822728745]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.9990121879740909, 0.9999304637804705, 0.9932222438125916]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.9985913511094683, 0.9998982548937833, 0.9923222934089252]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.9852644756734118, 0.9995463676516554, 0.976888071098325]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.9916705506956766, 0.9998737773079412, 0.9869031930566271]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9867052580107875, 0.999698202724567, 0.9791643240615351]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.9888616769194531, 0.9998275919016333, 0.9835531323456991]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.9993890865050279, 0.9999453915291063, 0.9953118057818374]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.9910814551678324, 0.9999157019599235, 0.9876570276789006]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.9994836682955657, 0.9999589672145016, 0.9951929693205935]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.9993326307261069, 0.9999677902393429, 0.9954609599807517]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.9999164330060998, 0.9999970104356715, 0.9988566108801645]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.9975699747806704, 0.9998099585417224, 0.9898503027093553]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.9996248477826105, 0.9999782091582851, 0.9972286837559076]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9912586999812837, 0.9998859024997782, 0.984518182903346]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.9996368679309049, 0.9999825994105911, 0.9973085358704935]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.9889565434114654, 0.999881958510688, 0.9824660940701524]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.9998307620756957, 0.9999928592926368, 0.9984107259365439]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.9998034876707474, 0.9999923301442448, 0.9985316166243141]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.9997734230381321, 0.9999893075206217, 0.9981916713454971]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.9881331291264797, 0.9997094292026318, 0.9771694169545174]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.9894652658173237, 0.9998284445304689, 0.9822570324266697]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.9998393986140074, 0.9999874616184069, 0.9983428054738216]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.9998872388128791, 0.9999937216701004, 0.9985075390486919]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.9997759856325088, 0.999987896165097, 0.9980522845849783]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.9996988536647624, 0.999969263845606, 0.9974104298261923]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.9991409483110519, 0.9999451894793867, 0.9948369319526434]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.9995499169350796, 0.9999500819185635, 0.9953710234557381]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.9902669414575883, 0.9999347389023904, 0.9857879196765627]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.9896877100753738, 0.9996995234941096, 0.9799386152360317]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.9862981306654497, 0.9995997537151431, 0.9781527527628877]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.9996625972617106, 0.9999681173583204, 0.9975872269835812]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.9999405120965891, 0.9999960879824354, 0.9987321585619966]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.9864632180670978, 0.999659556411424, 0.9751367854444034]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.9876575804833422, 0.9996688661936567, 0.9787575421833401]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.9999611467525136, 0.9999980240059677, 0.9992014635460782]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.9998331892192842, 0.9999862809895596, 0.9981623345882604]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.9987218170876101, 0.999943190291367, 0.9944193398041602]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.999540195542971, 0.9999815093919915, 0.9967425961466942]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.99214941068762, 0.9999238871946889, 0.9870532385508637]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.9911709534252641, 0.999874644290438, 0.9839575825865006]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.9999349301648831, 0.9999972573069125, 0.9992159548563715]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.9897434426474896, 0.9998229532831612, 0.9825087783942335]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.9991736254094189, 0.9999359833296287, 0.994552317211127]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.9817963280923745, 0.9994139655231197, 0.969945248815732]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.9998751648452441, 0.999989933087445, 0.9985104132515121]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.9995727006620158, 0.9999686848401865, 0.9958273865777606]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.9920089901243723, 0.9998826900671584, 0.9850659503749868]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.9996081532889864, 0.9999851541097311, 0.9967419463282713]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.9996986526058715, 0.9999692516193704, 0.99732732098775]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.988232626879084, 0.9996479120798956, 0.9778445658262298]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.9899525139942731, 0.999817516031633, 0.9811592643795986]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.9996182040339089, 0.9999884714431598, 0.996638743816408]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.9891644453428204, 0.9998336224417533, 0.9808386517012758]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.9989633474781087, 0.9999380458419764, 0.9935000679103203]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.9994582986248152, 0.9999466492520597, 0.9961126329074668]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.9990970015684025, 0.999962234055017, 0.9946425701524352]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.9893927776802564, 0.9997086044187218, 0.9783462095738659]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.9993433684733888, 0.999973263773769, 0.9952449473974333]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.9996621987217619, 0.9999781461443559, 0.9970689045954695]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.9997221372591311, 0.9999745900932826, 0.9974213879350644]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.9994500043035699, 0.9999809686749892, 0.9959637824614445]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.9988538933844523, 0.9999095935216864, 0.9930271828007994]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.9999468305294661, 0.9999980328960066, 0.998967412879361]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.9997075518976981, 0.9999868270147712, 0.9972636469280861]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.992026245350294, 0.9998581014329658, 0.9842947283627504]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.9995145243099419, 0.9999783968523556, 0.996079165518631]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.9994744149708981, 0.9999837146703865, 0.9961436629382473]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.9870727840068697, 0.9997166870504248, 0.9763782109520097]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.999656502402954, 0.9999649981561978, 0.9971878409908892]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.9992406081033146, 0.9999573495744565, 0.9938300594031986]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.9994998151044945, 0.9999778782086929, 0.9955320732578673]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.9998930158874543, 0.9999933943781206, 0.9981652842954528]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.9998698943320421, 0.9999934304700528, 0.9987356767961101]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.9998792329087097, 0.9999920825005527, 0.9988648116244855]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.9997587742348039, 0.9999824746822519, 0.997463412139615]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.9984027038120504, 0.9998884887781477, 0.9906959097012845]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.9982721831657558, 0.999918208886964, 0.9912309800035197]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.9996441786955194, 0.9999767950222587, 0.9972428872130525]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.9998368349806462, 0.9999916563917036, 0.9982503164047075]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.9816043952221012, 0.9994517031178884, 0.9712081698462935]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.9977034276884059, 0.999771064551122, 0.9880019693921909]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.999939314795599, 0.99999558567826, 0.9984403460952447]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.9992372798798804, 0.9999485141324747, 0.9948307167012498]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.9972099209150167, 0.9998598347038161, 0.9902816355724215]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.9885133046731954, 0.9997862512112079, 0.9791102516785855]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.9998177785488479, 0.9999814657730649, 0.9975648598409264]
Epoch 0 RMSE =  0.8130998777964139
Epoch 100 RMSE =  0.6908281110671718
Epoch 200 RMSE =  0.6900886550798317
Epoch 300 RMSE =  0.6899639268642532
Epoch 400 RMSE =  0.6899269180187187
Epoch 500 RMSE =  0.6899300711796451
Epoch 600 RMSE =  0.6898764303557645
Epoch 700 RMSE =  0.6897537957206183
Epoch 800 RMSE =  0.6880842894330976
Epoch 900 RMSE =  0.606357894889641
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.9945537891817644, 0.5465175132588963, 0.47301457092313975]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.9849405642208352, 0.26397796715780397, 0.2717272131341442]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.8798119393003325, 0.17012401405967728, 0.1408020930416937]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.9984582172246599, 0.8053960397296964, 0.6891164402249641]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.9947225414692972, 0.41586397817741116, 0.4952800735649326]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.9920626672504658, 0.37412185946355037, 0.35261911057133816]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.870785722993465, 0.13968359652883877, 0.13135754717873052]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.9990647476184744, 0.8500176188544469, 0.6424454545874211]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.9925561278640331, 0.5326399365051766, 0.41581823408936214]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.9883948736682271, 0.37014045128428524, 0.3087682326658794]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.9948645689926306, 0.53245818724986, 0.5198344574333494]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.9739389896760539, 0.24194385382387687, 0.23413240751606704]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.8682099503276741, 0.29099406870660194, 0.1504210825240414]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.9962704150556213, 0.6007503007219358, 0.5122731043757293]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.9919017087169922, 0.613124074488846, 0.4114076612235516]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.9939366861144425, 0.6379548053876668, 0.44407570218170345]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.8894598041732523, 0.1854551912866852, 0.14575157272912298]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.992896687027687, 0.5878843937510703, 0.4357333656894539]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.9804825443366334, 0.3205682294518016, 0.3068456983642696]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.8554955513475196, 0.14271780246891194, 0.11522766490792473]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.9922794153446196, 0.5525279887497073, 0.36289142309621414]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.9975003976962338, 0.6887703620836001, 0.5895802670934985]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.9840928725709115, 0.30495016132852526, 0.27839329066938523]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.8797758297790107, 0.19691750660677107, 0.13224095188927285]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.98986219367508, 0.5083469228474841, 0.3443384749726269]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.838992619842617, 0.07966128613927552, 0.1011190079873216]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.9947702696946268, 0.4686337487330912, 0.4965402975452686]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.8263468979544537, 0.0763632433983521, 0.08853754821519248]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.9971858433729006, 0.5984128333937855, 0.514134817913483]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.8628415618190969, 0.13377777286222478, 0.12462096924901968]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.8150179049789694, 0.05738725446385641, 0.09347414179185676]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.9896675728307404, 0.39318561058424095, 0.3250576834970009]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.9942296656821031, 0.5581529500412388, 0.4624430041113944]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.9969665864215257, 0.7785152915152461, 0.6250189994979879]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.998136680913475, 0.7695514823285191, 0.6994880745412992]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.9934275325796073, 0.45273029369279144, 0.382489862767008]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.9974203785303531, 0.658270678368983, 0.5784670688006784]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.9906058880566425, 0.3213003803404214, 0.34558214542496024]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.9965318693076387, 0.685819023578694, 0.5613370278315936]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.998064466478973, 0.7197141495023761, 0.6363861490796318]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.9943851248293472, 0.5923650027365435, 0.4856717694878668]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.9983411843524852, 0.7953639204118059, 0.5924900172373905]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.8491237287567849, 0.10672236718600567, 0.1153026800654511]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.8528104438019318, 0.12864187005978037, 0.1320580827619557]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.9983511071077292, 0.8256814054811464, 0.7221017376447268]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.9990801463813664, 0.8629452626253691, 0.6714906120968888]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.9948993261761805, 0.4209497742676636, 0.43780475023908794]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.9825824065458042, 0.2154402122015967, 0.2768008120968109]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.9991760062086473, 0.9278169099316097, 0.7204199151184717]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.8334862887665015, 0.08252446518797898, 0.10183220806364657]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.8595639416291748, 0.08018239690674274, 0.10903329612272739]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.891977161162287, 0.24853580129671718, 0.16623452453853882]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.9869320856880199, 0.3118766006549809, 0.32905718177736143]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.9965118871007419, 0.7035448386927158, 0.5799089014336352]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.9638276817502913, 0.11591292920257568, 0.19807388525106184]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.9950132817968376, 0.49268249158938954, 0.4260944423556031]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.8900805641935672, 0.14948324025118054, 0.14715696599034686]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.8123725933863402, 0.08843531329029675, 0.11942787829852539]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.898303766402695, 0.09372630698595531, 0.1304162947806979]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.847114028213305, 0.06638584959440576, 0.1068182282295873]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.8547534324570799, 0.17366340540511413, 0.12636948590241698]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.9761360528389779, 0.18033370047638325, 0.2249926674314425]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.9659750709815971, 0.09857408161111927, 0.1837432379704174]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.9812277462763559, 0.22497671847912226, 0.25579737012693454]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.9786054387082318, 0.20211418444629592, 0.2412417491993602]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.8446076800740162, 0.08400913686798007, 0.09786476415429748]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.8654886659652116, 0.1269229811367498, 0.12029135963726237]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.7815896720812546, 0.044070796660640704, 0.07845549860833313]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.9953870528659752, 0.45974627167319476, 0.49043679799932843]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.8268276002137505, 0.06165614987196967, 0.1049445579863181]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.9957600938597339, 0.5029411067625653, 0.5076185890108126]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.9953980545019004, 0.4506893572753395, 0.49945650638763545]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.9955335592191275, 0.6531824037032852, 0.5024254746611817]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.9948394875308986, 0.5346306213677167, 0.48011171117270496]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.9863106370203519, 0.3946617944455287, 0.33857478260726637]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.9945582604198655, 0.5201405948212718, 0.5007425150805044]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.9898015926701014, 0.43357734646198187, 0.36553178956350824]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.9915830377996965, 0.5696086494133017, 0.4045800185946014]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.8859334208938779, 0.16969536437032415, 0.16137236543784228]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.9994003692420812, 0.9280270415687557, 0.7662309640000272]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.9930979108742662, 0.33592110226743066, 0.3585448813453218]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.9647707120703737, 0.11330873279987284, 0.21050971079911274]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.9883236448104161, 0.3297579043643016, 0.35197743932215425]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.9973723184758788, 0.7836171172021204, 0.6196819275765736]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.859327318673621, 0.1300858971330023, 0.12882564789531567]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.9985796605644168, 0.777667370613883, 0.587157258918703]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.9982512065387062, 0.7979130902523078, 0.6417994544692092]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.9979862557511635, 0.787399537233005, 0.6940248356685591]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.9928720402705487, 0.42535136603996426, 0.4108911140054496]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.8540568937883662, 0.08486345271989962, 0.13263118259616755]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.7801054444802774, 0.043858949826554354, 0.09038407555385525]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.9835070845237622, 0.2948158956311155, 0.3050977133510564]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.9917006473550233, 0.318084488670405, 0.41582844987001166]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.9974859732579561, 0.7529685642762413, 0.6136973935572003]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.9936367635588224, 0.5828837054123361, 0.40854462389111545]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.9987050384397483, 0.8941871853193565, 0.6932395077754564]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.8296195037069366, 0.07768606696701275, 0.11031828614299494]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.9989958840963888, 0.9009014091155253, 0.774596944325578]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.9941371745099865, 0.6813018759317735, 0.44994409502016136]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.8787250197218344, 0.22935436778710033, 0.17198551208003626]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.9588640513876469, 0.150605722628767, 0.22157616623840087]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.9873408490446661, 0.2822436124528727, 0.326242114540265]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.8372540759112576, 0.08424701212928086, 0.10963622426001661]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.9136313136050157, 0.35700007441117115, 0.21194067388709048]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.8628353133381038, 0.07891162818801549, 0.11229434865062132]
Epoch 1000 RMSE =  0.6063945174750922
Epoch 1100 RMSE =  0.6063567322461246
Epoch 1200 RMSE =  0.6063481620840177
Epoch 1300 RMSE =  0.6063949549144946
Epoch 1400 RMSE =  0.6063896500672464
Epoch 1500 RMSE =  0.606397316316862
Epoch 1600 RMSE =  0.6063817303136548
Epoch 1700 RMSE =  0.6063891621651954
Epoch 1800 RMSE =  0.6061550130148746
Epoch 1900 RMSE =  0.5868396400602495
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.9779911346729959, 0.3648891875533463, 0.29043208657857794]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.977915034448971, 0.05445792771516363, 0.7934461296352839]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.9556923507009514, 0.35722159626898303, 0.15507335379581239]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.9346631984171689, 0.241123361511223, 0.1051964679726895]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.9837573272090226, 0.7141104777759208, 0.28557029968690795]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.983292604498905, 0.09034548776991153, 0.8925738818696152]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.934101109540651, 0.017780503159572508, 0.6359101172167192]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.9789109975521505, 0.06764000699496456, 0.7977514970154511]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.986138508387541, 0.6654924154167436, 0.2554830599135682]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.9868400882015623, 0.7931195025637745, 0.3126838197287341]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.9628641587873353, 0.39553404742827264, 0.17449722350350177]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.7428871014378078, 0.2195345574076692, 0.06871950989679847]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.7730353608215066, 0.38094975852801677, 0.06023548367365731]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.9794803994498917, 0.7126800356229701, 0.1625636508507689]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.9673528087941806, 0.5643189810102955, 0.1739130576827594]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.9749368112649347, 0.36173608186352835, 0.2721637828006986]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.9819470795419106, 0.29186508655518434, 0.6672976819535171]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.9884957505609511, 0.8554134660962077, 0.22971384979103995]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.7503320737077507, 0.2282057219708743, 0.05073834078836063]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.9844615635628158, 0.10228813313256654, 0.8051895289192487]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.773917991541436, 0.2538636909818409, 0.054584888073179946]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.7389361657016426, 0.28338567804271975, 0.047290158351177704]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.773950424158592, 0.2688298165639589, 0.05159132682890143]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.7141709087869218, 0.18034313132388197, 0.05316103373076865]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.9260861540981069, 0.2227012465490856, 0.1257398327416254]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.9684817886314492, 0.11122130644354387, 0.6207686474426327]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.9959275994476167, 0.8586890487391766, 0.4858200836450726]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.7302675322459572, 0.24389719722860798, 0.044184556427515995]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.9972398025736067, 0.8472965751446456, 0.6219917183482782]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.9503987467300622, 0.11826054846851586, 0.35390098070280307]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.9896719279494759, 0.1644949297221122, 0.9128602711960875]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.9704722212120177, 0.5223184709205754, 0.15692402786128468]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.8038000388071193, 0.47749196856464143, 0.0690063677867095]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.7256975152687257, 0.24229375575968215, 0.05311981204275999]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.8061866961153342, 0.4042105187947435, 0.08071845612543306]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.8068673446616131, 0.3558093473718644, 0.07856757387495979]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.9802015682251466, 0.6534091432272048, 0.20099897522295432]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.9579332261362555, 0.17665100054621535, 0.35035625867158626]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.9888358601875095, 0.7407717874673596, 0.38903619375056]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.9982155259123969, 0.9505891061816109, 0.4557652712092315]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.9966777253309012, 0.9202113267622344, 0.5981926834270976]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.7692655215278502, 0.48006273227218754, 0.06024783286203506]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.9646552801155981, 0.035289595933503524, 0.7258399352602287]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.9783271607490794, 0.6452879573411016, 0.23673867033500526]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.7908346681347662, 0.39884196666324606, 0.06254846115887165]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.9890442660252433, 0.8828299790785381, 0.27514767854432104]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.9842948777638609, 0.2967233741124356, 0.558393219518505]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.9946701627143932, 0.2943145418088067, 0.8972865450464638]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.8980332130223779, 0.1824610385245139, 0.16449379042401993]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.7696624337856092, 0.333569765344594, 0.06679349394200619]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.9339435249051697, 0.1840686609740197, 0.31159136131913306]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.7030239781474973, 0.1718504085844309, 0.04815919392000867]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.953115954418634, 0.5340364698676675, 0.12568145555619153]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.7821958943368811, 0.3647796743717055, 0.05925676816741358]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.9561957224453945, 0.03668484995159457, 0.7374942714217929]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.9972441224176969, 0.8900111973602066, 0.6967150156209364]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.9588071930598973, 0.2448632694702005, 0.29599264562714994]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.9857869639228783, 0.8093494828610311, 0.21802727780960984]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.6847303218858366, 0.2317334797664633, 0.0621445735393985]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.7892392485546194, 0.6240155543159629, 0.07258459800790663]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.9729917349065924, 0.6636127546095943, 0.19799864719160024]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.977920640851985, 0.16256690017184836, 0.7834675233079684]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.9857755651035854, 0.5734377618454405, 0.2221454734910321]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.9843004817451004, 0.2724510340211961, 0.7318365331783843]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.9846611115769176, 0.08831941194551601, 0.8671487710642667]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.9765075958589264, 0.22004290773716442, 0.5355191196909075]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.9498989171179735, 0.017574325957085687, 0.7400893388034188]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.7528161753774057, 0.3072168713883774, 0.05776785465980285]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.8090232997095099, 0.21541243055477904, 0.0737561094296248]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.776171055997067, 0.4334029663386521, 0.09892302109193261]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.9793465511614646, 0.663758132977367, 0.3023090796799945]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.9816418241739336, 0.7447762677501646, 0.2940117183233174]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.9630913267073792, 0.0983482146574939, 0.6470272924620037]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.9911338794208926, 0.4194020777267482, 0.7047423553075114]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.9702107999740499, 0.06809722220891723, 0.7698305239991118]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.968299861470635, 0.06605411322712493, 0.5779267438769067]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.6519198090461291, 0.14136109457821866, 0.042002888347185456]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.953407282060649, 0.40461731309152327, 0.12458790625195852]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.9480771393828697, 0.017033168103150698, 0.7627368205632108]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.9764041517190426, 0.33436923651069433, 0.3632427150061105]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.9636177829218388, 0.038871700174322385, 0.6884027899692661]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.9969046903705201, 0.8931909354331263, 0.4273979682496961]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.7329777560993125, 0.21419208915743654, 0.05444531474558023]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.977429670199259, 0.4985535691412359, 0.40267014388326605]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.9863935533388047, 0.22951873477972085, 0.5961423154087984]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.9854039568070271, 0.2792528675243264, 0.7142841075697396]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.9796270083844834, 0.057843231370749634, 0.8859404845840122]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.7499457092413563, 0.2658262971463313, 0.0474502500242555]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.9795613060594774, 0.08286392933910468, 0.8662035835932465]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.9683350203405332, 0.1433933427265352, 0.5855060022330038]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.9645389939334106, 0.45801218854031367, 0.19298972707347445]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.6625557796067414, 0.15580711499546426, 0.03779144577612749]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.9849597096271998, 0.46290037082610797, 0.3752850450895763]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.9111512084766412, 0.15971688111476695, 0.1441095496809611]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.8180319973593648, 0.5166514425555292, 0.14041374481490632]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.9561514616212193, 0.34681556698217264, 0.17622751688486663]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.9555146203965552, 0.02225671470075699, 0.7422156258663033]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.7737090459064487, 0.3098218276856429, 0.09924079921628216]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.7313526336127516, 0.2501568443042925, 0.055501552978954295]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.9775227947057515, 0.6711934548320463, 0.24662866690601964]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.7636864971565119, 0.37892729070788816, 0.06003926558892563]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.804864688863134, 0.5089128252467845, 0.09201898386357978]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.7457378363832129, 0.3153835796598863, 0.0711571487301424]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.9703798531087185, 0.14747634091308237, 0.5832805355165495]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.7928543701784878, 0.40750715519735503, 0.07346837713782994]
Epoch 2000 RMSE =  0.5699152466719019
Epoch 2100 RMSE =  0.40787997635241924
Epoch 2200 RMSE =  0.4000402884935634
Epoch 2300 RMSE =  0.40044426236246
Epoch 2400 RMSE =  0.40161839933844756
Epoch 2500 RMSE =  0.4025980411543623
Epoch 2600 RMSE =  0.39969838125531565
Epoch 2700 RMSE =  0.39963845073603516
Epoch 2800 RMSE =  0.3983383193442008
Epoch 2900 RMSE =  0.3999678392272887
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.04338672004755666, 0.467955214321503, 0.07349868037897993]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.05525295458191077, 0.5806994876766083, 0.04141804230542899]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.10771122968689792, 0.8490661715689019, 0.059664391970455595]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.03998844848414813, 0.4950692022453852, 0.033718553773685635]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.0025057556466126264, 0.07165368699964592, 0.9081857161391642]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.05335649873955723, 0.604136962432938, 0.028116507911597633]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.005908019427764102, 0.1271487208079148, 0.972033063782449]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.004693699220582561, 0.2711566464029369, 0.008674467424641447]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.003797696291313049, 0.19232596201662594, 0.00844391112480195]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.1276001106840355, 0.9020930221518341, 0.05878491558025632]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.0044396299890952905, 0.27421996384139036, 0.009629350893465815]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.002117625898701625, 0.04984817448198655, 0.9659160802532065]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.124171183620569, 0.8763624219583294, 0.06038071746352761]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.11970365466349918, 0.8423181737677549, 0.07656588679652676]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.022935299136434592, 0.35070428035317464, 0.019987246084200255]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.0018694815026191816, 0.034480792449790514, 0.9628470458380484]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0015343886127580537, 0.021185378531173898, 0.698682763068409]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.005421225501398436, 0.11977802751947486, 0.583704757282524]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.0007255295065111322, 0.008374493425321091, 0.810393072754329]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.004510447882955819, 0.2475157969709762, 0.009493557053448821]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.0018072749331821752, 0.02722289648937202, 0.9503121273129295]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.0012576643160500357, 0.015172714249112387, 0.9234819721063413]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.004074285041447814, 0.21348709521252862, 0.009200745881786878]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.005780960928175483, 0.4080431996724984, 0.011470337084779372]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.0014061639173227742, 0.021681923229307166, 0.9435785546076243]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.022375487209831253, 0.3574679963788231, 0.019225056856151414]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.0008503580564161121, 0.011503258614843031, 0.915485908085305]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.0010107317233261782, 0.0171446583141376, 0.9155000835093176]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.05675612062917055, 0.6477511545616029, 0.05027334005345926]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.06586805321542177, 0.6688402293905941, 0.04133830658047814]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.005523174419053304, 0.3757477696882098, 0.010923288166208304]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0012878770939816612, 0.04017999469883323, 0.9443882981484797]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07682051836432183, 0.7699182315286167, 0.042128303802314866]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.003470376353697893, 0.09435538254502213, 0.9730660339360303]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.0063296236881415415, 0.4884965371688134, 0.01269284807391891]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.017049847107148665, 0.25714078982022887, 0.315809695801644]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.004737985036475503, 0.23426723848525993, 0.009424173474667152]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.006939842423912183, 0.4764049537859246, 0.013890360412498224]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.005253706446658051, 0.2664050426275234, 0.009731950529380763]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.004193746320868505, 0.2589297223335855, 0.009521805134831826]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.022973859667345817, 0.3040513921167534, 0.018373710980186836]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.00906773403431094, 0.694592836767883, 0.021170560412342877]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.0007449115583336624, 0.013750486317073068, 0.9020191601923846]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.005700546885801462, 0.615114936947632, 0.014194544987993569]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.0006726279219616701, 0.008353790188838684, 0.9220391062714759]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.0013587367460147247, 0.019189796814000293, 0.9399593100701423]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.005413890854419972, 0.2556353258840945, 0.010122375757030533]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.019716254702572076, 0.4212511844660249, 0.022024191166132908]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.00154777792361857, 0.038986274014877025, 0.9133945677065013]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.007077554267702135, 0.5767623015835311, 0.015842644669208338]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.04189431796704344, 0.6415575518689852, 0.036056549681916836]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.002270156464814397, 0.042060435185222164, 0.9605091569153661]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.0062297967963247675, 0.5502309282698367, 0.016109971004359746]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.005150090828046441, 0.3800631195572863, 0.010959620171819139]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.005044053504256096, 0.46096045910813843, 0.011435784218324222]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.1006397425178646, 0.8172183766393049, 0.04648726844784404]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.004878318104363257, 0.3194942791965505, 0.010358290613043934]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.005446600159987661, 0.35985513084299176, 0.011813606365199469]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.0006630805637498968, 0.007290612226474677, 0.914229342749435]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.005300649761081362, 0.3651124713116731, 0.011384745877173023]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.08615512970491415, 0.7738715126770666, 0.12976752357519025]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.0584160665097641, 0.7206809568728119, 0.038422381176579964]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.004335124242120244, 0.25910659744274483, 0.009016156125477397]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.0007739534725517134, 0.008654093124705952, 0.8842437726023722]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.005075263501115498, 0.26976932822366945, 0.011407152390611228]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.0009970716415783102, 0.016764730970809535, 0.8929267019464806]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.10075982689124006, 0.8132095039226427, 0.05273493264785417]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.00208622219719248, 0.042818749860175025, 0.76519035052623]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.03147341022028775, 0.5442340516410927, 0.023812164858876987]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.050985524011317035, 0.6127706985345195, 0.03153272237269031]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.048885075879307634, 0.630460462349875, 0.03216245365338293]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0007554193895367486, 0.008523846383200445, 0.9145438084478227]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.0014687461155936628, 0.03237036981319311, 0.9415355461074636]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.0038156037210365624, 0.06330370853432524, 0.9566955064004433]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0007554158820937664, 0.008517700398109207, 0.9146807548399675]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.001172985160423411, 0.024865264662473883, 0.9261616909407954]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.08799558738650855, 0.6833377222023849, 0.04408469478886657]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.00413384710148228, 0.2308648946098854, 0.008115830942474717]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.11837895941372834, 0.8484632263007145, 0.05056671152352203]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.006985342254658451, 0.4169494835932598, 0.013327053540016965]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.042992309945594026, 0.5326255379505026, 0.026910079960998454]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.07095342732032964, 0.6559052137347323, 0.07254880682207672]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.0037519116956074424, 0.057385243899801035, 0.9499804820588142]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.0017457804197740256, 0.03913798605030751, 0.9588837884036294]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.003084610360509949, 0.15006086190247864, 0.006838479419859554]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.033855379307770515, 0.47656364517050015, 0.022491553020135894]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.0029045349357347226, 0.08884078254706172, 0.9576255045103433]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.0010963326606355565, 0.020420108818600468, 0.8687080717203347]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07847194882975603, 0.7935083612995909, 0.04280597146117501]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.006668138595825723, 0.45104256085627237, 0.01490065141717398]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.003068338980644743, 0.15937820601049027, 0.007424202815721947]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.09628320078833072, 0.8620665451787549, 0.05254000359944016]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.006299178119093971, 0.4499778858673667, 0.012674039124454644]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.007573710579658844, 0.29677058540770374, 0.011664883177018569]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.0032341051430792487, 0.05910887337137519, 0.9169786722960721]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.016662597207824387, 0.2602830641701505, 0.2375278495184311]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.005005122009217694, 0.37261402187405607, 0.011908532544936712]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.0682968614919691, 0.7009610603431502, 0.03363749888779001]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.003745602339977262, 0.28359833037719806, 0.010482588129112626]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.038071214766243644, 0.49671935849656357, 0.0249804068542891]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.0027835640226810186, 0.04045615315569597, 0.931065464187185]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.002139554653704831, 0.04372441366105778, 0.9463175338286768]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.06017408636841475, 0.6461649302468916, 0.03825604469322662]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.09907996802826753, 0.8203705556683495, 0.6084307184108855]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.09225108554680034, 0.8420915424055199, 0.05039507738571089]
Epoch 3000 RMSE =  0.3984938605099823
Epoch 3100 RMSE =  0.3982422713467396
Epoch 3200 RMSE =  0.3959546419311626
Epoch 3300 RMSE =  0.39595175701529994
Epoch 3400 RMSE =  0.3961232080292785
Epoch 3500 RMSE =  0.3966697435911097
Epoch 3600 RMSE =  0.3965049444971755
Epoch 3700 RMSE =  0.39570077290558425
Epoch 3800 RMSE =  0.3947655015742197
Epoch 3900 RMSE =  0.39564829270755164
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.0005417252056707775, 0.006009467743424175, 0.9125871675679809]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.0005968414175092544, 0.010409085857042338, 0.9050100924680436]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07640347250777316, 0.790471165824305, 0.03833016564645378]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.09368044166611778, 0.861415122650586, 0.04711504530091946]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.04746982512627039, 0.6350288757190187, 0.02882290904457317]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.06643254200691734, 0.7047027626897259, 0.030140644420801745]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.001536470002272355, 0.026529924283170334, 0.9629422256095251]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.00880485274764439, 0.689504786225416, 0.018740093140049822]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.019132975617924756, 0.42022369429338313, 0.01946142845422343]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.012511953296669449, 0.21168457509926214, 0.25611131067458864]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.002391221163154183, 0.04489254054322213, 0.9231639121196404]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.0010228946624585757, 0.01170111100648073, 0.9241586930831645]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.006111782570692685, 0.4551920313663882, 0.011274482434445428]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.0017834913961451864, 0.0384823434280777, 0.7600738932003624]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.11588663427129907, 0.8522980765437196, 0.04512680303762988]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.0006458958369051034, 0.011195986934085, 0.9108954219165086]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.0028446718933523235, 0.07337923162134143, 0.9729331138077355]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.0006717971598791513, 0.008522086075258239, 0.9176878453917033]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.0060348578585187615, 0.5516341879490162, 0.014129248207578864]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.006470644567131349, 0.44776345110245885, 0.013221294685615748]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0006167511316214539, 0.006843195259007554, 0.914168622478847]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.030514992699560265, 0.5520703403639275, 0.021225068882151237]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.005633642530131216, 0.39639689168949044, 0.010244176823106008]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.013311045695860014, 0.3181374732675147, 0.894748973109431]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.0011627596499405631, 0.02507207117035961, 0.9428824543325596]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.04005319651980598, 0.4909623481264116, 0.02940414465369012]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.0040093514270571695, 0.23144085139511994, 0.007214556448989342]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.10514350105727488, 0.8445149629957933, 0.05348125427841807]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.0407075899375457, 0.640552265494009, 0.03171978072552053]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.005129367953229303, 0.37466009843085485, 0.009988694053591145]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.005384016546312264, 0.35833749060123093, 0.009751608470608427]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.004322623554908961, 0.2627807783343639, 0.008590373763090575]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.004732124530711777, 0.3202726626092561, 0.009092229056488816]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.0323038502597054, 0.38516855159145014, 0.08255610182045059]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.0005208950022839014, 0.0054129353530314, 0.8292391943714464]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0011703257417993026, 0.028246475971256402, 0.9175525472975251]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.003071996618627963, 0.04579691395434284, 0.9499501545715071]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.06463127205951352, 0.6533627108370351, 0.03697333354387853]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.0015306156261952267, 0.031290728567003476, 0.7899419993638981]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.002072406510503815, 0.02926575321278582, 0.9366188087492496]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.09806375149814778, 0.8208043019099367, 0.04114923950406018]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.006869268066848701, 0.5728588112816585, 0.013999228983400175]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.0029775501171506857, 0.15688672033730658, 0.006628527556080794]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0004826798131975357, 0.005664662230390901, 0.8468911879713071]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.004891760815655748, 0.46022158069271285, 0.00999042361943948]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.0036964934276921277, 0.18093990941613025, 0.007492127710123925]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0009777010454736868, 0.02722138132088569, 0.9475833304562422]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.12170479600486656, 0.8689707632488888, 0.053814367697097305]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.002373586761576959, 0.06948502073963372, 0.9571985538615952]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.02185057998278479, 0.3420943370294862, 0.017094438238613573]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.0017385918191916046, 0.03821669712793608, 0.9655797337640954]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.005541024243758003, 0.6066802860369056, 0.012426235334355372]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.004203096919713183, 0.2604067821246268, 0.00788414195642475]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.0005513503906698797, 0.006430146120708361, 0.9199414479298587]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.004920731010310056, 0.27092603041725144, 0.010073632966135946]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.0014316230429190242, 0.03030396442341089, 0.9584317535795597]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.0007904886032296099, 0.011864914973856139, 0.9183852183572107]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.004624155857624622, 0.2190282954647636, 0.008260425633446141]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.007366850647706422, 0.2894758360688774, 0.010330646742520082]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.09013394388870279, 0.8356835095543246, 0.04453125348057216]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.006797438893336635, 0.4074783099279996, 0.01181419663252809]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.005271846591061813, 0.24377978042980894, 0.008857540253675031]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003985515729359928, 0.19480117000037242, 0.008191380075469153]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.042140434667595164, 0.5225908077312446, 0.023778659891836973]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.005131622529459843, 0.25036616341926077, 0.00852890848757735]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.09177682073894572, 0.6906927390510081, 0.03680116438086925]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.05691339695460143, 0.719103589888612, 0.033729267881685905]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.007170369781475527, 0.10901905473506279, 0.4619705389308966]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.0011162938548180238, 0.014637682776334492, 0.9393113402548529]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.10094307984890756, 0.8029809972659768, 0.07973753187940397]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.022996945715606103, 0.29673877029345225, 0.0159455001810079]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0006180688103970664, 0.0067052974746379225, 0.9153071547128662]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.09992104570912926, 0.8105517888648298, 0.09951728024181597]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.006768074040452656, 0.4594103204842443, 0.012333641212140875]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.005287067829991354, 0.3580343441922146, 0.010465729703928188]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.037082460384546136, 0.48804637099551307, 0.02248399409563735]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.004576454249576138, 0.25474695030701594, 0.007804963178166717]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.001158536260879598, 0.016134765017982538, 0.9441431058926789]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.00500636740736718, 0.37236100494177243, 0.009708210991651037]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.00441384278628647, 0.22865361447060067, 0.008567378891146601]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.04987554214678429, 0.6092889893503983, 0.02829306622422041]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.0005918005284369986, 0.01001446713821884, 0.9039541064461968]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.003643507933424393, 0.2759759224930041, 0.009439103521036954]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.06985068346584512, 0.6485541613033342, 0.06489312202069139]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.0005894679671375486, 0.00643642173142268, 0.8897471091688485]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.05216949554282757, 0.5846345760571195, 0.025139738263685842]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07528899764684141, 0.7541227596113929, 0.03773101763182923]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.0999535688496405, 0.8197993640039871, 0.046557437154009604]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.05834159492714675, 0.6398613244545001, 0.034315600891052706]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.001860282467721008, 0.033278670049154004, 0.9595357599667426]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.0008833164479951628, 0.01831763314864653, 0.9307324479588732]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.004084238934725931, 0.2517132284176413, 0.008381555545374475]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.0014893437251196175, 0.020438701955909275, 0.9503178076030269]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.12468735867806546, 0.8966203149208449, 0.052777138322871335]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.00617424933230261, 0.46854584495440454, 0.01134292250907608]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.032956564086498094, 0.4703296769495112, 0.02003102802868567]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.004768535436253186, 0.09637411856079826, 0.9723826156273235]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.06249927896881332, 0.6628568011276194, 0.04038477497154278]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.003117930254866215, 0.05141330029544213, 0.9565675957193279]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.0016574038623893067, 0.03287390048412944, 0.9485658224169018]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.022435326645794163, 0.3397119104599333, 0.017864113246177395]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.0029941352948600155, 0.1507728419398064, 0.006084129020601035]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.0013136622625737383, 0.035240472198451236, 0.9366027948143634]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.004857215494726138, 0.37385743284219514, 0.01063583290824047]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.0526725549359972, 0.5686352627233282, 0.03782183516071671]
Epoch 4000 RMSE =  0.39563691049462324
Epoch 4100 RMSE =  0.39440934284418666
Epoch 4200 RMSE =  0.39431812866557187
Epoch 4300 RMSE =  0.3944068821997965
Epoch 4400 RMSE =  0.3945725334819419
Epoch 4500 RMSE =  0.3937653121747796
Epoch 4600 RMSE =  0.39428543160860413
Epoch 4700 RMSE =  0.39402393503694516
Epoch 4800 RMSE =  0.39406574224336144
Epoch 4900 RMSE =  0.39367510069163586
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.00466538601046968, 0.37821296535589943, 0.010674813362439757]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.0016069922192425037, 0.0311267657388617, 0.9603456782997086]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.0005801203099740508, 0.007716313035079024, 0.9196895158243193]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.0005343026338501009, 0.009687353996309316, 0.9153609173466692]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.05662741863150004, 0.6043962560989166, 0.0761578279109925]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.09702463880434867, 0.8281403161945308, 0.04610254022805407]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.0024613218243874145, 0.06787559451536021, 0.9732964114365756]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.035648518230885454, 0.5065294963310215, 0.022213792248841218]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.10175090747405283, 0.8510306135632785, 0.05318188462328334]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.0065088406904639535, 0.4289400223570156, 0.011830925356225533]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.0005194570245901898, 0.006435227137721755, 0.8874751068783878]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.05459418791336823, 0.7353771190591153, 0.03349316909251479]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0004038419637994673, 0.005258676686239928, 0.8493827496122903]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.05544008698236475, 0.5633986611747437, 0.04863419723716623]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.019323522978041848, 0.296984472255426, 0.24369674195967891]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0005321379877514892, 0.006544166574489266, 0.9141497293280643]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.12584321711675917, 0.864556294168022, 0.07638365783896342]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.06260202763883563, 0.6678935472882096, 0.036279246841979085]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.003477849010340877, 0.29448796867418076, 0.009269631057713994]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.04564620217462778, 0.6460516118171947, 0.0284426598549768]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.004139729657660695, 0.2756718180691825, 0.008520081833534305]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.0821436041289394, 0.7205817253019282, 0.052821911749912104]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.004022182554004936, 0.27643057697958273, 0.007853134645866362]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.0044228757148088264, 0.23419068856853706, 0.00821673015927697]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.005310618751919748, 0.6203139392999916, 0.01237817826040604]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.018379533971909646, 0.42788183127906637, 0.01920560130695514]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005867829094652355, 0.4597170868935207, 0.011167058236344354]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.09443158285105878, 0.8258062837805581, 0.04055706880563238]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.002868941691812888, 0.15414378460694167, 0.006022851567124329]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.022189901321920807, 0.3141967466044306, 0.015562584306472355]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07350087070766984, 0.799327048210134, 0.0378066759120648]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.05666611879506567, 0.6588936190140446, 0.03362342098875335]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.006479290657632187, 0.479905481270837, 0.01211508342111602]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.003539538601106584, 0.19192188715708508, 0.007459773761280227]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.11113858193074808, 0.8401175051759993, 0.06899941839313442]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07221427178987733, 0.7672878238440696, 0.03733312603651181]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.001831354825471413, 0.037020101514630366, 0.9302341699243807]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.0013256770843279562, 0.025108771492422653, 0.9626095923311078]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.004213726056290233, 0.24560884370627895, 0.008412474831645447]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.008453251266813736, 0.6962959085256245, 0.018484456285931598]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.00046699719158310965, 0.005682940371977293, 0.9119981334566355]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.0026436693710102264, 0.0434028772161265, 0.9497226938748308]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.08934218399777633, 0.7124660581427722, 0.03611161052133188]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.005037425694072038, 0.26101695394601554, 0.008815225173942292]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.06418286698577577, 0.7105678510110744, 0.029654636354606554]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.0014252602948028575, 0.031224773655337698, 0.9485142914548909]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.0026859921565598, 0.04893769195789734, 0.9564893563368526]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.001497951940945651, 0.036599919995900274, 0.9657533517781668]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.005057848490097861, 0.37468318443100973, 0.010292262336680581]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.004787652009046757, 0.3872595638488242, 0.009544969645197883]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003912640117255542, 0.25718382456514666, 0.008307989143983237]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.003847340980860917, 0.23401311987544474, 0.007155515461299129]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.0046927420151047375, 0.46621904133557224, 0.00996066541784757]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.09033687683471155, 0.861529656251394, 0.0465308229435261]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.0009604784071421085, 0.013930312897484405, 0.9385329017330224]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.005168911842352669, 0.3590018702202335, 0.009677553419379795]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.1173562876726833, 0.8713646728117878, 0.053612448821585404]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.041615218220851825, 0.5148153754276728, 0.027159460042613555]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.00043478341347745773, 0.004824731268568601, 0.8334962050204415]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0008749598849966766, 0.0224517866257559, 0.9267530649748705]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.008870113387914632, 0.24322554320827156, 0.9162743801143689]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.04035405332272131, 0.538513171600255, 0.02374004657827208]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.004921798914483897, 0.37614874549266114, 0.009942646467703296]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.00621339675705297, 0.4477677137303839, 0.013179387153446723]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.006595453261620601, 0.5706501724802202, 0.013850281357155783]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.0028574924726326198, 0.15566527704784733, 0.006555850507464431]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.004543036815262114, 0.31864494950130434, 0.009049924353288968]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.004914587238333384, 0.25410842349019463, 0.008518646119628263]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.11203648784208863, 0.8460032914020507, 0.0446626949338312]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.0010580634577380674, 0.02240533635905658, 0.8179182448130067]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.05332012271752871, 0.5754633008301325, 0.035915908787856096]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.005804691024540861, 0.541692108213319, 0.014116912510242299]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.03920297198112732, 0.6327210194143056, 0.0316400374278778]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005930267093669918, 0.4649814175409739, 0.01130503624235941]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.0007467874870680266, 0.015815944069664214, 0.9324169693227671]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.000672766312731934, 0.010608268465122857, 0.9202657791759055]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003817696124384392, 0.19789420120502277, 0.008206940701552092]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.0017699580385788925, 0.025270396635124336, 0.9372360365180269]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.0009973548006406696, 0.02022476027846487, 0.8309152944112885]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0005332503663634148, 0.005993730560415283, 0.9153569101863324]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.0008841469254261421, 0.009909712100671845, 0.9251189691550102]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.08674023954635739, 0.8363033769111985, 0.04477558839951836]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.004116709077044669, 0.08594405618242064, 0.9727615830962167]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.0070646916420192725, 0.28947800174939714, 0.010388051623201595]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0008269064392491013, 0.023788570332791487, 0.949511804707211]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.05012213333110906, 0.581815423321168, 0.025174757175910004]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.0004759380218485159, 0.005810835784028443, 0.9214820763691495]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.004389930319926953, 0.253411350860742, 0.007758897896921271]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.0005093420837483064, 0.009052628544863996, 0.9044178603586772]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.12031422117705372, 0.8940692302391842, 0.052912411250926765]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.029353588770225914, 0.5438433200028703, 0.021318084594778193]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.03167018281980826, 0.46932424589428084, 0.02008068428017856]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.0009979496969555676, 0.014941829093159198, 0.9443663515243856]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.005413182635076802, 0.39166295518871275, 0.010291756051125821]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.06690756878543537, 0.6871853314014096, 0.03672270910886912]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.0005147433588075545, 0.009332254158401674, 0.9062302184360842]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.0009972034043511044, 0.022536213314792688, 0.944169005920495]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.0019839527151132895, 0.061831388118547465, 0.9593757264189868]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.0010827865001340415, 0.029778960745960655, 0.9400837339264302]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.0012342131355002244, 0.028061547942781845, 0.959402680285123]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.021576058326844456, 0.33685118141761794, 0.01792403913756182]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.0047203032784489014, 0.27634363817051033, 0.010167673393256357]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.04790810702556385, 0.6184890720558543, 0.02815755837442578]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.021007160502512556, 0.3480421094347066, 0.017259448788182932]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.0012846228944353054, 0.019095459894714843, 0.951183628596107]
Epoch 5000 RMSE =  0.3934437202404964
Epoch 5100 RMSE =  0.3936700571933224
Epoch 5200 RMSE =  0.39340618679392153
Epoch 5300 RMSE =  0.39337335571158544
Epoch 5400 RMSE =  0.39333052747335084
Epoch 5500 RMSE =  0.3934767510589105
Epoch 5600 RMSE =  0.39289398663548664
Epoch 5700 RMSE =  0.39226129813982563
Epoch 5800 RMSE =  0.39297522796114953
Epoch 5900 RMSE =  0.39323580156633536
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.04139065189740335, 0.5258193456043687, 0.02668847177946154]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.004524922213882964, 0.3778637569897412, 0.010556194090894939]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.038095660148361854, 0.6446802435021359, 0.03152756972546138]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.006408379698485587, 0.5782286091084513, 0.013831399775737205]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.001801151397821503, 0.06582616001166267, 0.9579838277720101]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.005025485715229088, 0.3610332490488697, 0.00969447925685796]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.05316070182533633, 0.728945048932263, 0.03365495078894672]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.061779706096223186, 0.662165983589285, 0.035970778426659]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005708522705156684, 0.4520679948825174, 0.011193776398810138]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.0011497526800061685, 0.01933354901443628, 0.950054703143156]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.09451346046638488, 0.8246609765628354, 0.0458940498031401]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.0023678247928829222, 0.04317618565500215, 0.9496100214757309]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.0015823289913837583, 0.027074585409044415, 0.9364378409946604]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.08814386384161835, 0.7100149298551501, 0.03580543450697698]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0004760229691078432, 0.006466341954872156, 0.9140611816610579]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.0021999384404949216, 0.06797978370192893, 0.9729444958847966]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.004256062078469245, 0.26242566548113105, 0.007697411057541364]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.006326048904892697, 0.4195895641590799, 0.011797347505538281]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.004585468592239865, 0.2762356334047373, 0.010063732589118628]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.00045791047722873647, 0.009194757305733416, 0.9163590816139777]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.0007904312626929255, 0.010476470905601101, 0.9232305488790595]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.002792561629636558, 0.14831649804696667, 0.006006220689473987]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.001268612530368204, 0.030291048395034565, 0.7822818439440434]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.008167028214412245, 0.24528994040109323, 0.9149109415850102]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.0034462038424234935, 0.18226036166725346, 0.007473942865074525]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.0012761210073368725, 0.030034386162081955, 0.9484066191048318]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.0004435255157215665, 0.005912421744572528, 0.8900397002945322]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.11454704990826245, 0.8702380190514353, 0.05320458439824598]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.020413389514304162, 0.3444847642814881, 0.01690328867188503]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.004789303239942035, 0.3727206354420078, 0.00983340890476832]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.04106520883644316, 0.4765647434718959, 0.06144509140587976]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.004925353184907091, 0.36649661187798627, 0.010207308404013404]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.00038472065559820725, 0.00478095750536363, 0.8328836814904204]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.004774174123731471, 0.2586879261290573, 0.008441590852588934]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.006316192476440332, 0.46234089177647714, 0.012057430868374512]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07057353235113384, 0.7547417717571945, 0.03717159856112489]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003710222379820563, 0.2012581235329233, 0.00810472991109706]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.0011915385458154816, 0.023876708814387446, 0.962129727138177]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.008246719103958917, 0.6818708683519082, 0.018397266277809972]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0007741728360363588, 0.021559277816241583, 0.9265810679955521]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.004317415753199004, 0.21917281234734276, 0.008176666740595943]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.004574302129770613, 0.4537216220113748, 0.009905791304103642]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0003073764984127517, 0.0040912986357019895, 0.8657149288564192]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.08824987754659992, 0.855514935881692, 0.045842359113525556]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003815563201218076, 0.24431222121591237, 0.00817792792118581]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.006062506829150038, 0.43363939256000356, 0.012934145474948084]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.0040464233572514545, 0.2525435295960972, 0.008390123415475592]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.0008969905758923304, 0.021967453496201116, 0.9414947887958999]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.055681008534152476, 0.6324168570628493, 0.03303033548442666]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.08760388355940414, 0.7193588094955858, 0.04810062803347328]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.001028945235114693, 0.031133148044725733, 0.9344317247387033]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.005657546283167438, 0.5336977440432847, 0.013827703842844739]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.004119664196284563, 0.22427890049173596, 0.008293621774735525]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.0051900363984658405, 0.5923275711722932, 0.012198467117562641]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.0006042274786744489, 0.010352591402323676, 0.9174396433084795]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.0004772041242095172, 0.009202117427517807, 0.9000681892393383]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.09956348579804009, 0.8340986692946235, 0.052168892230440626]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.044656609698397694, 0.6156185206223027, 0.028059093339316093]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.10942500880346882, 0.8407683212054424, 0.04391484871898112]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.0552267847291756, 0.5828681097600055, 0.033356024734214916]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.001646337914435685, 0.03371114160527471, 0.9288318158517419]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.0052701041285805285, 0.380573288839681, 0.01003176314748289]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.0006741374096411228, 0.015764626031561434, 0.9298219211748932]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.00046658905439895816, 0.009190193870305669, 0.8990361816222464]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.030869366362497576, 0.4587479098743647, 0.019609008193676565]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.0036917364757141366, 0.08628726231292061, 0.9716829648866832]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.000895188001981576, 0.014643784441228581, 0.9421178096774617]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.046705512448349175, 0.60691659519321, 0.027455798559162296]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.002781896205315968, 0.15194287035475762, 0.0064538891341912735]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.0004264963580196439, 0.005855200490259632, 0.9185485497796191]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.02168079123794666, 0.29640278385313185, 0.015377584974980945]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005774595420806249, 0.46264956431100923, 0.01111307061946102]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.0005191816568694782, 0.007369862693378207, 0.9163770731589856]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.0033980534322592838, 0.2738614880772803, 0.009168887152394412]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.00041991452113085935, 0.0053266438059462904, 0.9103553297432715]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.007077051647781007, 0.12104898504344827, 0.42403209074464926]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.06668026475765954, 0.6861967093646957, 0.03572854830774309]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.017941027719172893, 0.408638042170092, 0.019282741971227528]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.001347303367023106, 0.034819595498401607, 0.9656393363206418]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.021036946394037012, 0.33355767840927414, 0.017747549644772175]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.06275260752494842, 0.6999294698349727, 0.029760769288329037]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.10644634899776599, 0.8291120789612955, 0.07046970262089901]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.039373493347203935, 0.5356148926795262, 0.02372716873580305]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.0037458093745530753, 0.23154482347488214, 0.00717516200243172]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.0014385716032825338, 0.031127624000472493, 0.9593465073361791]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.12126426663994, 0.8573264681276361, 0.07734556436971685]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.028547548815053315, 0.5534702719622376, 0.02113007508881128]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.0024089352065989407, 0.04852185415158138, 0.9563361178913711]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07167280781652698, 0.792943163994357, 0.03793432236850445]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0007348729113257682, 0.02560519669044655, 0.9487931547562131]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.03390214807586463, 0.4732056370063385, 0.11150938331397921]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.08436326860193052, 0.8454158000723474, 0.04440279472336419]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.004408650539220581, 0.3297934241170033, 0.009033967997886453]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.11709016069061125, 0.8997903497343344, 0.05239510208156964]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.09192222992807393, 0.8240855936126625, 0.04062840006677866]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.006844096162629706, 0.3029591448834481, 0.01029335469305329]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.0048967789082585075, 0.2572503099727041, 0.008826798203607693]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.004653466135696468, 0.3834223197932197, 0.009553424526021996]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.048671408218375116, 0.5940341158838611, 0.024941060001881867]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.000859104085412532, 0.014405793391168275, 0.9382089161825862]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.001103104796047256, 0.029628764463164346, 0.9584004808251693]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.03463273963557816, 0.5024889919779428, 0.02210961850523235]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00047596085457518047, 0.006560768326568167, 0.9138852763646466]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.003913917486783305, 0.2705706663144961, 0.00787386183294577]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.0011375061120558695, 0.028734597893484648, 0.7936996393380441]
Epoch 6000 RMSE =  0.393131072216753
Epoch 6100 RMSE =  0.39293069740172804
Epoch 6200 RMSE =  0.39288201900940056
Epoch 6300 RMSE =  0.39276810573160836
Epoch 6400 RMSE =  0.39280040571225694
Epoch 6500 RMSE =  0.39267767588194474
Epoch 6600 RMSE =  0.3925268325930141
Epoch 6700 RMSE =  0.39254356037448285
Epoch 6800 RMSE =  0.3925777839571814
Epoch 6900 RMSE =  0.39226744215245396
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.030144783942121348, 0.4803497213297552, 0.01937404606213462]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.002730443217558489, 0.15309291996548055, 0.005878431077746398]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.006713671120000822, 0.303404126706361, 0.010031037722759095]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0002664477849920577, 0.004232924295060604, 0.8689418246229922]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.0011635760067786737, 0.03170972850947725, 0.946510002284343]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.004320620795617392, 0.3280659397431271, 0.00871604372305918]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.0010870937543565999, 0.025095679483555743, 0.9610847423863671]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.06707300800771356, 0.7073940786069364, 0.03357621177878986]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.03397051331922227, 0.5038843998183933, 0.021340759428498265]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.005522986441173051, 0.5549843437016521, 0.013567045086547993]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.006201531448029998, 0.422502336916126, 0.011365497021868773]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.0050675169802983444, 0.6112576810497827, 0.01196811671218759]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00043591810878557054, 0.006453702981111105, 0.9108019925589462]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.004178414844569245, 0.25895893475787957, 0.007419480070863226]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.047830110408374385, 0.5901901676467834, 0.02410323718067334]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.08682431962730396, 0.7090420989024264, 0.034494052682563975]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005596597158434889, 0.44948460118372763, 0.01080279888802666]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.1330639276926533, 0.8728374135614018, 0.06743615544986978]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.004755945767482679, 0.09272256755060432, 0.4894619850747746]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.0373509119190498, 0.6394652079220158, 0.030808638358658977]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.0008080640064850501, 0.023670864001289614, 0.9421146249876253]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.0003822135465409494, 0.005723228215660243, 0.9100253729257078]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00043520618214900356, 0.006508121674110117, 0.912289884854155]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.0015880214712676141, 0.06397527137110283, 0.958391373032771]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.00047234428180645277, 0.007863222980999115, 0.916379248695286]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.0033168610379677383, 0.2837530341461136, 0.009094480766273536]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.08632668102910675, 0.8610818858027461, 0.04567498938862252]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.0012265804428511066, 0.03647199002194316, 0.9649830247837023]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.0004016176510079405, 0.009019231737518568, 0.9173550667447715]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.003946393183621588, 0.26433398712974154, 0.008301734049057794]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.002011751170630666, 0.06784218261174536, 0.9721007001886255]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.020549456255483524, 0.33942104983401966, 0.01723196342238038]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.07025425064960823, 0.792536615250003, 0.036888641394307496]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.0199886188389431, 0.35057752406775144, 0.01659536784050701]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.0402911753864825, 0.5270673672427092, 0.02610460135173755]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.0014325765489161554, 0.02811375803584905, 0.9350615757110616]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.021170735435678906, 0.3151307662651156, 0.015160521944884054]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.09260978584931479, 0.8315790060973061, 0.04467924036196265]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.0006356257317665702, 0.01745967858607638, 0.8552399696255949]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.0044281762641302505, 0.3847452935807112, 0.010281715043823678]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.0037222717575221306, 0.2619871730815412, 0.008105810346584898]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.03236228418992978, 0.4330471577662137, 0.0718769810473839]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.04358141563258794, 0.6475958681284284, 0.027761199500669374]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.004667226045793604, 0.27185752306291855, 0.008281425250972592]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.1106542903867545, 0.8483900037115641, 0.06494272551590183]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.0003879184824476205, 0.0066480575771362205, 0.9179887192612086]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.06898814909690075, 0.7693644643639703, 0.03649488431004682]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.0008086216137943151, 0.023208604771804687, 0.8239409196639056]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.004793237598679621, 0.26587197576153954, 0.008622498238778798]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.00032058297768386315, 0.004706021969635875, 0.8422925154618328]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0006696839124839274, 0.02719614968160617, 0.9478637592836248]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.004699355231547304, 0.17635066272885883, 0.9405535974806444]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.0008143373214505857, 0.01664619693914242, 0.9421919394758653]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.0005954128431867094, 0.01758161174904496, 0.9317128168601033]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.0040131387463974, 0.2462652406545208, 0.008262647524766102]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.006271834365268759, 0.5845248329691258, 0.013547718622152951]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.003660574691697445, 0.2377545177211537, 0.007019354355995868]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.005917688126897901, 0.4545222178032026, 0.012890583510085235]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.005151960607979593, 0.40023487604343216, 0.009980857604231987]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.1150593547632048, 0.8988758161745996, 0.05137913571776623]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.10703381418755303, 0.8519646649195005, 0.043696233988842995]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.003831632360278182, 0.26810146217571695, 0.007705642284850442]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.027930340987971616, 0.5558000634009694, 0.0206768703500477]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.08181611939550794, 0.7253738682076786, 0.04978753121701881]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.0005452720997200734, 0.011665905366993433, 0.9182904618439677]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.004482873100036361, 0.28207944919622596, 0.00984356100499659]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.00040013788034734283, 0.0061776659108225655, 0.8892345825293763]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.04209700893973746, 0.5442148876290787, 0.08881838016544542]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.01749828631871082, 0.42782061250267694, 0.01868214398011858]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.004914206740605615, 0.369179319266182, 0.009402780213407754]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003624982566135893, 0.21004452765519713, 0.007923383824361974]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.003362117239078539, 0.09522249398197123, 0.9713762271570789]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.0004117437546222261, 0.010012355132876228, 0.9006222530507926]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.0010080771331656391, 0.030658088252885764, 0.9572245168023877]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.09726637671562796, 0.8497900452848763, 0.05148174087921039]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.004815553856784367, 0.37370564946267876, 0.010009930364467828]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.05433279742312049, 0.608293768156255, 0.03271774719285478]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.0520253046940012, 0.7354150015517461, 0.032690967920041177]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.0007843662892965506, 0.015025333259786972, 0.93651944871204]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.060626026596980145, 0.6701199498413921, 0.03487484509858619]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.0061672766956933035, 0.4766393165043373, 0.011798366753189564]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.0013903189222516537, 0.03478103533618397, 0.9326263805532701]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.0008817587203482494, 0.03239009192126993, 0.9373261712070031]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.0027143035058872205, 0.16305621362039932, 0.006359650619655134]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.001310537956087661, 0.03277600250404003, 0.9582019893811804]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.061315326693105174, 0.7106517412187964, 0.028888174115666302]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.038485488151663555, 0.5477171378138331, 0.023038934499425744]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.004460105672161318, 0.4758467144585973, 0.009695981818356693]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.0056352082295718685, 0.4811213383065945, 0.010946426637672818]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.0021988808475585505, 0.050103487094145374, 0.955131060460292]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.002161792149148332, 0.044677002660952835, 0.9482261422184197]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.0010492522520900629, 0.020076204313005143, 0.9486881924474383]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.0007212124320553748, 0.011010482163091686, 0.9218394923581746]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.008059530251355003, 0.6917689464117264, 0.018041080205882413]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.0033731800154000804, 0.18614815133313603, 0.007277456361073677]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.11240884742540709, 0.8730930064473467, 0.052282575042671565]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.004215473054582272, 0.22690969044727255, 0.008010391949134528]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.0004116065606950192, 0.009669900784384521, 0.9038603883219731]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.0544357319660803, 0.6496137653377195, 0.032685278951100544]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0006691343189680404, 0.021722175905462552, 0.9286900256578697]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.090289601691899, 0.8228091422786985, 0.03962090970979258]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.004685756296703433, 0.376701220825822, 0.00966688036441801]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.0045598250304433865, 0.38193083778691383, 0.009309158991112948]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.04566517953464854, 0.6202619536752616, 0.027115498432640266]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.08274172478388986, 0.8428514999652202, 0.04327746215168286]
Epoch 7000 RMSE =  0.39247788919262
Epoch 7100 RMSE =  0.3924826789149624
Epoch 7200 RMSE =  0.392247944850516
Epoch 7300 RMSE =  0.392116133645763
Epoch 7400 RMSE =  0.392106790565975
Epoch 7500 RMSE =  0.3922219819907268
Epoch 7600 RMSE =  0.39216709149697826
Epoch 7700 RMSE =  0.39198725793906625
Epoch 7800 RMSE =  0.3920448800446745
Epoch 7900 RMSE =  0.3921756340339468
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.00410748657000216, 0.26048084590220627, 0.00712469766401522]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.004368235679817644, 0.3710356022750543, 0.009783276504320988]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0006211063887905406, 0.025432886227294053, 0.9455745405499442]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.04707355615024371, 0.5899899168821993, 0.02316756631251018]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.000609319687584745, 0.017014325619305475, 0.8476135193040764]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.059801656379251056, 0.6604405333145777, 0.03342279071087368]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.003607118745930181, 0.2321291145321033, 0.006655442867741514]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0005766734715103683, 0.019954750208943595, 0.9303535426502537]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.0033218399494402383, 0.18463858056905458, 0.006949414482459743]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.05318770717947474, 0.599679214057215, 0.03155997326108992]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.0026772379027973654, 0.15934169466375633, 0.006079457737385781]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.0014721229652006917, 0.06437300877948972, 0.956673899122942]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003575557751776964, 0.20485644704184336, 0.007589196705830558]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.004259557184665627, 0.3239139096673552, 0.008397263309865666]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.00037126160105373565, 0.009379704696918066, 0.9001032064315515]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.004405183850019536, 0.46210249919618185, 0.009280177800002499]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.08525277980066137, 0.859563432189702, 0.043463123388906566]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.0005505363952015999, 0.01638190057167708, 0.9292294490437933]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.09154188460888454, 0.8220252025579664, 0.04278537072329908]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.0005061547110462538, 0.011276061426328482, 0.9151013241295219]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.003268201914485957, 0.2808803836631409, 0.008644297677396644]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.004604762980857313, 0.25621058156602144, 0.007916834292594856]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.0003549995377830339, 0.005607426362855195, 0.9068239872651342]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.10572828054540473, 0.8474877835842869, 0.041638793885582365]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.00023613842726936295, 0.00388683153709185, 0.8694804818579935]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005507507694190732, 0.44331050161260727, 0.010325536456338063]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.040521872995258855, 0.5272675527529759, 0.08591496317641886]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.05136476807049454, 0.7246487650707666, 0.031072006879569174]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.002006815876757494, 0.043206264938702985, 0.9458889525205026]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.004494987261382974, 0.37684022730219485, 0.008813320306191662]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.08570802931268756, 0.7050250153351143, 0.032927496443636614]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.00442282027621838, 0.2735830887168579, 0.00926488365668016]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.09604148453148989, 0.8426580448442202, 0.0489596431807666]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.0007284992505842041, 0.014224847901314543, 0.9338084872727215]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.000791750252489893, 0.029769383932201552, 0.9363792741860582]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.0012397881012826575, 0.03174239758578997, 0.9320598070858969]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.001078195349800293, 0.030584406075413666, 0.9448298452972672]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.0013301469316884577, 0.026713862946612096, 0.932346185902349]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.003120224565084006, 0.09101355667004561, 0.9702271817059793]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.02753646344695036, 0.5482904326183181, 0.01951474461181435]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.1133905519963065, 0.8968904757124041, 0.04855502531023393]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.003882500439879947, 0.26279904768569884, 0.007880288303116724]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.006098665595930671, 0.4145079170550884, 0.010878918330430603]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.044988035519650714, 0.6148870665521595, 0.02573059915644748]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.0009350404197269809, 0.029389227680360456, 0.9554610104295189]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.08595901093898121, 0.7329215922699045, 0.04447265630597192]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0004031701741641734, 0.006460230949603593, 0.9080944305854206]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.004613372152108684, 0.3705232844648238, 0.009171076442919186]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.005067773560155712, 0.39070769113211345, 0.009413721167915168]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.017253613031587404, 0.4132757899195615, 0.01779636990363208]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.004747353150787754, 0.36244993875937187, 0.009522996984671973]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.004153535838755888, 0.2213479601317826, 0.007606251272686864]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.004896693299368877, 0.1809755283136969, 0.932018885131918]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.0011388135087150207, 0.03568941411141247, 0.9631939673448797]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.003780298343316132, 0.25972641488080256, 0.007273602444410542]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.00544465413765673, 0.5386153000508335, 0.013002434699615959]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.01972805528555342, 0.33737903703653893, 0.01577902405147873]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.0003744615607306738, 0.008818277050962518, 0.9131327049890198]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.0055628804716290675, 0.4636924124585089, 0.010336596848163853]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.08175235266702888, 0.8360585734072272, 0.04086591354922644]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.0032516006507451248, 0.065420933247794, 0.5477423277556205]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.1243754027180233, 0.8615870758867155, 0.068165886710603]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.04001207728208883, 0.5127299886483981, 0.024754756297436616]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.06810154381595927, 0.7536329487728004, 0.0347932989951541]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.10758696021804835, 0.8356441362902389, 0.06276324520839634]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.110653683241661, 0.8698559306798809, 0.04990330384829238]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.000284687522250895, 0.004180229415710169, 0.8411388781407043]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.004971893399270577, 0.6047499031097125, 0.011522551785829763]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00040277370596664597, 0.006306143937435086, 0.9087237824168238]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.0007469135521775943, 0.022915847017793634, 0.9398967967968562]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.0005648759638485766, 0.015721116930290262, 0.852075381428768]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.0007548588766157058, 0.015314625034448122, 0.9398902568971731]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.03339770139400557, 0.49051868371610097, 0.020618254691107084]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.002035708598330821, 0.0480629561862886, 0.9537752649456535]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.006600898772403624, 0.2932940961970609, 0.00959740057924551]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.0010061834414443272, 0.024200205240290852, 0.9603224816286049]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.03272258508537992, 0.42586250435250056, 0.06709514238256019]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.005818854891321588, 0.4432447487582835, 0.012260412910426954]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.06042167631096089, 0.6987377794057806, 0.027706052812088277]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.0026870414131660494, 0.14709340006287785, 0.0056209277505258354]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.020851263565637337, 0.30295267984506546, 0.014515128231289023]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.042924885312991344, 0.6338766279046925, 0.026574465768547224]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.007929025394233288, 0.6869797444364073, 0.01727361281989719]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.0003648195385973207, 0.009113589988646226, 0.9043251160421584]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003666931636389645, 0.25110312142045077, 0.007754329128393986]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.06610843719895741, 0.7013889338179132, 0.03244484226889844]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.08899546717715294, 0.8193742182994319, 0.03796319422637671]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.003952484879740331, 0.23485147469130563, 0.007859418139993045]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.0006684599700682185, 0.010662593738225846, 0.9195459111188159]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.05361126494139903, 0.6458176928564128, 0.031324065699107236]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.020216842049705235, 0.34027962357178776, 0.016522490841403874]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.006076102010336168, 0.46769028438962296, 0.011327645265075853]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.004726312199445017, 0.25369691467323774, 0.008233547768514175]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.0012163403091469804, 0.031729768962808526, 0.9570012225975221]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.00097332980951103, 0.0195683108653345, 0.9472040245116214]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.006182040499381633, 0.5709151987643759, 0.012901618983838592]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.06927805863892378, 0.7880939178218644, 0.0354150402975177]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.03679338307331438, 0.6372701531076314, 0.029436937286347456]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.0003702251751159445, 0.006024992113049065, 0.8855012396076689]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.0018655363993611556, 0.06816959571138521, 0.9709336830545086]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.0004379576626731528, 0.0078346870492387, 0.912347010229659]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.004844752862833284, 0.3570428250006058, 0.008963657616592879]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.03797431081100499, 0.5342897338546773, 0.02193854537442299]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.0003603923566791007, 0.006271547462581256, 0.9147012229558579]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.029724027639489545, 0.4754549328280433, 0.018419688052097677]
Epoch 8000 RMSE =  0.3921299217744847
Epoch 8100 RMSE =  0.3921346271850368
Epoch 8200 RMSE =  0.3920807314677741
Epoch 8300 RMSE =  0.3920513519881369
Epoch 8400 RMSE =  0.3920001518803613
Epoch 8500 RMSE =  0.39203524808261575
Epoch 8600 RMSE =  0.3918487967495367
Epoch 8700 RMSE =  0.3919035872042311
Epoch 8800 RMSE =  0.39167200142989644
Epoch 8900 RMSE =  0.391864000782616
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.007847906248779013, 0.6858547570478216, 0.016538060466202813]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.020025330240985288, 0.3373039360759109, 0.015801497129105505]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.003736530120101621, 0.2649838713928101, 0.0070134084488433605]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.0032315423922313247, 0.2819546478388261, 0.008276333299138362]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.04658088820837718, 0.5898945413385952, 0.022281922351338405]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.0654893636734556, 0.7042635197498185, 0.031048437482254634]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.027246832966639312, 0.5544583734253873, 0.01883944421482604]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.0042079991614728885, 0.3266299304602893, 0.008051534487542556]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.017078279364895328, 0.4218970491437956, 0.017171904992370216]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.09056181409804433, 0.8272901568206257, 0.04102957558598348]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00037842236468497303, 0.006673589001412303, 0.9069155627568506]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.004694979266560356, 0.37122413584646063, 0.009191767434888714]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.005765662174583579, 0.45029165843980873, 0.011737574249185985]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.0036314996961254674, 0.2540017451937358, 0.007417464515577289]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.000628216854480319, 0.010760502385807126, 0.9175930641089634]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.03314596294155361, 0.4987667799546671, 0.019755121496454052]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.08431061066015103, 0.8629393063617203, 0.041745435618007164]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.0007017866504035193, 0.0239653987460914, 0.9387375770221764]
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.00041091410763022327, 0.007981954715095326, 0.9113316160430942]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.004569597944826242, 0.3760246690250372, 0.008864370272660174]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.0003327067225924316, 0.0057566957828577, 0.9049211662206195]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.037590846549818306, 0.5391704448438632, 0.02120679678783346]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.001374567890372687, 0.06464989765849452, 0.9560630432318532]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.0005137569223026999, 0.016787127765606636, 0.9281833238052405]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.04455541066225884, 0.624313451596843, 0.02493592674791445]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0005815736110611719, 0.026284051352938296, 0.9449277334749885]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.09505842602362569, 0.8487501895321625, 0.04756146999597312]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.08456316308081283, 0.714718148410595, 0.03201969610751749]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.05074073279069184, 0.7348278939505737, 0.03016096524478694]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.002923123316600524, 0.09574202369319924, 0.9699798697037267]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.11217192360062701, 0.9011240639973026, 0.04707478681077151]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.0003439851749047695, 0.00968304496146948, 0.8991355615011346]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.0060031936716203075, 0.47408612483711915, 0.010872882841772907]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.0008757848897893276, 0.03051393551700159, 0.9551598040211859]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.0005172057219752117, 0.015735146269806002, 0.8565961708530052]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.03634324174006315, 0.6466409788785225, 0.028321599980803323]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.004364548355291486, 0.28370186530114505, 0.009009415170548126]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.0012431321792018251, 0.027869629875680618, 0.9322143998895835]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.0003460821758993801, 0.009238231919011821, 0.9134670132859044]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.0812753855595183, 0.7318285821621136, 0.044644227020453085]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.0006818259630685057, 0.014970844044189306, 0.9330642460848393]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.0049199968660062245, 0.6173111443263563, 0.011052420065406368]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.0011392411891336823, 0.03242513875988584, 0.955829252104468]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.004312274670505088, 0.37629394410088823, 0.009401053374200318]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00037809211416862666, 0.006581218901502022, 0.9069311894988751]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.06733369023498079, 0.7605582598727526, 0.03348138389757972]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.003098230178345905, 0.1321876124624262, 0.9496087030216922]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.004785005395397135, 0.36162947649726346, 0.008640296427138799]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.0015855949169544035, 0.035365374004940095, 0.677416448401921]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.0018787896238383244, 0.043848075710465124, 0.9458991612193759]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.00046712740646406696, 0.014259743382664197, 0.863860303907071]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.0006957815826360034, 0.02849483462246522, 0.9396402919172852]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.029370565906005182, 0.4773990116969916, 0.017913008478063503]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.059062976262434654, 0.6647395616104693, 0.03240644390706551]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.00264212041790945, 0.16159849831873196, 0.005885237543459076]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.0010091373942928692, 0.03176952169472769, 0.9449352725801746]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.00537279374837863, 0.5525935400453545, 0.012660677924621987]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.0007089239594569979, 0.01594221830647721, 0.9392591523584604]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.00391156806133202, 0.23771823457372723, 0.007592167551675171]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.0035656664799119443, 0.2319793390445252, 0.006449240684891765]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.00454991511461936, 0.25970169263741605, 0.0076554364565848635]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.0009455834007349262, 0.024702074506607524, 0.9598867209549256]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.00033110329150119023, 0.008834320166379937, 0.9058632931361693]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.10459714406889889, 0.8498108567466242, 0.04031615580357397]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.019475365345948306, 0.34739880410207347, 0.015382586529514323]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.00033775256474698564, 0.006352446143641471, 0.9147385900121254]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.05982687362470105, 0.7063143431345887, 0.02682994748187258]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.0424856863313797, 0.6392282575392522, 0.025733890391337874]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.001911520238171642, 0.05019436285157475, 0.9533791032149235]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.10961609827336014, 0.8750684120448221, 0.048594998949881854]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.004054277183492731, 0.2655172030700054, 0.006927129899246573]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.0032778841745431703, 0.18785074384910974, 0.006739620358348043]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.05301519884506978, 0.6527702511969553, 0.03031141959287066]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0005228726570286041, 0.01977337540509047, 0.9317532703839337]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.1061657213928452, 0.8421939058194329, 0.06123486334128178]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.05213654027971416, 0.605635361721641, 0.030787481692480828]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.000910382970238454, 0.020374028088164915, 0.9466695938945952]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005429019892453646, 0.45797211380606323, 0.010091814164375284]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.0044326155296567525, 0.38668128459377477, 0.00861770391368263]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.0061038666618982, 0.5766083395339656, 0.012471809046003509]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.039624406109875054, 0.5243387753327414, 0.024025929963598522]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.0017474191788686914, 0.06942545690291609, 0.9710172551136902]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.0880336301828884, 0.8231020029866225, 0.03675652281888498]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.00434022961329581, 0.4678676774055208, 0.008992599203323645]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.06842947518179851, 0.7923268330296572, 0.03425763042866514]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.0040978496064732394, 0.22695102381585755, 0.007420288745376786]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.00021628199609222254, 0.0038944226600710332, 0.8705091273708502]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.03009135105434358, 0.4632231978864209, 0.1039177581369652]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.00026843585438783427, 0.004361997706768657, 0.8384295037903202]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.020614446410937046, 0.3113791379352306, 0.013927509215036002]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.0038373037304638667, 0.2713635637957309, 0.0076269994698813]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.0026570574005607577, 0.1530820253509434, 0.005391644745977065]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005486526841397253, 0.47832271575025903, 0.010065857495441194]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.001066513793637902, 0.037227358053198774, 0.9628217468149964]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.0003461109393806891, 0.006134181909114399, 0.883379918899802]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.005011962154939017, 0.39894804698519665, 0.009024861309285887]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.004676471608809009, 0.25486233583015006, 0.00782157237066356]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.006544373502943948, 0.29796906727334793, 0.009122751666762728]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.003536201554857494, 0.20315808325131482, 0.007231407543761475]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.00047475086625508754, 0.011309057017814645, 0.91252841501764]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.08086031202025142, 0.8407803611754968, 0.039511824300377835]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.04147423909015608, 0.49811101617762943, 0.05213359249533575]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.006039866298128887, 0.4195612365529385, 0.0104399541731832]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.00114278946003604, 0.031617828027071686, 0.931788436267698]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.1331923732581426, 0.8765393520525606, 0.06079947090330351]
Epoch 9000 RMSE =  0.3916401036131736
Epoch 9100 RMSE =  0.39181913283262815
Epoch 9200 RMSE =  0.3917008688622102
Epoch 9300 RMSE =  0.3917938601975591
Epoch 9400 RMSE =  0.3916404151639288
Epoch 9500 RMSE =  0.3917105157912161
Epoch 9600 RMSE =  0.3916950815245612
Epoch 9700 RMSE =  0.39164629529273687
Epoch 9800 RMSE =  0.39165211382078907
Epoch 9900 RMSE =  0.39154036735077735
Sample [5.9 3.  5.1 1.8] expected [0. 0. 1.] produced [0.0003878909103818295, 0.007715420292576583, 0.9090675302659359]
Sample [6.5 3.  5.2 2. ] expected [0. 0. 1.] produced [0.0004840045719597134, 0.016170506494863467, 0.9263173118261259]
Sample [6.4 2.7 5.3 1.9] expected [0. 0. 1.] produced [0.0004474631223056111, 0.01115708495948958, 0.9113523239360691]
Sample [6.  2.2 5.  1.5] expected [0. 0. 1.] produced [0.00024846696264583243, 0.004094757540032454, 0.837623392362973]
Sample [6.4 3.2 4.5 1.5] expected [0. 1. 0.] produced [0.09424105990256354, 0.8440110034359888, 0.04549972243369713]
Sample [4.6 3.4 1.4 0.3] expected [1. 0. 0.] produced [0.004329578541843839, 0.2749232370300813, 0.008583438340478356]
Sample [5.6 2.9 3.6 1.3] expected [0. 1. 0.] produced [0.03607241964385601, 0.6372163695434474, 0.02700544179505429]
Sample [5.8 2.7 3.9 1.2] expected [0. 1. 0.] produced [0.04213044189382926, 0.6333493644073532, 0.02440113375706428]
Sample [7.3 2.9 6.3 1.8] expected [0. 0. 1.] produced [0.001174537013788923, 0.026944957498286792, 0.9303719497344441]
Sample [5.1 2.5 3.  1.1] expected [0. 1. 0.] produced [0.01690605319923074, 0.42039126554279344, 0.016467897638590027]
Sample [6.7 3.1 4.4 1.4] expected [0. 1. 0.] produced [0.08347662015920555, 0.8633515016552606, 0.04002976665214567]
Sample [5.  3.3 1.4 0.2] expected [1. 0. 0.] produced [0.004163943410354253, 0.3271506206411294, 0.007717020052656057]
Sample [6.7 3.  5.2 2.3] expected [0. 0. 1.] produced [0.0005481534889035145, 0.02568420393819769, 0.9436622702791082]
Sample [5.2 3.4 1.4 0.2] expected [1. 0. 0.] produced [0.004398793218373166, 0.3824582319244888, 0.008167487025354088]
Sample [7.2 3.6 6.1 2.5] expected [0. 0. 1.] produced [0.0016519672015258807, 0.06796296346567947, 0.9699741202932878]
Sample [5.7 2.8 4.1 1.3] expected [0. 1. 0.] produced [0.05266056751634554, 0.6486427717080595, 0.02874037018637758]
Sample [6.2 2.2 4.5 1.5] expected [0. 1. 0.] produced [0.06513685998417681, 0.7075272347637239, 0.029678246141116884]
Sample [5.1 3.7 1.5 0.4] expected [1. 0. 0.] produced [0.005705883642232422, 0.450041097288466, 0.011247564735783599]
Sample [6.3 2.7 4.9 1.8] expected [0. 0. 1.] produced [0.00047961261514969255, 0.014927189522666844, 0.8556084038752005]
Sample [4.6 3.6 1.  0.2] expected [1. 0. 0.] produced [0.003198901529907358, 0.2839644625956344, 0.007946637566001991]
Sample [5.2 4.1 1.5 0.1] expected [1. 0. 0.] produced [0.0059611856233809666, 0.46412898789411566, 0.010400614566257625]
Sample [4.4 3.  1.3 0.2] expected [1. 0. 0.] produced [0.003254640913863771, 0.182319780052856, 0.006398358582366065]
Sample [7.7 3.8 6.7 2.2] expected [0. 0. 1.] produced [0.0027659674825148916, 0.09043245829185995, 0.9693464412701501]
Sample [6.1 2.8 4.  1.3] expected [0. 1. 0.] produced [0.05038019365771884, 0.7250165317212718, 0.02889730414280971]
Sample [6.2 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.06685875880606137, 0.7564517188257385, 0.0321754797343498]
Sample [6.7 3.1 5.6 2.4] expected [0. 0. 1.] produced [0.0008285011946445461, 0.02929980458446515, 0.9541704280179492]
Sample [6.4 2.9 4.3 1.3] expected [0. 1. 0.] produced [0.06796000003165517, 0.7899494011335021, 0.03257217273292896]
Sample [6.3 3.3 4.7 1.6] expected [0. 1. 0.] produced [0.10945504792480762, 0.843795449774736, 0.056357795736188314]
Sample [5.1 3.4 1.5 0.2] expected [1. 0. 0.] produced [0.004741892614400753, 0.3580978636690957, 0.008291402719699859]
Sample [4.8 3.  1.4 0.3] expected [1. 0. 0.] produced [0.0038058058780646815, 0.2625536101181985, 0.007301904082676777]
Sample [5.3 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.005391932774230321, 0.44449322177753237, 0.00957815797990579]
Sample [4.9 3.1 1.5 0.1] expected [1. 0. 0.] produced [0.004025423952693772, 0.2550907588690043, 0.006574426921637077]
Sample [4.4 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.0035019504273821585, 0.19890398937249668, 0.006985074067534532]
Sample [6.5 2.8 4.6 1.5] expected [0. 1. 0.] produced [0.08983607276712852, 0.8191607438500805, 0.03943822496634331]
Sample [6.7 3.  5.  1.7] expected [0. 1. 0.] produced [0.12838199370473152, 0.8698707687586684, 0.06027649025878475]
Sample [5.5 3.5 1.3 0.2] expected [1. 0. 0.] produced [0.004303632179881291, 0.4557761515724648, 0.00853126400247294]
Sample [4.6 3.1 1.5 0.2] expected [1. 0. 0.] produced [0.00406498330070176, 0.2180022764528953, 0.00703976512607253]
Sample [6.7 3.3 5.7 2.5] expected [0. 0. 1.] produced [0.0010072148545901321, 0.03479674681168822, 0.9620294437446595]
Sample [6.  3.  4.8 1.8] expected [0. 0. 1.] produced [0.0003134223024831394, 0.00844432855016107, 0.9028454731294221]
Sample [5.8 2.6 4.  1.2] expected [0. 1. 0.] produced [0.04410926798506007, 0.6098705418402637, 0.02388522604598655]
Sample [6.3 2.9 5.6 1.8] expected [0. 0. 1.] produced [0.0005920418797024763, 0.01029172742572477, 0.9163505642672393]
Sample [5.4 3.9 1.3 0.4] expected [1. 0. 0.] produced [0.005325898679312502, 0.5380086721410036, 0.012051420892375217]
Sample [5.7 2.5 5.  2. ] expected [0. 0. 1.] produced [0.0003138055833941292, 0.005403801465986292, 0.9033870816442201]
Sample [6.3 2.5 4.9 1.5] expected [0. 1. 0.] produced [0.03936863983304049, 0.5243870806372236, 0.08036624249141054]
Sample [4.7 3.2 1.3 0.2] expected [1. 0. 0.] produced [0.0035938869185648116, 0.2461390816147743, 0.007125862931210593]
Sample [5.  2.  3.5 1. ] expected [0. 1. 0.] produced [0.020466746296856943, 0.298237698163454, 0.01335316908160539]
Sample [4.8 3.4 1.9 0.2] expected [1. 0. 0.] produced [0.006482332574385691, 0.29194134500465296, 0.00882382590578263]
Sample [5.  3.5 1.3 0.3] expected [1. 0. 0.] produced [0.0042816491949880485, 0.3639898629343845, 0.009041014977142495]
Sample [4.9 2.4 3.3 1. ] expected [0. 1. 0.] produced [0.019861497378819863, 0.33159779029236663, 0.01519000238976927]
Sample [7.2 3.2 6.  1.8] expected [0. 0. 1.] produced [0.00105621572384574, 0.029972988044686465, 0.932537310570984]
Sample [6.3 2.5 5.  1.9] expected [0. 0. 1.] produced [0.0003272026762368253, 0.00904816683609621, 0.8971261226127153]
Sample [5.7 4.4 1.5 0.4] expected [1. 0. 0.] produced [0.007788429821883068, 0.680288357715813, 0.015919530821411714]
Sample [5.6 2.5 3.9 1.1] expected [0. 1. 0.] produced [0.03728960385106989, 0.5275356613133315, 0.020364159668288712]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.0003579625796202042, 0.006269414532630777, 0.9059253561473727]
Sample [6.1 2.8 4.7 1.2] expected [0. 1. 0.] produced [0.08437458552944846, 0.7040806879760496, 0.030641993467732734]
Sample [6.6 2.9 4.6 1.3] expected [0. 1. 0.] produced [0.08748463549408494, 0.817212111692893, 0.03504349675721773]
Sample [5.4 3.7 1.5 0.2] expected [1. 0. 0.] produced [0.0054442717225373334, 0.4661802880265557, 0.0096678594989317]
Sample [5.6 2.8 4.9 2. ] expected [0. 0. 1.] produced [0.00031931706085647823, 0.00603007151838461, 0.9125852796774938]
Sample [5.8 4.  1.2 0.2] expected [1. 0. 0.] produced [0.004888484345800089, 0.6013880548301849, 0.010658647590808385]
Sample [4.6 3.2 1.4 0.2] expected [1. 0. 0.] produced [0.00388308845372069, 0.2276430820491972, 0.007242325465699821]
Sample [7.1 3.  5.9 2.1] expected [0. 0. 1.] produced [0.0009557298097325844, 0.02936012289647245, 0.9436617217607088]
Sample [5.8 2.7 5.1 1.9] expected [0. 0. 1.] produced [0.00035798303298661944, 0.006138356435112475, 0.9061350227029896]
Sample [6.4 2.8 5.6 2.1] expected [0. 0. 1.] produced [0.0006461635664524705, 0.013649879254508973, 0.932606040899735]
Sample [6.5 3.  5.8 2.2] expected [0. 0. 1.] produced [0.0008635292762146363, 0.018511779406663438, 0.945391220701789]
Sample [6.  2.7 5.1 1.6] expected [0. 1. 0.] produced [0.0003272062549282547, 0.005684830709424917, 0.8821897104249105]
Sample [6.3 2.8 5.1 1.5] expected [0. 0. 1.] produced [0.002341388208911284, 0.050944499534937925, 0.5871873873992836]
Sample [5.  3.6 1.4 0.2] expected [1. 0. 0.] produced [0.004660652465850646, 0.35381844479575725, 0.008876124610285618]
Sample [7.7 3.  6.1 2.3] expected [0. 0. 1.] produced [0.001297022425640582, 0.05963770430423152, 0.9556333553522747]
Sample [6.3 3.4 5.6 2.4] expected [0. 0. 1.] produced [0.0008942799021908987, 0.02324010647872226, 0.9590163067594387]
Sample [6.8 3.2 5.9 2.3] expected [0. 0. 1.] produced [0.0010796511352410803, 0.029794278712127118, 0.9555458052226006]
Sample [5.7 2.8 4.5 1.3] expected [0. 1. 0.] produced [0.03600490487253722, 0.4502381155308798, 0.05659370689067853]
Sample [4.9 3.  1.4 0.2] expected [1. 0. 0.] produced [0.0037091263767396266, 0.25637128203791043, 0.006776250095166208]
Sample [6.5 3.2 5.1 2. ] expected [0. 0. 1.] produced [0.0005068768431341842, 0.018598601408073154, 0.9289780073920908]
Sample [4.8 3.1 1.6 0.2] expected [1. 0. 0.] produced [0.004518396312815106, 0.2498872270043259, 0.007326973183342619]
Sample [5.7 2.9 4.2 1.3] expected [0. 1. 0.] produced [0.05907193483297346, 0.6494999220685652, 0.030926311816425904]
Sample [5.1 3.5 1.4 0.2] expected [1. 0. 0.] produced [0.004537023943405218, 0.3617402630513245, 0.008552536746920714]
Sample [5.6 2.7 4.2 1.3] expected [0. 1. 0.] produced [0.0533987055262989, 0.5922284019790159, 0.028768285269377903]
Sample [4.8 3.  1.4 0.1] expected [1. 0. 0.] produced [0.0035432280554219746, 0.22301023939854858, 0.006171018270011646]
Sample [6.7 3.1 4.7 1.5] expected [0. 1. 0.] produced [0.10911117043475291, 0.8664605983279, 0.04651530887696272]
Sample [5.8 2.7 4.1 1. ] expected [0. 1. 0.] produced [0.04622233938918354, 0.5810960380269652, 0.021541537352264432]
Sample [4.3 3.  1.1 0.1] expected [1. 0. 0.] produced [0.002626109379985731, 0.15318304538391392, 0.005633054474089106]
Sample [7.7 2.8 6.7 2. ] expected [0. 0. 1.] produced [0.0017795419143840194, 0.04172384202103919, 0.944977346275742]
Sample [5.5 4.2 1.4 0.2] expected [1. 0. 0.] produced [0.006071482463650174, 0.5621233123719189, 0.011920576574516204]
Sample [7.  3.2 4.7 1.4] expected [0. 1. 0.] produced [0.11162931834243503, 0.8927130955524657, 0.04541938273343719]
Sample [5.7 2.6 3.5 1. ] expected [0. 1. 0.] produced [0.02703692281786604, 0.5403112745432385, 0.01821438020558172]
Sample [7.6 3.  6.6 2.1] expected [0. 0. 1.] produced [0.0018094054726878214, 0.046729647457403205, 0.9523074155182996]
Sample [5.2 3.5 1.5 0.2] expected [1. 0. 0.] produced [0.004976490314313659, 0.38510890919289026, 0.00877830289213629]
Sample [5.1 3.8 1.6 0.2] expected [1. 0. 0.] produced [0.005993011690817315, 0.40552410073903594, 0.010152420937340749]
Sample [6.1 2.9 4.7 1.4] expected [0. 1. 0.] produced [0.08517057396370549, 0.7256987391570096, 0.04132300727902151]
Sample [4.5 2.3 1.3 0.3] expected [1. 0. 0.] produced [0.0026413374550395166, 0.1423857956261416, 0.0051960821395664325]
Sample [5.5 2.3 4.  1.3] expected [0. 1. 0.] produced [0.03981552412931384, 0.511721484300074, 0.02276746489969009]
Sample [4.7 3.2 1.6 0.2] expected [1. 0. 0.] produced [0.004643492289491168, 0.24520932124758477, 0.007605876433580848]
Sample [6.3 2.3 4.4 1.3] expected [0. 1. 0.] produced [0.059473323374001776, 0.6933153796274354, 0.025652908538212414]
Sample [6.2 2.8 4.8 1.8] expected [0. 0. 1.] produced [0.0005725919254259946, 0.01746453274911822, 0.8341504468547202]
Sample [5.5 2.4 3.8 1.1] expected [0. 1. 0.] produced [0.03288074519229422, 0.48810740881854464, 0.01911914380905871]
Sample [6.8 3.  5.5 2.1] expected [0. 0. 1.] produced [0.0006636608385333702, 0.022779026279138054, 0.9384528227230662]
Sample [5.9 3.2 4.8 1.8] expected [0. 1. 0.] produced [0.0003280412777939815, 0.008662250524966242, 0.9124129720462842]
Sample [5.  2.3 3.3 1. ] expected [0. 1. 0.] produced [0.01934569384424062, 0.3412921620631518, 0.014641268022454804]
Sample [6.9 3.1 5.4 2.1] expected [0. 0. 1.] produced [0.0006721390615252845, 0.028338132904277724, 0.937094458983512]
Sample [6.4 2.8 5.6 2.2] expected [0. 0. 1.] produced [0.0006711273207582627, 0.015474897531092678, 0.9376537714105682]
Sample [6.6 3.  4.4 1.4] expected [0. 1. 0.] produced [0.08022563286225351, 0.8398617999167693, 0.038282633121595536]
Sample [5.4 3.  4.5 1.5] expected [0. 1. 0.] produced [0.0002065676814345615, 0.0038306543081882806, 0.8662501443397327]
Sample [6.8 2.8 4.8 1.4] expected [0. 1. 0.] produced [0.10381268192227747, 0.8488032221068194, 0.038065932760241054]
Sample [5.5 2.4 3.7 1. ] expected [0. 1. 0.] produced [0.029148568860783614, 0.4736781345764613, 0.01691278815708811]
Sample [7.9 3.8 6.4 2. ] expected [0. 0. 1.] produced [0.0031903584638879995, 0.139837595025811, 0.9446730296999175]
Epoch 10000 RMSE =  0.39167260391932307
Final Epoch RMSE =  0.39167260391932307
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.], Produced [0.007474180157273131, 0.5480608987151635, 0.012938706642953213]
Sample [5.  3.4 1.5 0.2] Expected [1. 0. 0.], Produced [0.0047075922025523265, 0.33760811628910226, 0.008179785617927041]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.], Produced [0.003414221178703417, 0.17549352790726103, 0.006198512890940496]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.], Produced [0.004026540120763533, 0.26032846668881193, 0.006526496346600209]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.], Produced [0.005032592586311128, 0.29734103218318547, 0.008241120933203783]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.], Produced [0.004700111982174186, 0.3943020606061595, 0.009201286299443476]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.], Produced [0.007129462496996957, 0.5795757196364173, 0.01156456055787639]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.], Produced [0.0057083268142843185, 0.4399681044973795, 0.010707304406635535]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.], Produced [0.005777129269191007, 0.4259144377438712, 0.008726970840221804]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.], Produced [0.00606733533051555, 0.40806015906653587, 0.010652919482692093]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.], Produced [0.004425178560437606, 0.2844239405946476, 0.006997839179842654]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.], Produced [0.005521529238527087, 0.3790365371569068, 0.009937766000257559]
Sample [5.4 3.4 1.5 0.4] Expected [1. 0. 0.], Produced [0.005258007678727, 0.4698508746918368, 0.009987429709283823]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.], Produced [0.003393850111619135, 0.30975977642594144, 0.007055924937829764]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.], Produced [0.004026540120763533, 0.26032846668881193, 0.006526496346600209]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.], Produced [0.0061724424118751835, 0.4373866021405071, 0.012361557219958427]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.], Produced [0.00830566592296007, 0.4630280192017119, 0.012648145362727861]
Sample [6.9 3.1 4.9 1.5] Expected [0. 1. 0.], Produced [0.1284598905939516, 0.8918498998098214, 0.0482504110785205]
Sample [5.2 2.7 3.9 1.4] Expected [0. 1. 0.], Produced [0.025767930293532772, 0.4088739837024633, 0.04259140999467877]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.], Produced [0.0666420149632464, 0.7509233771749844, 0.03816569674124087]
Sample [6.  2.2 4.  1. ] Expected [0. 1. 0.], Produced [0.036276412293469125, 0.5624344995162412, 0.01696757061286374]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.], Produced [0.00041628399398431666, 0.009358661728006284, 0.7850710071046997]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.], Produced [0.07534569662881596, 0.7388639833569032, 0.042071260251703074]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.], Produced [0.08361502913169747, 0.7909882342585665, 0.061512541537927214]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.], Produced [0.05587314441933955, 0.6552815892240896, 0.031014809071739622]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.], Produced [0.042736155321201626, 0.5577712434602934, 0.02444502328820159]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.], Produced [0.027482415697675962, 0.35311182936736846, 0.0480729025317961]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.], Produced [0.08818290130996605, 0.7661844224632111, 0.038861785928221135]
Sample [5.7 3.  4.2 1.2] Expected [0. 1. 0.], Produced [0.059087810392279415, 0.6570096320739947, 0.029138260329001787]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.], Produced [0.0012547368113200931, 0.025417892835331045, 0.9629117487347747]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.], Produced [0.00017067752022439098, 0.0020569273301367856, 0.8590041615693275]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.], Produced [0.0006285258649825669, 0.012109128043976524, 0.9063182457361623]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.], Produced [0.0004476897941626288, 0.010739263068135612, 0.9390315629471709]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.], Produced [0.0006246565310080149, 0.021861390292755148, 0.9479242651337414]
Sample [6.5 3.  5.5 1.8] Expected [0. 0. 1.], Produced [0.0005758383776556268, 0.0136838082287483, 0.9181301811008278]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.], Produced [0.0021892807401361256, 0.0498840374422625, 0.9542948519820623]
Sample [6.9 3.2 5.7 2.3] Expected [0. 0. 1.], Produced [0.000917565497070306, 0.03454606393065188, 0.9531743681593315]
Sample [6.7 3.3 5.7 2.1] Expected [0. 0. 1.], Produced [0.0008672500994048966, 0.025723194169331123, 0.946280797617724]
Sample [6.1 3.  4.9 1.8] Expected [0. 0. 1.], Produced [0.000347035685919228, 0.009789699299625986, 0.903680543189389]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.], Produced [0.047911970417630206, 0.626592129816202, 0.2349025367728398]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.], Produced [0.0010244055915916604, 0.030934501983076245, 0.9292802995725026]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.], Produced [0.00044851367574249434, 0.005165442265537239, 0.8679785203373185]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.], Produced [0.0005914088409375718, 0.013255706787245726, 0.9210142932920943]
Sample [6.9 3.1 5.1 2.3] Expected [0. 0. 1.], Produced [0.0007250866058537397, 0.04394177960955563, 0.930146737906392]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.], Produced [0.0007189242030177524, 0.02058870040553125, 0.952473935075609]
Final Test RMSE =  0.4203491813055856
Sample [0.99] expected [0.83602598] produced [0.7952086285447592]
Sample [1.28] expected [0.95801586] produced [0.8272278102344484]
Sample [0.84] expected [0.74464312] produced [0.7770623542009896]
Sample [0.07] expected [0.06994285] produced [0.6620466627327155]
Sample [0.89] expected [0.77707175] produced [0.7821600916974619]
Sample [1.04] expected [0.86240423] produced [0.800185909419519]
Sample [1.] expected [0.84147098] produced [0.7956197446711255]
Sample [0.35] expected [0.34289781] produced [0.7065471746170341]
Sample [1.23] expected [0.9424888] produced [0.820668740521277]
Sample [1.52] expected [0.99871014] produced [0.8490242735976805]
Sample [0.75] expected [0.68163876] produced [0.7640647920105226]
Sample [0.54] expected [0.51413599] produced [0.7346900432307892]
Sample [0.46] expected [0.44394811] produced [0.7224454576002433]
Sample [1.21] expected [0.935616] produced [0.8179138248783783]
Sample [0.19] expected [0.18885889] produced [0.6797829836186357]
Epoch 0 RMSE =  0.24950015082395519
Epoch 100 RMSE =  0.2304648880654213
Epoch 200 RMSE =  0.22955259038630857
Epoch 300 RMSE =  0.2286572343998155
Epoch 400 RMSE =  0.2276848132837909
Epoch 500 RMSE =  0.2265510640421847
Epoch 600 RMSE =  0.22512538174540572
Epoch 700 RMSE =  0.2232468066985505
Epoch 800 RMSE =  0.22058972784693107
Epoch 900 RMSE =  0.21664094445530674
Sample [0.46] expected [0.44394811] produced [0.6164504613575906]
Sample [0.99] expected [0.83602598] produced [0.7150898088286259]
Sample [1.28] expected [0.95801586] produced [0.7562823601127489]
Sample [1.52] expected [0.99871014] produced [0.7849011804222744]
Sample [0.54] expected [0.51413599] produced [0.6345290967995998]
Sample [1.04] expected [0.86240423] produced [0.7238718766037848]
Sample [1.21] expected [0.935616] produced [0.7483221545487497]
Sample [0.75] expected [0.68163876] produced [0.676441319557202]
Sample [1.23] expected [0.9424888] produced [0.7514892618326117]
Sample [0.19] expected [0.18885889] produced [0.5560497724296705]
Sample [0.07] expected [0.06994285] produced [0.5257519517751205]
Sample [1.] expected [0.84147098] produced [0.7170861614428969]
Sample [0.35] expected [0.34289781] produced [0.5920135035329117]
Sample [0.84] expected [0.74464312] produced [0.6905034644611501]
Sample [0.89] expected [0.77707175] produced [0.699183494922925]
Epoch 1000 RMSE =  0.21058407804200355
Epoch 1100 RMSE =  0.20148650523088846
Epoch 1200 RMSE =  0.1889891302719852
Epoch 1300 RMSE =  0.17402840406702638
Epoch 1400 RMSE =  0.1582950479367038
Epoch 1500 RMSE =  0.14311431508378003
Epoch 1600 RMSE =  0.1291233775624414
Epoch 1700 RMSE =  0.11651706572597684
Epoch 1800 RMSE =  0.10529773520999565
Epoch 1900 RMSE =  0.09540359029578958
Sample [0.07] expected [0.06994285] produced [0.23952103271744338]
Sample [1.] expected [0.84147098] produced [0.7860408186662541]
Sample [0.75] expected [0.68163876] produced [0.6866510404867399]
Sample [1.21] expected [0.935616] produced [0.8399005187037296]
Sample [0.19] expected [0.18885889] produced [0.31635473062450625]
Sample [0.46] expected [0.44394811] produced [0.5103693031237589]
Sample [0.99] expected [0.83602598] produced [0.7827645907267701]
Sample [1.23] expected [0.9424888] produced [0.8438894491882939]
Sample [0.89] expected [0.77707175] produced [0.7479562815484292]
Sample [1.28] expected [0.95801586] produced [0.8535793841356827]
Sample [1.52] expected [0.99871014] produced [0.8890645388474553]
Sample [0.84] expected [0.74464312] produced [0.728153940880783]
Sample [0.35] expected [0.34289781] produced [0.4313982065948153]
Sample [0.54] expected [0.51413599] produced [0.5652808658671246]
Sample [1.04] expected [0.86240423] produced [0.7981920161517638]
Epoch 2000 RMSE =  0.08674156197775086
Epoch 2100 RMSE =  0.07919941961088722
Epoch 2200 RMSE =  0.0726585558973867
Epoch 2300 RMSE =  0.06699742407391124
Epoch 2400 RMSE =  0.06210157209008283
Epoch 2500 RMSE =  0.05786701927937681
Epoch 2600 RMSE =  0.054200665114632056
Epoch 2700 RMSE =  0.0510232569954468
Epoch 2800 RMSE =  0.04826456362321849
Epoch 2900 RMSE =  0.0458666406808824
Sample [0.46] expected [0.44394811] produced [0.46895249676821865]
Sample [0.99] expected [0.83602598] produced [0.8180473722406776]
Sample [1.52] expected [0.99871014] produced [0.9221313354814269]
Sample [0.07] expected [0.06994285] produced [0.14678054718009645]
Sample [1.21] expected [0.935616] produced [0.8774773222891452]
Sample [1.] expected [0.84147098] produced [0.8216029044627442]
Sample [1.28] expected [0.95801586] produced [0.8905839774120264]
Sample [0.35] expected [0.34289781] produced [0.36437740083747977]
Sample [1.23] expected [0.9424888] produced [0.8815118706037086]
Sample [0.84] expected [0.74464312] produced [0.7542729631656997]
Sample [0.89] expected [0.77707175] produced [0.778254189349707]
Sample [0.54] expected [0.51413599] produced [0.5428124542202069]
Sample [0.75] expected [0.68163876] produced [0.7031956600373033]
Sample [1.04] expected [0.86240423] produced [0.834641670616011]
Sample [0.19] expected [0.18885889] produced [0.22598103345172657]
Epoch 3000 RMSE =  0.04377841522087856
Epoch 3100 RMSE =  0.04195727715674131
Epoch 3200 RMSE =  0.04036652279048818
Epoch 3300 RMSE =  0.03897485749144317
Epoch 3400 RMSE =  0.03775539529637947
Epoch 3500 RMSE =  0.036685158214334324
Epoch 3600 RMSE =  0.03574433293761818
Epoch 3700 RMSE =  0.03491581608592063
Epoch 3800 RMSE =  0.0341848365216672
Epoch 3900 RMSE =  0.03353873279793876
Sample [0.19] expected [0.18885889] produced [0.19626768211994808]
Sample [1.23] expected [0.9424888] produced [0.8943168769014568]
Sample [0.89] expected [0.77707175] produced [0.7893622317361925]
Sample [1.21] expected [0.935616] produced [0.8904790906785935]
Sample [0.75] expected [0.68163876] produced [0.7093063470384205]
Sample [0.07] expected [0.06994285] produced [0.12001937665368041]
Sample [1.04] expected [0.86240423] produced [0.8477300343056978]
Sample [0.35] expected [0.34289781] produced [0.33900576456475245]
Sample [1.] expected [0.84147098] produced [0.8344487068182233]
Sample [0.84] expected [0.74464312] produced [0.7639520012599392]
Sample [0.46] expected [0.44394811] produced [0.452170594533684]
Sample [1.28] expected [0.95801586] produced [0.9030913861968568]
Sample [0.54] expected [0.51413599] produced [0.5332105479614694]
Sample [1.52] expected [0.99871014] produced [0.9328496422642724]
Sample [0.99] expected [0.83602598] produced [0.8308864706994781]
Epoch 4000 RMSE =  0.03296641389426636
Epoch 4100 RMSE =  0.0324583313229489
Epoch 4200 RMSE =  0.03200636727960364
Epoch 4300 RMSE =  0.03160317668535813
Epoch 4400 RMSE =  0.031242626750425505
Epoch 4500 RMSE =  0.030919370563548026
Epoch 4600 RMSE =  0.0306287211689714
Epoch 4700 RMSE =  0.030366610175868853
Epoch 4800 RMSE =  0.030129311570773306
Epoch 4900 RMSE =  0.029914307050299214
Sample [1.] expected [0.84147098] produced [0.8403034246676652]
Sample [0.07] expected [0.06994285] produced [0.11000655694327964]
Sample [0.54] expected [0.51413599] produced [0.5281179972910011]
Sample [1.04] expected [0.86240423] produced [0.8537319214322349]
Sample [0.46] expected [0.44394811] produced [0.44396208159455514]
Sample [0.99] expected [0.83602598] produced [0.8366504889159553]
Sample [1.23] expected [0.9424888] produced [0.9003708621020386]
Sample [1.28] expected [0.95801586] produced [0.9090749845045756]
Sample [0.19] expected [0.18885889] produced [0.18425595497863984]
Sample [0.75] expected [0.68163876] produced [0.7115666799361003]
Sample [1.21] expected [0.935616] produced [0.8965641741214498]
Sample [0.84] expected [0.74464312] produced [0.7681474608381901]
Sample [0.89] expected [0.77707175] produced [0.794209454138428]
Sample [0.35] expected [0.34289781] produced [0.32757978121884024]
Sample [1.52] expected [0.99871014] produced [0.9380607186847414]
Epoch 5000 RMSE =  0.029718403048607237
Epoch 5100 RMSE =  0.029539426942595787
Epoch 5200 RMSE =  0.029375341784941913
Epoch 5300 RMSE =  0.029224394936355043
Epoch 5400 RMSE =  0.029085105023414548
Epoch 5500 RMSE =  0.028956062599615997
Epoch 5600 RMSE =  0.028836054091187224
Epoch 5700 RMSE =  0.028724140341856115
Epoch 5800 RMSE =  0.028619348527382835
Epoch 5900 RMSE =  0.02852098860607805
Sample [0.89] expected [0.77707175] produced [0.7965386149686096]
Sample [0.46] expected [0.44394811] produced [0.43948755038057175]
Sample [0.19] expected [0.18885889] produced [0.1790750335239098]
Sample [0.75] expected [0.68163876] produced [0.7121582374348719]
Sample [1.] expected [0.84147098] produced [0.8432359591679428]
Sample [0.84] expected [0.74464312] produced [0.769856339263246]
Sample [0.99] expected [0.83602598] produced [0.8395282420014638]
Sample [0.54] expected [0.51413599] produced [0.5248100859481318]
Sample [0.07] expected [0.06994285] produced [0.10600580480093946]
Sample [1.23] expected [0.9424888] produced [0.9037175327863639]
Sample [1.52] expected [0.99871014] produced [0.9411581354419732]
Sample [0.35] expected [0.34289781] produced [0.32195643310341404]
Sample [1.21] expected [0.935616] produced [0.8999529696466085]
Sample [1.28] expected [0.95801586] produced [0.9124637560700274]
Sample [1.04] expected [0.86240423] produced [0.8569275969675677]
Epoch 6000 RMSE =  0.028428160005971474
Epoch 6100 RMSE =  0.02834066556777605
Epoch 6200 RMSE =  0.02825756411544564
Epoch 6300 RMSE =  0.028178522360665783
Epoch 6400 RMSE =  0.028103237934912418
Epoch 6500 RMSE =  0.028031217661060818
Epoch 6600 RMSE =  0.027962199692909858
Epoch 6700 RMSE =  0.02789589209667412
Epoch 6800 RMSE =  0.027832057155742683
Epoch 6900 RMSE =  0.027770488074879078
Sample [1.04] expected [0.86240423] produced [0.8587312912608397]
Sample [0.75] expected [0.68163876] produced [0.7121125546386551]
Sample [1.] expected [0.84147098] produced [0.8449049706213455]
Sample [1.52] expected [0.99871014] produced [0.9433361766558731]
Sample [0.54] expected [0.51413599] produced [0.522814131955967]
Sample [0.89] expected [0.77707175] produced [0.7975500600955576]
Sample [0.19] expected [0.18885889] produced [0.17697770673825294]
Sample [0.35] expected [0.34289781] produced [0.31920918074834]
Sample [0.99] expected [0.83602598] produced [0.8412186315234111]
Sample [0.46] expected [0.44394811] produced [0.4368288820994576]
Sample [1.21] expected [0.935616] produced [0.9021271642331832]
Sample [1.28] expected [0.95801586] produced [0.9146823707805668]
Sample [0.84] expected [0.74464312] produced [0.7707074146462487]
Sample [1.23] expected [0.9424888] produced [0.9059766129303318]
Sample [0.07] expected [0.06994285] produced [0.10475401713301771]
Epoch 7000 RMSE =  0.02771093850995077
Epoch 7100 RMSE =  0.02765331883787044
Epoch 7200 RMSE =  0.027597465973845107
Epoch 7300 RMSE =  0.027543134015952507
Epoch 7400 RMSE =  0.027490410782405886
Epoch 7500 RMSE =  0.027439047786087564
Epoch 7600 RMSE =  0.02738895592100824
Epoch 7700 RMSE =  0.027340080285715617
Epoch 7800 RMSE =  0.0272921928208933
Epoch 7900 RMSE =  0.027245648324229946
Sample [1.23] expected [0.9424888] produced [0.9074817164986859]
Sample [0.99] expected [0.83602598] produced [0.842197094355936]
Sample [1.21] expected [0.935616] produced [0.9036302508791803]
Sample [0.84] expected [0.74464312] produced [0.7708918914720936]
Sample [0.75] expected [0.68163876] produced [0.7116947308598308]
Sample [1.28] expected [0.95801586] produced [0.9162002224378977]
Sample [0.07] expected [0.06994285] produced [0.10463122698258483]
Sample [0.54] expected [0.51413599] produced [0.5212163899962129]
Sample [1.] expected [0.84147098] produced [0.845856580110106]
Sample [1.52] expected [0.99871014] produced [0.9449484302594379]
Sample [1.04] expected [0.86240423] produced [0.8597924115361578]
Sample [0.46] expected [0.44394811] produced [0.43514326201426334]
Sample [0.19] expected [0.18885889] produced [0.1764331397328144]
Sample [0.89] expected [0.77707175] produced [0.798054089752421]
Sample [0.35] expected [0.34289781] produced [0.31783741916177516]
Epoch 8000 RMSE =  0.027199966851039387
Epoch 8100 RMSE =  0.027155272634081178
Epoch 8200 RMSE =  0.02711147688692049
Epoch 8300 RMSE =  0.027068531490106136
Epoch 8400 RMSE =  0.027026412570852216
Epoch 8500 RMSE =  0.026985066160067038
Epoch 8600 RMSE =  0.02694453026966105
Epoch 8700 RMSE =  0.026904695048824173
Epoch 8800 RMSE =  0.02686550283584963
Epoch 8900 RMSE =  0.026827053797470194
Sample [1.28] expected [0.95801586] produced [0.9174064109283102]
Sample [0.99] expected [0.83602598] produced [0.8427449923130821]
Sample [0.54] expected [0.51413599] produced [0.5200863541655929]
Sample [1.23] expected [0.9424888] produced [0.9086034637366585]
Sample [1.04] expected [0.86240423] produced [0.860569646059129]
Sample [0.84] expected [0.74464312] produced [0.7707767533016788]
Sample [0.07] expected [0.06994285] produced [0.10508284323313787]
Sample [0.89] expected [0.77707175] produced [0.7981686805106765]
Sample [0.35] expected [0.34289781] produced [0.31710381935524035]
Sample [0.75] expected [0.68163876] produced [0.7111106742822145]
Sample [1.21] expected [0.935616] produced [0.904658799546024]
Sample [1.52] expected [0.99871014] produced [0.9462400551601137]
Sample [0.19] expected [0.18885889] produced [0.17653714819354213]
Sample [0.46] expected [0.44394811] produced [0.4340306483895928]
Sample [1.] expected [0.84147098] produced [0.8465007630421348]
Epoch 9000 RMSE =  0.02678928900318997
Epoch 9100 RMSE =  0.026752037774103475
Epoch 9200 RMSE =  0.02671552528946296
Epoch 9300 RMSE =  0.0266795610488556
Epoch 9400 RMSE =  0.026644146728604344
Epoch 9500 RMSE =  0.026609258939403947
Epoch 9600 RMSE =  0.026574920589803878
Epoch 9700 RMSE =  0.02654118999790064
Epoch 9800 RMSE =  0.026507927793144533
Epoch 9900 RMSE =  0.02647514204941205
Sample [0.46] expected [0.44394811] produced [0.43321042294767403]
Sample [0.19] expected [0.18885889] produced [0.17698230173575746]
Sample [0.89] expected [0.77707175] produced [0.798232939643888]
Sample [0.84] expected [0.74464312] produced [0.7705162103052764]
Sample [0.54] expected [0.51413599] produced [0.5189650508070607]
Sample [1.21] expected [0.935616] produced [0.9055334516150353]
Sample [1.23] expected [0.9424888] produced [0.9094713144099852]
Sample [1.28] expected [0.95801586] produced [0.9183552690247507]
Sample [0.99] expected [0.83602598] produced [0.8430972048737737]
Sample [1.52] expected [0.99871014] produced [0.9473395183199099]
Sample [0.75] expected [0.68163876] produced [0.7106106703498147]
Sample [0.35] expected [0.34289781] produced [0.31681076981938394]
Sample [1.04] expected [0.86240423] produced [0.8610667574473951]
Sample [0.07] expected [0.06994285] produced [0.10576775464821501]
Sample [1.] expected [0.84147098] produced [0.8468913053868504]
Epoch 10000 RMSE =  0.02644289698129146
Final Epoch RMSE =  0.02644289698129146
Sample [0.] Expected [0.], Produced [0.07675028263515006]
Sample [0.01] Expected [0.00999983], Produced [0.08039907225154536]
Sample [0.02] Expected [0.01999867], Produced [0.08420500189040814]
Sample [0.03] Expected [0.0299955], Produced [0.08817276682776279]
Sample [0.04] Expected [0.03998933], Produced [0.09230700056823585]
Sample [0.05] Expected [0.04997917], Produced [0.09661225260786353]
Sample [0.06] Expected [0.05996401], Produced [0.10109296478017823]
Sample [0.08] Expected [0.07991469], Produced [0.11059784715720695]
Sample [0.09] Expected [0.08987855], Produced [0.11563013122558978]
Sample [0.1] Expected [0.09983342], Produced [0.12085404709460246]
Sample [0.11] Expected [0.1097783], Produced [0.12627309888276436]
Sample [0.12] Expected [0.11971221], Produced [0.13189051593599377]
Sample [0.13] Expected [0.12963414], Produced [0.1377092220305203]
Sample [0.14] Expected [0.13954311], Produced [0.14373180424284854]
Sample [0.15] Expected [0.14943813], Produced [0.1499604817349127]
Sample [0.16] Expected [0.15931821], Produced [0.15639707472675526]
Sample [0.17] Expected [0.16918235], Produced [0.16304297395158107]
Sample [0.18] Expected [0.17902957], Produced [0.16989911090824802]
Sample [0.2] Expected [0.19866933], Produced [0.18424335760992655]
Sample [0.21] Expected [0.2084599], Produced [0.19173078435483346]
Sample [0.22] Expected [0.21822962], Produced [0.1994270344000449]
Sample [0.23] Expected [0.22797752], Produced [0.2073303486712722]
Sample [0.24] Expected [0.23770263], Produced [0.21543836642916103]
Sample [0.25] Expected [0.24740396], Produced [0.2237481108407687]
Sample [0.26] Expected [0.25708055], Produced [0.2322559781105154]
Sample [0.27] Expected [0.26673144], Produced [0.24095773046319918]
Sample [0.28] Expected [0.27635565], Produced [0.2498484932385567]
Sample [0.29] Expected [0.28595223], Produced [0.25892275631731104]
Sample [0.3] Expected [0.29552021], Produced [0.26817438005305066]
Sample [0.31] Expected [0.30505864], Produced [0.27759660583323187]
Sample [0.32] Expected [0.31456656], Produced [0.2871820713368328]
Sample [0.33] Expected [0.32404303], Produced [0.2969228304966479]
Sample [0.34] Expected [0.33348709], Produced [0.30681037811198436]
Sample [0.36] Expected [0.35227423], Produced [0.326989201460486]
Sample [0.37] Expected [0.36161543], Produced [0.3372609549396071]
Sample [0.38] Expected [0.37092047], Produced [0.3476405313711479]
Sample [0.39] Expected [0.38018842], Produced [0.3581171500511536]
Sample [0.4] Expected [0.38941834], Produced [0.3686797055040416]
Sample [0.41] Expected [0.39860933], Produced [0.3793168179271056]
Sample [0.42] Expected [0.40776045], Produced [0.3900168857136586]
Sample [0.43] Expected [0.4168708], Produced [0.40076813953241114]
Sample [0.44] Expected [0.42593947], Produced [0.411558697420773]
Sample [0.45] Expected [0.43496553], Produced [0.4223766203391856]
Sample [0.47] Expected [0.45288629], Produced [0.44404685185280207]
Sample [0.48] Expected [0.46177918], Produced [0.4548754924152499]
Sample [0.49] Expected [0.47062589], Produced [0.46568426758487613]
Sample [0.5] Expected [0.47942554], Produced [0.4764617643254084]
Sample [0.51] Expected [0.48817725], Produced [0.48719682558212984]
Sample [0.52] Expected [0.49688014], Produced [0.49787859461684697]
Sample [0.53] Expected [0.50553334], Produced [0.5084965560639337]
Sample [0.55] Expected [0.52268723], Produced [0.5295009228208681]
Sample [0.56] Expected [0.5311862], Produced [0.5398683227196246]
Sample [0.57] Expected [0.53963205], Produced [0.5501339597303552]
Sample [0.58] Expected [0.54802394], Produced [0.5602895102148916]
Sample [0.59] Expected [0.55636102], Produced [0.5703271578368414]
Sample [0.6] Expected [0.56464247], Produced [0.580239607059679]
Sample [0.61] Expected [0.57286746], Produced [0.5900200926960542]
Sample [0.62] Expected [0.58103516], Produced [0.5996623856457555]
Sample [0.63] Expected [0.58914476], Produced [0.6091607949956415]
Sample [0.64] Expected [0.59719544], Produced [0.6185101666854211]
Sample [0.65] Expected [0.60518641], Produced [0.6277058789683279]
Sample [0.66] Expected [0.61311685], Produced [0.6367438349154966]
Sample [0.67] Expected [0.62098599], Produced [0.6456204522273443]
Sample [0.68] Expected [0.62879302], Produced [0.6543326506247134]
Sample [0.69] Expected [0.63653718], Produced [0.6628778370972787]
Sample [0.7] Expected [0.64421769], Produced [0.6712538892871387]
Sample [0.71] Expected [0.65183377], Produced [0.6794591372820217]
Sample [0.72] Expected [0.65938467], Produced [0.6874923440856412]
Sample [0.73] Expected [0.66686964], Produced [0.6953526850228782]
Sample [0.74] Expected [0.67428791], Produced [0.703039726325169]
Sample [0.76] Expected [0.68892145], Produced [0.7178939970901675]
Sample [0.77] Expected [0.69613524], Produced [0.7250621138501321]
Sample [0.78] Expected [0.70327942], Produced [0.7320586604720911]
Sample [0.79] Expected [0.71035327], Produced [0.738884823072888]
Sample [0.8] Expected [0.71735609], Produced [0.7455420447578124]
Sample [0.81] Expected [0.72428717], Produced [0.7520320039982397]
Sample [0.82] Expected [0.73114583], Produced [0.7583565935604251]
Sample [0.83] Expected [0.73793137], Produced [0.7645179000790655]
Sample [0.85] Expected [0.75128041], Produced [0.776359862430389]
Sample [0.86] Expected [0.75784256], Produced [0.7820454875206151]
Sample [0.87] Expected [0.76432894], Produced [0.7875777327904055]
Sample [0.88] Expected [0.77073888], Produced [0.7929593750491541]
Sample [0.9] Expected [0.78332691], Produced [0.8032823845148859]
Sample [0.91] Expected [0.78950374], Produced [0.808229689562738]
Sample [0.92] Expected [0.79560162], Produced [0.813038241063444]
Sample [0.93] Expected [0.80161994], Produced [0.8177111213701258]
Sample [0.94] Expected [0.8075581], Produced [0.82225143772824]
Sample [0.95] Expected [0.8134155], Produced [0.8266623122300819]
Sample [0.96] Expected [0.81919157], Produced [0.8309468725870325]
Sample [0.97] Expected [0.82488571], Produced [0.8351082436881675]
Sample [0.98] Expected [0.83049737], Produced [0.8391495399116659]
Sample [1.01] Expected [0.84683184], Produced [0.8505838237792652]
Sample [1.02] Expected [0.85210802], Produced [0.8541755241281082]
Sample [1.03] Expected [0.85729899], Produced [0.857662342924313]
Sample [1.05] Expected [0.86742323], Produced [0.864332999528084]
Sample [1.06] Expected [0.87235548], Produced [0.867522550497136]
Sample [1.07] Expected [0.8772005], Produced [0.8706186407314858]
Sample [1.08] Expected [0.88195781], Produced [0.8736239964736869]
Sample [1.09] Expected [0.88662691], Produced [0.8765412882685103]
Sample [1.1] Expected [0.89120736], Produced [0.8793731295286515]
Sample [1.11] Expected [0.89569869], Produced [0.8821220754095929]
Sample [1.12] Expected [0.90010044], Produced [0.8847906219667994]
Sample [1.13] Expected [0.90441219], Produced [0.887381205569905]
Sample [1.14] Expected [0.9086335], Produced [0.8898962025500121]
Sample [1.15] Expected [0.91276394], Produced [0.8923379290576925]
Sample [1.16] Expected [0.91680311], Produced [0.8947086411106936]
Sample [1.17] Expected [0.9207506], Produced [0.8970105348117563]
Sample [1.18] Expected [0.92460601], Produced [0.8992457467182897]
Sample [1.19] Expected [0.92836897], Produced [0.9014163543469524]
Sample [1.2] Expected [0.93203909], Produced [0.9035243767974381]
Sample [1.22] Expected [0.93909936], Produced [0.9075604549399672]
Sample [1.24] Expected [0.945784], Produced [0.9113689954713636]
Sample [1.25] Expected [0.94898462], Produced [0.9131923897024671]
Sample [1.26] Expected [0.95209034], Produced [0.9149641331222571]
Sample [1.27] Expected [0.95510086], Produced [0.9166858606161138]
Sample [1.29] Expected [0.96083506], Produced [0.919985555266116]
Sample [1.3] Expected [0.96355819], Produced [0.9215665436115095]
Sample [1.31] Expected [0.96618495], Produced [0.9231035607926124]
Sample [1.32] Expected [0.9687151], Produced [0.9245980002518315]
Sample [1.33] Expected [0.97114838], Produced [0.926051210740807]
Sample [1.34] Expected [0.97348454], Produced [0.9274644975264855]
Sample [1.35] Expected [0.97572336], Produced [0.9288391235920348]
Sample [1.36] Expected [0.9778646], Produced [0.9301763108293751]
Sample [1.37] Expected [0.97990806], Produced [0.9314772412204899]
Sample [1.38] Expected [0.98185353], Produced [0.9327430580050378]
Sample [1.39] Expected [0.98370081], Produced [0.933974866832111]
Sample [1.4] Expected [0.98544973], Produced [0.9351737368942813]
Sample [1.41] Expected [0.9871001], Produced [0.9363407020423514]
Sample [1.42] Expected [0.98865176], Produced [0.9374767618794649]
Sample [1.43] Expected [0.99010456], Produced [0.9385828828334576]
Sample [1.44] Expected [0.99145835], Produced [0.9396599992065345]
Sample [1.45] Expected [0.99271299], Produced [0.9407090142015312]
Sample [1.46] Expected [0.99386836], Produced [0.9417308009241898]
Sample [1.47] Expected [0.99492435], Produced [0.942726203361021]
Sample [1.48] Expected [0.99588084], Produced [0.9436960373324553]
Sample [1.49] Expected [0.99673775], Produced [0.9446410914211086]
Sample [1.5] Expected [0.99749499], Produced [0.9455621278750809]
Sample [1.51] Expected [0.99815247], Produced [0.9464598834863124]
Sample [1.53] Expected [0.99916795], Produced [0.9481883771638928]
Sample [1.54] Expected [0.99952583], Produced [0.9490204690917583]
Sample [1.55] Expected [0.99978376], Produced [0.9498319894845441]
Sample [1.56] Expected [0.99994172], Produced [0.950623560166332]
Sample [1.57] Expected [0.99999968], Produced [0.9513957822614063]
Final Test RMSE =  0.031219566915702626
Sample [1. 0.] expected [1.] produced [0.7365183837240322]
Sample [1. 1.] expected [0.] produced [0.841047392234971]
Sample [0. 0.] expected [0.] produced [0.5607331848037854]
Sample [0. 1.] expected [1.] produced [0.706556523537172]
Epoch 0 RMSE =  0.5425205195472859
Epoch 100 RMSE =  0.5295531846251248
Epoch 200 RMSE =  0.5211198358549987
Epoch 300 RMSE =  0.5161190748152366
Epoch 400 RMSE =  0.5133376379266376
Epoch 500 RMSE =  0.5118620876720882
Epoch 600 RMSE =  0.5111086099778731
Epoch 700 RMSE =  0.5107342944454317
Epoch 800 RMSE =  0.5105530077065289
Epoch 900 RMSE =  0.5104663845970021
Sample [1. 1.] expected [0.] produced [0.6451326505545857]
Sample [0. 0.] expected [0.] produced [0.385005868698227]
Sample [0. 1.] expected [1.] produced [0.45753971871769505]
Sample [1. 0.] expected [1.] produced [0.5716919753444624]
Epoch 1000 RMSE =  0.5104254540324344
Epoch 1100 RMSE =  0.5104063337861131
Epoch 1200 RMSE =  0.5103974197699337
Epoch 1300 RMSE =  0.5103929713858832
Epoch 1400 RMSE =  0.5103905381611085
Epoch 1500 RMSE =  0.5103896301749972
Epoch 1600 RMSE =  0.5103884241255912
Epoch 1700 RMSE =  0.5103878019189679
Epoch 1800 RMSE =  0.5103868904603709
Epoch 1900 RMSE =  0.5103865131606777
Sample [1. 1.] expected [0.] produced [0.6367463902277847]
Sample [0. 0.] expected [0.] produced [0.38028892337509324]
Sample [0. 1.] expected [1.] produced [0.44873907624547144]
Sample [1. 0.] expected [1.] produced [0.566386479006568]
Epoch 2000 RMSE =  0.5103858644096934
Epoch 2100 RMSE =  0.5103850195457642
Epoch 2200 RMSE =  0.5103846154043231
Epoch 2300 RMSE =  0.510383692027861
Epoch 2400 RMSE =  0.5103829340893429
Epoch 2500 RMSE =  0.510381673120739
Epoch 2600 RMSE =  0.5103806201535086
Epoch 2700 RMSE =  0.5103795988647438
Epoch 2800 RMSE =  0.5103781046069771
Epoch 2900 RMSE =  0.510377643817553
Sample [1. 1.] expected [0.] produced [0.6361671549132204]
Sample [0. 0.] expected [0.] produced [0.38029950082840047]
Sample [1. 0.] expected [1.] produced [0.5666282312396362]
Sample [0. 1.] expected [1.] produced [0.4479247242222193]
Epoch 3000 RMSE =  0.5103759789126813
Epoch 3100 RMSE =  0.5103746669076799
Epoch 3200 RMSE =  0.5103732566573246
Epoch 3300 RMSE =  0.5103711162609362
Epoch 3400 RMSE =  0.510370346766764
Epoch 3500 RMSE =  0.510367844775019
Epoch 3600 RMSE =  0.510366927529815
Epoch 3700 RMSE =  0.5103650116684083
Epoch 3800 RMSE =  0.5103630300295747
Epoch 3900 RMSE =  0.5103608936620865
Sample [0. 0.] expected [0.] produced [0.38098927565832597]
Sample [1. 1.] expected [0.] produced [0.6351699941591511]
Sample [0. 1.] expected [1.] produced [0.4457569317092209]
Sample [1. 0.] expected [1.] produced [0.5686240300137136]
Epoch 4000 RMSE =  0.510358784551804
Epoch 4100 RMSE =  0.5103560103696075
Epoch 4200 RMSE =  0.5103531515557888
Epoch 4300 RMSE =  0.5103511656754972
Epoch 4400 RMSE =  0.5103474503016037
Epoch 4500 RMSE =  0.5103454156596176
Epoch 4600 RMSE =  0.5103419151768694
Epoch 4700 RMSE =  0.5103385707833371
Epoch 4800 RMSE =  0.5103345473689823
Epoch 4900 RMSE =  0.5103315709158961
Sample [1. 1.] expected [0.] produced [0.6347627724421119]
Sample [1. 0.] expected [1.] produced [0.5700043622724493]
Sample [0. 0.] expected [0.] produced [0.3806171394672465]
Sample [0. 1.] expected [1.] produced [0.44408013173532185]
Epoch 5000 RMSE =  0.5103274273764751
Epoch 5100 RMSE =  0.5103233260841489
Epoch 5200 RMSE =  0.5103184691343764
Epoch 5300 RMSE =  0.5103136792903097
Epoch 5400 RMSE =  0.5103088698919086
Epoch 5500 RMSE =  0.5103036648486832
Epoch 5600 RMSE =  0.5102978648739749
Epoch 5700 RMSE =  0.5102918542347405
Epoch 5800 RMSE =  0.5102856010740424
Epoch 5900 RMSE =  0.5102789079823631
Sample [1. 0.] expected [1.] produced [0.5730813235755232]
Sample [0. 1.] expected [1.] produced [0.442847707195502]
Sample [0. 0.] expected [0.] produced [0.38188819032388316]
Sample [1. 1.] expected [0.] produced [0.6348152578638911]
Epoch 6000 RMSE =  0.5102713090070284
Epoch 6100 RMSE =  0.5102644977261515
Epoch 6200 RMSE =  0.5102566003029922
Epoch 6300 RMSE =  0.5102481045432431
Epoch 6400 RMSE =  0.5102392627410818
Epoch 6500 RMSE =  0.5102297461777207
Epoch 6600 RMSE =  0.5102195139837695
Epoch 6700 RMSE =  0.5102092014991234
Epoch 6800 RMSE =  0.5101977167012782
Epoch 6900 RMSE =  0.510185789445144
Sample [0. 0.] expected [0.] produced [0.38081989541565814]
Sample [1. 0.] expected [1.] produced [0.5757280921355373]
Sample [0. 1.] expected [1.] produced [0.43867884805326646]
Sample [1. 1.] expected [0.] produced [0.6332406950084719]
Epoch 7000 RMSE =  0.5101729260713872
Epoch 7100 RMSE =  0.510159157553643
Epoch 7200 RMSE =  0.5101447791294637
Epoch 7300 RMSE =  0.5101302459266069
Epoch 7400 RMSE =  0.5101143402166622
Epoch 7500 RMSE =  0.5100970830211904
Epoch 7600 RMSE =  0.5100791716084009
Epoch 7700 RMSE =  0.5100601650701416
Epoch 7800 RMSE =  0.5100394576298218
Epoch 7900 RMSE =  0.5100187072254213
Sample [1. 0.] expected [1.] produced [0.5796299072485298]
Sample [0. 0.] expected [0.] produced [0.38073662942309716]
Sample [1. 1.] expected [0.] produced [0.629726654020007]
Sample [0. 1.] expected [1.] produced [0.4324100197158393]
Epoch 8000 RMSE =  0.5099964314499044
Epoch 8100 RMSE =  0.5099722917192903
Epoch 8200 RMSE =  0.509947082441781
Epoch 8300 RMSE =  0.5099194996252764
Epoch 8400 RMSE =  0.5098910530780879
Epoch 8500 RMSE =  0.5098616641335267
Epoch 8600 RMSE =  0.5098297811199467
Epoch 8700 RMSE =  0.5097955991889782
Epoch 8800 RMSE =  0.5097606042459112
Epoch 8900 RMSE =  0.5097222961960916
Sample [1. 1.] expected [0.] produced [0.6260110953140448]
Sample [0. 0.] expected [0.] produced [0.3790432850607905]
Sample [1. 0.] expected [1.] produced [0.5832092326775431]
Sample [0. 1.] expected [1.] produced [0.42569449681890403]
Epoch 9000 RMSE =  0.5096825134618854
Epoch 9100 RMSE =  0.509640748150819
Epoch 9200 RMSE =  0.5095962497113649
Epoch 9300 RMSE =  0.5095486184715615
Epoch 9400 RMSE =  0.5094991324591551
Epoch 9500 RMSE =  0.5094462882862311
Epoch 9600 RMSE =  0.5093913593572883
Epoch 9700 RMSE =  0.5093329294853305
Epoch 9800 RMSE =  0.5092709754207452
Epoch 9900 RMSE =  0.5092060585326592
Sample [1. 0.] expected [1.] produced [0.5895783975366904]
Sample [0. 1.] expected [1.] produced [0.41847064401279316]
Sample [1. 1.] expected [0.] produced [0.6220546954114589]
Sample [0. 0.] expected [0.] produced [0.3785613082543227]
Epoch 10000 RMSE =  0.5091372584662494
Final Epoch RMSE =  0.5091372584662494
"""




