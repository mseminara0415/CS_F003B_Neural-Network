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

Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.999163239978931, 0.9996023115698532, 0.9996148954044229]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [0.999724013765846, 0.9999162721844579, 0.9999291692360766]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9950543234105856, 0.99534862020543, 0.9926212176314804]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.9987295377721613, 0.9993513750980668, 0.9993520196477046]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [0.9997871546131196, 0.9999101222602466, 0.9999395506715598]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9947239856249738, 0.9949966162899978, 0.9912168231589538]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.999024645134908, 0.9995833783530735, 0.9995204504583651]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.996560102891105, 0.9972350438765428, 0.9932780130046569]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9957728472992293, 0.9963889187484413, 0.9920054619672196]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.998999503348596, 0.9994206733333242, 0.9996436113450172]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9964768424454326, 0.9972049828240672, 0.9933627742962281]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.9986822940930917, 0.9992361426666105, 0.9992712744761241]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.998267264526649, 0.9988543739357674, 0.9989097782070995]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.9991965103443992, 0.9995765098825966, 0.9995946986870671]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9907790879346394, 0.9906043366216187, 0.9888001851983098]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9958082398864332, 0.9962405249547981, 0.9927322233267339]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9954941313689514, 0.9961097214085137, 0.9920433547639054]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.9988242371139928, 0.9994260948951849, 0.9993198647708591]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [0.9995522056133187, 0.9998229410699908, 0.9998537371358546]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9947180836443363, 0.9949739163816439, 0.9911427504419472]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [0.999335096771285, 0.9997480120629799, 0.9997153847773773]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [0.9990926793588322, 0.999536840900331, 0.9996912479687938]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.9984055967875007, 0.9990436322593859, 0.9991175898322152]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.9971172201320211, 0.9981830250173829, 0.9981344940103123]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9970318640329966, 0.9978737603108326, 0.9949520074807084]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [0.9998488824153218, 0.9999420572739539, 0.9999638464786429]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [0.9994053442614785, 0.9997354734361483, 0.9997747809067871]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [0.9995775402353695, 0.9998500008971314, 0.9998889810223056]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9958965961184091, 0.9966776582911174, 0.9936158985590393]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.9993766382160965, 0.9996459729318845, 0.9996960208897973]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.9993616072110794, 0.999644117395127, 0.9996857335297847]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [0.9983100707402071, 0.9990426740152742, 0.999217841006355]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9961538136790944, 0.9965523053247195, 0.9931214953392727]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.9993841557983904, 0.999699885912908, 0.9996905268221532]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9947098596437336, 0.9949630682491011, 0.9911031295829928]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9925311457486907, 0.9933734522469119, 0.9882683497829169]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [0.9991343759849055, 0.9996669491721161, 0.9997406442521544]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.999414571428437, 0.9996731907884435, 0.9997025962526082]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [0.9998477755274431, 0.9999367232105123, 0.9999648216379013]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.9987855607612763, 0.9993661735165584, 0.9993462126540367]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [0.9993351416523005, 0.9997873090160416, 0.9998237098539616]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.998436122665813, 0.9991507867426389, 0.999221040363129]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9961715434859487, 0.9971522107789775, 0.9944262153449377]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [0.9997303168173549, 0.9998730712894028, 0.9999041207847652]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.9988762556747356, 0.9994359941331439, 0.999381333975616]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [0.999900051852742, 0.9999685767548728, 0.999970373413878]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9953283662926342, 0.9961235757460944, 0.991769479744091]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.9993195830181953, 0.9997171330653262, 0.9996400088245461]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.9992275276406031, 0.9995569182200704, 0.9996005579573388]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9939711147015431, 0.9947874445739318, 0.9896354246832075]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [0.9995951940829158, 0.9998205471698074, 0.9998869202452799]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9975541148066697, 0.9978675009451758, 0.9956560538464195]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [0.9996837067021871, 0.9998941368726396, 0.9999178505128211]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [0.9998948237622545, 0.9999613118436503, 0.9999644542721329]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9963945310268081, 0.9964210566805178, 0.9934687612049158]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9979418663990111, 0.9984215344796172, 0.995603537482344]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.999149479188075, 0.9995319314311133, 0.9995522511728412]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [0.9996548467142541, 0.9998837013871226, 0.9999049939326026]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [0.9996077045950209, 0.9998821361223682, 0.9998729587028437]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [0.9993124837903632, 0.9997210943674546, 0.9997251244641558]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.9973346349111345, 0.997433622443223, 0.9943038051856791]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.9987387842829117, 0.9992441829383202, 0.9992353409959903]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [0.9997743063488641, 0.9998994355387104, 0.9999389952130773]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9961419361818346, 0.996886681152588, 0.994561552112681]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [0.9992712721222927, 0.9996205400807927, 0.9997238597376059]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.9989213437489609, 0.9995556163638485, 0.9994521561204459]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [0.9997009872677945, 0.9999203175173392, 0.9999225034495208]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [0.9993396818556066, 0.9997013162107405, 0.999799289064724]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9945103061769744, 0.9955996389835153, 0.9903701870196542]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.999046080365753, 0.9995392018207288, 0.9995133861596077]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9946163094288983, 0.9947424798748805, 0.9907025987176327]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9973207051558947, 0.9979171062752197, 0.9951941395427889]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [0.9998540444983577, 0.9999439560918418, 0.9999735307861245]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [0.9994002689233806, 0.9997169572848078, 0.9997532500474525]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9958437674803896, 0.9967236849874841, 0.9929173109703885]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [0.9991583116202044, 0.99969691348155, 0.9997111805398944]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9939634351566787, 0.994741738417729, 0.9900306358205978]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9966343860814744, 0.9968789590387672, 0.9943448834033606]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.998507341819188, 0.9991829022328961, 0.9990936045045884]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.995853908773836, 0.9963861537948517, 0.9927010346259245]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9967630508216857, 0.9970864715756805, 0.9937873274330042]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9969055961307145, 0.9974759093827891, 0.9927030753893745]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.9940135464733202, 0.9945853602773905, 0.9894997864322617]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.9992466525968962, 0.9996307850203336, 0.9996358910348798]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9968618866262149, 0.99736870056805, 0.9940060693178049]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [0.9993235509246218, 0.9996991132472244, 0.999748372668017]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.997033410034898, 0.9981109402783263, 0.9981348363621757]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.99948515118134, 0.9997335253725141, 0.9997571028540598]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9957848966492803, 0.9961987958474189, 0.9921413166421291]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [0.9992524769239205, 0.9997127950376137, 0.9997475199861072]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.9970050561401888, 0.9981734977936414, 0.9980342152458292]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [0.9995982003751652, 0.9998654339005714, 0.999877289851864]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [0.9995669925122829, 0.9998318510571197, 0.9998426098975122]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9941540372494889, 0.9946332774672819, 0.991086660779311]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.9992457639022047, 0.9996319233707842, 0.9996164945399619]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [0.99868883738236, 0.9993243092087128, 0.9993111211876754]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [0.9998335078969804, 0.999950107943479, 0.999955043918779]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [0.9996581549533664, 0.9999011215323035, 0.9998968754121077]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [0.9997464749416171, 0.9999008889451709, 0.9999295382820714]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.993944402268123, 0.9948082535856468, 0.9871536929809546]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9973599363704543, 0.9977844389502226, 0.9938941598839991]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [0.9995832079286737, 0.9998389343778841, 0.9998443561770559]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [0.9996740059139971, 0.9998698544981462, 0.9998981334311137]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9924936757438199, 0.9933072788775894, 0.9873499666693961]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [0.9983769091041981, 0.9993660440416398, 0.9993424631186508]
Epoch 0: RMSE = 0.8144526515084926
Epoch 100: RMSE = 0.5774812638851553
Epoch 200: RMSE = 0.3522387460121564
Epoch 300: RMSE = 0.3246403366839052
Epoch 400: RMSE = 0.30346319447864933
Epoch 500: RMSE = 0.3001410235171435
Epoch 600: RMSE = 0.2895663257446465
Epoch 700: RMSE = 0.281471341468364
Epoch 800: RMSE = 0.27405684551528114
Epoch 900: RMSE = 0.17266851602346697
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [1.370783724237309e-07, 0.06990395616020599, 0.8952918050871094]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.890213331936384, 0.029745939086568744, 3.6912823716085943e-06]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.8889276535241237, 0.03162001539320647, 4.748217940296634e-06]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [4.908477526304887e-07, 0.09961398601054589, 0.7399465080427482]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [9.958013912036853e-08, 0.0831969550444851, 0.9493272894250921]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9494604368022301, 0.06432168857365593, 7.135882390167331e-06]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.8590450880465662, 0.02737223488338038, 3.818999390668355e-06]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.002031462596864888, 0.8595158308776988, 0.24929804914262846]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [7.456070086923254e-08, 0.06001917455755059, 0.9320208734790935]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [5.3107944420903645e-08, 0.03638616922493558, 0.8297406581220066]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.8978129978733748, 0.02636040384205649, 2.6566395722768613e-06]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [5.9053866009611155e-08, 0.034542882724583994, 0.879210535493155]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.12139241237335392, 0.9084091235072036, 0.0057914026803972275]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [2.293038500692134e-07, 0.17575202603141413, 0.9706321999257684]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.792278322510484, 0.031472392822063905, 6.54308386808794e-06]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.8992240751097348, 0.03271308991955461, 4.572291205458422e-06]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.03131636683180484, 0.9445701638122108, 0.06510311249129709]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [9.549892580629756e-08, 0.07774351945504726, 0.9413798562399651]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.052352208558085006, 0.9570587183503628, 0.06412233780005679]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9416614202567631, 0.06858482447227322, 7.3249073050163635e-06]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.017653223385798576, 0.9363283234902905, 0.11692440673863345]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [8.813139939918831e-07, 0.33576012995127336, 0.974765360440432]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9123142059030671, 0.03935099786723374, 4.616520073132973e-06]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.011805058938348049, 0.8089808064171103, 0.014820095505550768]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [9.896836725810354e-08, 0.027908462454810164, 0.7558549099383667]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [2.2375894831637912e-07, 0.10058628511852337, 0.9232064829504705]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.00038237282889565623, 0.6969508327975732, 0.17753682561825307]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [8.245741942907357e-08, 0.04474047109823006, 0.8356303337466372]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [2.5803631994327266e-07, 0.12622038378733336, 0.8958917843450853]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9241238901442494, 0.036963517420600375, 4.460673432728762e-06]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9622260152722784, 0.07763896958702332, 7.18370259939077e-06]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [1.6064581136523385e-07, 0.09236210138565264, 0.9115478238175292]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.08993192956878018, 0.9654144641877186, 0.03231954962649436]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.919618546397698, 0.03512969619496021, 4.229939691689257e-06]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [2.7017924205477372e-05, 0.26332003606483767, 0.197811916107454]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9435890902945825, 0.04969627845046079, 5.46757894919963e-06]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9368256379380782, 0.04893086522165653, 4.941263883305505e-06]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.2524008268735697, 0.9794170212650369, 0.021285248107313397]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [1.6380482841381627e-07, 0.04227530792896731, 0.5668605722231398]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.3969606400305212, 0.982764853169569, 0.01524150712525826]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [0.000449471835492433, 0.7777318757503989, 0.28230044541563265]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.19882879024756256, 0.9689248904132699, 0.016051150077978946]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.5968146856394648, 0.9926260123836855, 0.03702504786185822]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.23434453740911887, 0.97462173261574, 0.018721881422340905]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.8925473849645629, 0.027940682123892696, 3.5508930193758674e-06]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [1.148516574863294e-07, 0.11362276467641547, 0.9505441616506736]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9038146509763034, 0.027598659899229622, 3.840831880523526e-06]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [2.7771675902474186e-07, 0.12011852108326453, 0.9536467942937444]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [4.151489146866725e-07, 0.1519011801914083, 0.9484035707059849]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.06645324485214478, 0.9643834229059357, 0.04647859387711963]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.8842049944950904, 0.032751969704582426, 4.242336542372023e-06]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [1.481476454017661e-07, 0.11573621580343613, 0.9511949402251791]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.012878053538166416, 0.9069007445384482, 0.05152999317433636]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9202157100966244, 0.04618272692487668, 5.918498130511929e-06]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.2897153079007343, 0.9856977820617753, 0.048741217648403376]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.0013265587601886597, 0.8084684141604653, 0.16424664498013047]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.001763465746481948, 0.755538840212378, 0.07103455863953938]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.8956011937024593, 0.04100985258783238, 6.226526788801902e-06]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [2.907502053090996e-07, 0.1314760115569497, 0.9452404015524583]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.10630902261893324, 0.9732551521814757, 0.05064970529152515]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9279870021874124, 0.041271029352823445, 5.072195904683254e-06]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.32202403654072276, 0.9896221195254283, 0.04007215363880344]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9514651060282031, 0.05807034340653729, 5.327678904081121e-06]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.8974354633138578, 0.03311165894062602, 4.5582061626055e-06]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [2.1876878636329781e-07, 0.13692547987934953, 0.9768516561722014]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.923713571575623, 0.03904108130772181, 4.786806220987826e-06]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [1.877089505639186e-05, 0.38734521693015983, 0.5393494931715942]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.933781500457625, 0.03797487971723784, 5.186164781855247e-06]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [3.547912147955795e-08, 0.02791429238244427, 0.8562828381545894]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9398129218553228, 0.04668001707033264, 5.518942801944269e-06]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9529237267078313, 0.06331602033715701, 7.909398258982979e-06]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.007787012639022069, 0.9189372400815335, 0.1393540752597878]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.8977127344529848, 0.032872334910811214, 4.579121988353901e-06]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [1.383871596615052e-07, 0.052403380719188514, 0.8197851887149368]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [2.1021019444411824e-07, 0.09134497981203529, 0.883142560890106]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9348865626634646, 0.047540102623002956, 4.99115893767789e-06]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.07236432961916463, 0.9425856019298362, 0.01989954431731229]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [2.2910555785949767e-05, 0.29833213710533857, 0.24583494018638877]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [9.632253651957425e-08, 0.05720423828993763, 0.9274405745552672]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [1.2938692666399314e-06, 0.12172603929901032, 0.6808827415838575]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [0.001447206854791663, 0.8911120808513914, 0.5201711402532628]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [2.1288582692625742e-07, 0.14558413332681658, 0.9764754199801189]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.00019650086979778944, 0.6280541314693898, 0.18558664456570048]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [7.540228055325791e-07, 0.12137473164411577, 0.760013126192848]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [1.6019579201993754e-07, 0.11636186701974797, 0.9281626383777537]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [0.0007083897238696375, 0.9219297215974968, 0.8254582495745358]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [1.9504288484259946e-07, 0.14270514698355397, 0.9836041229913465]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9105783454567113, 0.057633740162010935, 6.613095204674944e-06]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.9512027795462171, 0.04994407216944758, 5.64947663568079e-06]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9328715606494942, 0.05042495925861849, 6.957738747542447e-06]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.052482133967707324, 0.9540972883744597, 0.03331028001304798]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.02975999093363004, 0.8743584242374475, 0.010516088327512923]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [4.498830958638262e-08, 0.042769544616902716, 0.8980849506959138]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9008442560529021, 0.03337403447917932, 3.771775481235638e-06]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [4.9322742990048936e-08, 0.02442048492986806, 0.8179608971315971]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.21817158826277977, 0.9611829633652597, 0.011243423640223377]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9229031062379517, 0.03795938577136385, 4.218477612428959e-06]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.0014272811636371779, 0.8209244673166718, 0.17305576171244444]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.0004936374115805428, 0.7314860944326723, 0.22960566962373852]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9210526403076312, 0.03905901152671734, 4.942146391696519e-06]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.8657194725608447, 0.024772138100190834, 3.2491430670922736e-06]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9259374996642515, 0.05248247127757149, 6.658958199494783e-06]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9199813804377975, 0.05548383346927013, 7.87939577077201e-06]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [3.1304048776617347e-06, 0.27292449255432094, 0.7725917201563026]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9437118694200897, 0.05187451630324461, 4.5191965111264066e-06]
Epoch 1000: RMSE = 0.1743396489825419
Epoch 1100: RMSE = 0.16394709899071697
Epoch 1200: RMSE = 0.16636669931295825
Epoch 1300: RMSE = 0.15124961546948731
Epoch 1400: RMSE = 0.16047310580848564
Epoch 1500: RMSE = 0.1818941265673385
Epoch 1600: RMSE = 0.11918280765674061
Epoch 1700: RMSE = 0.14896060623386484
Epoch 1800: RMSE = 0.1352831749442992
Epoch 1900: RMSE = 0.14266195581811544
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [1.6174963062835906e-09, 0.02911017155334674, 0.9660369635198724]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [2.5533959073730655e-09, 0.02231174852453429, 0.8588216572440779]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [2.6370905248017192e-09, 0.020845147391169227, 0.8753659614927991]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9843674680478458, 0.020157852746489832, 1.260021439072111e-05]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.06601262457585055, 0.9966214632462158, 0.03983418248231189]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [1.3236098043358652e-09, 0.026442675322731594, 0.948557391851089]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9804884497123922, 0.011863541961959564, 1.178060309674482e-05]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.04022870312296055, 0.9947854864612322, 0.028548981743155233]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [3.0189405594577445e-06, 0.3772159617457971, 0.18375546780615046]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9746865620752638, 0.012168795607834593, 9.279007408978516e-06]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [1.3307488608519604e-06, 0.6690332468506932, 0.9279298592136865]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9750718928090504, 0.014279392872960074, 1.2295830623090823e-05]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9869528322232949, 0.026007754793943846, 1.2320964855394463e-05]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9822769128687328, 0.02084576839355535, 1.0584405544274404e-05]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9768504204622879, 0.012682570927282748, 1.0760564074326645e-05]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9786174415491603, 0.020883576710913697, 1.2534076470943238e-05]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [5.963723527824297e-10, 0.009432000004094193, 0.9011849973816985]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.984648978741431, 0.027321398488594374, 1.344547466658394e-05]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9812041519574124, 0.016593792975081803, 1.161545574805861e-05]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.047561802027649105, 0.9952573817838174, 0.031021888933922066]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9741080704855761, 0.02017616955792883, 1.0554195018782128e-05]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [1.2304804352799142e-09, 0.011306663024910584, 0.8814986554235613]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.02909556209990007, 0.9916328752068215, 0.017301421371944327]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9849109568655979, 0.016665194810701528, 1.3882188597509258e-05]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [3.335074019435884e-09, 0.047508251595598947, 0.9849413496288676]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [3.272475337257331e-10, 0.005093236939492895, 0.7886801310825051]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [6.28129587372718e-09, 0.03947832300135105, 0.8177651280972305]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.01925557126339615, 0.9883620565772647, 0.015605320593053314]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.05319360635971451, 0.9967378535551483, 0.05352894426807206]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.000788950139716417, 0.9393985548203995, 0.04342746630815149]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9746565956999921, 0.012185401729636577, 1.1129893635691944e-05]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.008168694260500927, 0.9892791567059098, 0.05349222575146977]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [1.9509625279055906e-09, 0.027115606216953296, 0.9561049346667119]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9812697141439829, 0.027412958598366805, 1.1997029570156721e-05]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [2.190752432893459e-06, 0.3884725649451371, 0.22900152640422744]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9582512777397838, 0.009532463268660698, 1.4064593262381715e-05]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.061424827266759414, 0.996907571729987, 0.037884495300525574]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9810005424937135, 0.01649496292318153, 1.0271161479989591e-05]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.9766474470915515, 0.011739137439736005, 9.732140442636652e-06]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9804240056360715, 0.01610109553057691, 1.1227144000749619e-05]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [8.719759916852648e-10, 0.01136193265314302, 0.9186437664629494]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9862838966942591, 0.026919971619557028, 1.5064154958262105e-05]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.016206734955913408, 0.980317039142216, 0.007205964225881765]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.03240274012512831, 0.9909450647565345, 0.012937809192115571]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.985103693384831, 0.022284744165572946, 1.3003345013068517e-05]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [7.722546566945304e-10, 0.01489078115756829, 0.9289044156113273]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.980687999276777, 0.015453552295885591, 1.0946395583101584e-05]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.008220323154004438, 0.9893962830608957, 0.05572341715396728]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [4.065891785296221e-08, 0.06167633688790508, 0.6519129606917488]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [3.3771186940098273e-09, 0.052642629272217856, 0.9840583030539006]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [3.4242089451049137e-09, 0.06494756234927297, 0.9802226653072166]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.02459002336695332, 0.9941561919206625, 0.03625991193494157]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9685764966623485, 0.010343467931609179, 9.012456833150885e-06]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9723303997866136, 0.012543863790062984, 9.299631347212127e-06]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.00598880708571817, 0.9874369243951384, 0.04197034872874922]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [3.437848733249357e-07, 0.2587312039869918, 0.6149715042723014]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [1.2689127004361977e-09, 0.017642094793825024, 0.9529182361986493]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [5.90675959118452e-09, 0.0999583469383573, 0.9865130516301721]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [2.4657828329184633e-09, 0.02718086843602087, 0.9256694587990119]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.051794016744301445, 0.9969363160687724, 0.033416499981523094]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [2.0957016380464943e-09, 0.03230618723162066, 0.9703095075047147]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.988330621268482, 0.022830943358476113, 1.4784542283848542e-05]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9766577652225336, 0.01276591352749752, 1.0526891084527911e-05]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [3.4813605012713843e-09, 0.05424221460558221, 0.9885719164707656]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.975501142403066, 0.014019626565033382, 8.94476451753161e-06]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.020114821656294656, 0.9928752917033905, 0.03742219886314785]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [1.8599675182725558e-09, 0.03672700871657753, 0.9691738566630694]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9883866050477234, 0.03528266908980393, 1.428849102035731e-05]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [2.346908600661359e-09, 0.03158794251072441, 0.9737770039020233]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9766784959092285, 0.012762237165551625, 1.0536614964238907e-05]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9767931091741613, 0.01593808115331191, 9.603753558329403e-06]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9846033124880951, 0.023118343729031692, 1.0061254768433059e-05]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.013451633986046319, 0.9892345890792554, 0.023832019269323452]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [1.7034702652383283e-09, 0.03874422885396567, 0.9672459245233171]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.05943024231827801, 0.996346511035735, 0.0389357732931596]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.03337465647607753, 0.9939356409683486, 0.017850298239518586]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [6.128525495932618e-10, 0.007157894562392958, 0.8802110928753817]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [1.4817266398101006e-09, 0.026899571167808344, 0.9601995691408078]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9782428921604358, 0.018317683785250165, 1.1551440376125294e-05]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [1.935087977533708e-09, 0.02536765008268082, 0.9298750019026877]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.037239328876739745, 0.9949730994623447, 0.025699498902539506]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.03119640190709413, 0.9937909453529942, 0.0204723745742865]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9715961059830215, 0.009932606681295475, 8.83490853508361e-06]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9772758614712267, 0.020415406257514694, 1.3368532615958405e-05]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [1.20958404980562e-09, 0.020867470130079915, 0.9534443387237378]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.06113695201057799, 0.9963843128285519, 0.0372109141584482]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [9.588322883755352e-08, 0.1894496621654478, 0.8075056795059337]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [1.0124767923657331e-05, 0.6740336272888999, 0.19628307969933453]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.030879540229643496, 0.9932679422466673, 0.01914611916660099]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [1.522050854662878e-09, 0.02594403555122523, 0.9477101163608453]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9818071005979657, 0.01737053583394767, 1.1546236186345129e-05]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.981383260514401, 0.016026543842244324, 1.1015046945660777e-05]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.03290720373758814, 0.9924241309328892, 0.01627820597040632]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [6.481340680094229e-10, 0.010900328205782286, 0.8888174731555482]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [7.140545947183425e-10, 0.011408167384877722, 0.9026701181740501]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [2.4708688077967834e-09, 0.0348580873771564, 0.9735193626637466]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9822264570223451, 0.020782993290278395, 1.0806693697293538e-05]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [1.5336241396721912e-09, 0.032193013265533116, 0.9575103305556476]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9796942081316702, 0.012181465240725782, 9.023754716338817e-06]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.028298041625066935, 0.9874762647608746, 0.010973759979533643]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.01657273313614784, 0.9807717621949513, 0.006686730247994344]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [1.8913754546913454e-07, 0.25766358991721483, 0.8634536743741497]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [8.949260626846167e-10, 0.012185363541410475, 0.9099604586404842]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9820849638853697, 0.019615743766603244, 1.3293356353918761e-05]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.023680855642316773, 0.975002365662174, 0.005716652373622888]
Epoch 2000: RMSE = 0.14677194070880994
Epoch 2100: RMSE = 0.14254368203516873
Epoch 2200: RMSE = 0.1288173398303799
Epoch 2300: RMSE = 0.1359021568065657
Epoch 2400: RMSE = 0.14051795216171123
Epoch 2500: RMSE = 0.1400356979105008
Epoch 2600: RMSE = 0.12515543960343475
Epoch 2700: RMSE = 0.1246380204842072
Epoch 2800: RMSE = 0.1362987887561296
Epoch 2900: RMSE = 0.14091797966385042
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.015955934277112192, 0.9930273686216167, 0.01159615286487276]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9905109595566567, 0.010509561270598947, 3.4212151457162245e-05]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [1.5720447539424033e-08, 0.07137300747714911, 0.7424303639941364]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [6.243374663982642e-10, 0.013822173766885168, 0.9461894899605592]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9905286232338304, 0.010508993679648106, 3.4351255199219426e-05]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [8.012134591095194e-10, 0.023235869376519197, 0.9602989085807687]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [1.438156979247358e-09, 0.02735188542027557, 0.9436730856071417]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [7.320599862642048e-10, 0.02117977128193245, 0.9544890443916366]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [0.0005987132830336926, 0.9566070135196225, 0.03238508452430227]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [1.0066199910677154e-09, 0.025053911703693456, 0.9663325564620927]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9863050083576457, 0.008444592638507498, 2.7400364628283552e-05]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9899063149011714, 0.008223310840408195, 3.302555709915833e-05]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [2.8237673366910335e-10, 0.005424064565197744, 0.8651516171622475]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.016526684275278807, 0.994092727271024, 0.014282625119613326]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.016435145810273246, 0.9949184167989025, 0.0165815952833145]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [8.861821167156027e-10, 0.018145803540583354, 0.9222956214481344]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [6.1465590907515165e-09, 0.04257788151428329, 0.784527757218374]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.03137788518163644, 0.9972011037266755, 0.03433879538221176]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [1.6653575918376615e-09, 0.041331892190150404, 0.9820461851685037]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9964246498583053, 0.01436873631527155, 7.957302281312073e-05]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [1.1832307614199407e-09, 0.026939530979624277, 0.9700356489991052]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9825636983445292, 0.016388283529839533, 2.0310287285190893e-05]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9935456430564417, 0.014451780214896603, 4.3715925776704806e-05]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.008215295925333948, 0.9942323963752475, 0.040749480690112916]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.9983383741546027, 0.020374828237772345, 0.00014734257283767772]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [3.7187270604148925e-09, 0.04299618988298027, 0.9333593335549655]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.00011268464927375287, 0.9315225904378579, 0.08004772489025613]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [1.5427533849308667e-09, 0.04872107965629556, 0.9783023027889869]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.037615581598662844, 0.9979092570159663, 0.04345310107777761]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.02906178438884833, 0.9977439242279502, 0.02912785147209533]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [5.970816382731589e-10, 0.016404642963000152, 0.9469861422688847]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.0140272282206725, 0.9953226411396534, 0.030207697269035307]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [2.3119053391045872e-06, 0.7953342597472022, 0.8861939241109213]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.0013482408334131797, 0.9743860622531704, 0.026680323670994984]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [3.804506327110353e-10, 0.011547462105705661, 0.9190099405489599]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9880343328949313, 0.016954638020387824, 2.9161231212103118e-05]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9949906061269486, 0.016980573997758625, 5.259728474114297e-05]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9952208861005185, 0.010102901700849652, 6.445816238536512e-05]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9905637360842536, 0.010455378572950749, 3.459590928322941e-05]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9972369316933871, 0.022671121619005412, 8.020882273947129e-05]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.00318868223931082, 0.9865711548953894, 0.025863165867958227]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [3.0292836661738487e-10, 0.008203296894570345, 0.8755923641775094]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.008026056881451384, 0.9849088299800473, 0.0058712349682964035]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.031519937282700534, 0.9975739184302571, 0.03383115080127429]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [8.444704138441442e-10, 0.030295210597418402, 0.9629048887301548]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.011347285779986794, 0.9808634718500765, 0.005279369271498151]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.015281876882436334, 0.9950373641032336, 0.01826680639568535]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9932831786469313, 0.017373070164884902, 3.7174202382628106e-05]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [7.420954421128354e-10, 0.024756077516725313, 0.9519942773352928]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [4.810071794179718e-10, 0.007896373536834459, 0.8705142474380162]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9874790980429535, 0.010181192228545757, 2.726486810598808e-05]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [1.654145331891699e-09, 0.03770152893171598, 0.9824983207771518]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9943529745736385, 0.013401801489695718, 4.901984533697694e-05]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [6.307972909790807e-10, 0.015692984161454322, 0.9283292673250645]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [3.867622954980721e-10, 0.008315817788346022, 0.908998922824564]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.026031410773164763, 0.9964740268760448, 0.026523456230759287]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9937582133709727, 0.013914273751428983, 4.647371784693319e-05]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [2.926994015129838e-10, 0.00741881802510267, 0.8848563671307649]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [1.5867134940263996e-10, 0.00398590734724836, 0.7590421195591991]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [1.119448852576408e-09, 0.024157867913883697, 0.9704695924039778]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.007893499068773895, 0.9845893742994205, 0.00625193606796661]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [2.536952971967713e-09, 0.07337983206736406, 0.9853064434991274]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9869873183920249, 0.011633653977466886, 3.0590615358399324e-05]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9964023544002559, 0.01903240032624411, 7.415325663955427e-05]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.023147010637049444, 0.9962361135178002, 0.024054799316772655]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [6.500968202597586e-10, 0.020775619600058547, 0.940361343670911]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9971926346244449, 0.01074272448782624, 9.176869467929834e-05]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9944191123056945, 0.01383605584753271, 4.745123018799587e-05]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.011114641396709538, 0.9951565109125494, 0.03584352415634818]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.031162031514432614, 0.9971586155013016, 0.03319798959491385]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9879267920020322, 0.022081560845819106, 2.433701657172534e-05]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.985749117033615, 0.016384374021629698, 2.7957185438758706e-05]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [1.7982789612734183e-09, 0.028219273880584796, 0.904055237377015]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [4.2532477873428284e-10, 0.009291370510062033, 0.8984059749140457]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9910208042357074, 0.01156896681658315, 3.279276786503562e-05]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9926649132704252, 0.013266741381358341, 4.006272884361678e-05]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [1.7281884195526365e-09, 0.042537234984444325, 0.9870295948003971]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.03433188440829196, 0.9973970450178338, 0.03501040080318765]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [4.642880928336677e-10, 0.0060629191855289125, 0.8257982007910839]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [3.4036529576899193e-10, 0.008681251954417482, 0.8892238855959047]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9911760038021998, 0.015942903713655098, 3.544303085948811e-05]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.011836083068277103, 0.9919499446355882, 0.012694000078018335]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.993816019970678, 0.017270301845845396, 4.179597119832299e-05]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [1.4099119969553136e-06, 0.4246211231881891, 0.19028667331267313]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.010947922240987775, 0.9949942390737725, 0.033268729785905914]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9945673331729667, 0.01305590474003771, 5.254437452935295e-05]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.989149399927023, 0.010044256277844358, 3.408058027993259e-05]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.016039245240891103, 0.993907311178831, 0.014611855759044416]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.0018991466493264975, 0.9864726766620731, 0.04213068564724976]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.020807841099351347, 0.996312484362821, 0.02203784185669616]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9893914402073795, 0.01305858643428496, 2.776815175712881e-05]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9960538893921452, 0.019811246233341965, 5.394035116852108e-05]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9938911158128055, 0.022167647329193473, 4.481204464409509e-05]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.9927664052950588, 0.009811323969465177, 4.19788403315003e-05]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [6.611734635262546e-10, 0.01899055955056533, 0.9419567418580809]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [9.131669200165774e-10, 0.02885167891401133, 0.9646944402924613]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.013851470903143584, 0.99027942420708, 0.00974325563636184]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [4.062294996853414e-06, 0.7130096015462365, 0.3259615735605908]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9888655661677319, 0.014953647447812581, 2.957849891710965e-05]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.017658422618538856, 0.9954415092223242, 0.015474933533738758]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9904966510179654, 0.010085026272651192, 3.233214540724778e-05]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9926442529976824, 0.022638464352263316, 3.658851148677728e-05]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9967409394207949, 0.030242328438412274, 6.96977416076136e-05]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [8.590071836402159e-07, 0.47807740042792585, 0.438794558712592]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9775141666392702, 0.007764266322854081, 3.413979623635553e-05]
Epoch 3000: RMSE = 0.1327963758848428
Epoch 3100: RMSE = 0.13215108757521069
Epoch 3200: RMSE = 0.12344995947121043
Epoch 3300: RMSE = 0.13858308724682922
Epoch 3400: RMSE = 0.12841230756938213
Epoch 3500: RMSE = 0.11621216198184427
Epoch 3600: RMSE = 0.12983902974661415
Epoch 3700: RMSE = 0.136503168302852
Epoch 3800: RMSE = 0.11759826853145063
Epoch 3900: RMSE = 0.11901190589123896
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9951741079378863, 0.010722060312635121, 5.62484975745192e-05]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.023008827979777945, 0.9980799955158399, 0.02441365569243172]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [2.032375603036739e-10, 0.0037104379799065656, 0.8731496378280249]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.01920689520897704, 0.997610643931765, 0.01881174650247374]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.027707146391152203, 0.9985681499286586, 0.031018092271321112]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [6.14911229640292e-10, 0.021233058012706023, 0.9651113387427588]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.9964018135547984, 0.008959209027458073, 7.861906586983542e-05]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9965795968379771, 0.020756420192291566, 7.394838326947623e-05]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [4.795108888830946e-10, 0.010596020284285294, 0.9324277920582661]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9947439273615132, 0.009243545265810156, 5.4206622005621924e-05]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [7.280486623280402e-10, 0.017274760573107277, 0.968662550265827]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9935355839718493, 0.012124065706978179, 4.185454806529118e-05]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [5.406995688523826e-10, 0.017331123706195335, 0.9548125633079251]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [1.1116941318743405e-09, 0.033787105024790105, 0.9797449179288289]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [5.326312884384383e-10, 0.014634529020016595, 0.9575515960995954]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.012063669384035991, 0.9965255587713705, 0.011718103772115143]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9970413658305942, 0.016130416215192355, 8.159999952813288e-05]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.011657342192489295, 0.9958152497708113, 0.010370887478533394]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [6.636570750240574e-10, 0.0199759350112003, 0.9672637310506547]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.015418386106530348, 0.997490973109198, 0.015658718220765602]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [1.2125193388758523e-09, 0.02875017314398052, 0.9832051710474652]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9960163243738499, 0.012316692709771584, 6.793397804006514e-05]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [1.0430461191881364e-08, 0.07341974730516643, 0.8373932264075383]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9986483952025631, 0.017643708105528247, 0.00018683889792295795]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [2.7670930749346624e-10, 0.008038284091843519, 0.9238958150738391]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.009981250274409798, 0.9933010498104347, 0.006924844748433496]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9830385112931235, 0.007138873994505779, 4.047794095995774e-05]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [1.1511141639751124e-10, 0.002766737596197512, 0.7716176119486515]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9985340445915869, 0.01835887771494048, 0.00013841593752846558]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [5.2822080601585665e-05, 0.9010471221868818, 0.05723270109914765]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9920807044477492, 0.009443963557738744, 3.945919024403642e-05]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [5.815370296647864e-10, 0.01604109605363032, 0.9632576933237629]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9933448832882954, 0.009238851018245148, 5.106009928803604e-05]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [1.2047315042167365e-09, 0.02646002891977219, 0.9838360153816171]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9892350619992595, 0.015454835135134464, 3.3325847669401635e-05]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.008776176095518628, 0.9963723372212243, 0.02287483491870096]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [2.8086399783248093e-10, 0.0057809292288448335, 0.9152671543303876]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9944056562040483, 0.00966493927659595, 5.3468663068935116e-05]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9943914786326348, 0.007500164410620629, 5.493938518738352e-05]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [8.1481782791731e-10, 0.01688262176851265, 0.9725279412177591]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9985602139118775, 0.013134455766010108, 0.00018637079932560274]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [8.598117636860597e-10, 0.018635001898688382, 0.9721503573564881]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.997909917520639, 0.009203490377191013, 0.00013816142968774375]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9973767724865487, 0.012387588377634814, 9.887008893660615e-05]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [2.474727375175205e-10, 0.006039232175297113, 0.8971099507927446]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [1.1719759115266318e-07, 0.3937191856682607, 0.9425884160513571]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.00809962815452866, 0.9867726160003215, 0.003759629202772447]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [3.9619464413316095e-06, 0.6985567104840055, 0.15066923008199037]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.023000422552047792, 0.998070448134397, 0.023646722511772043]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9863520123285345, 0.015398204601584517, 2.3290183840708402e-05]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [8.637482856541975e-10, 0.011636870238019311, 0.8653079000962548]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [1.3739424664526413e-06, 0.6053034313940225, 0.34985574588992946]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [4.5416479499422904e-10, 0.009557832613747616, 0.9499357503414944]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.999538302728229, 0.018598872424180762, 0.0005116687973151954]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.009491315331944361, 0.9948371284852561, 0.008721953293375353]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.007254852561979121, 0.9945852598249127, 0.010733487598659245]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [1.107001427557084e-09, 0.020176267866303263, 0.9459610573616694]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.991816461311472, 0.016003228633820695, 3.8659138142545455e-05]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9927741183853084, 0.013945477882193136, 4.180762327019803e-05]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.012891487840753847, 0.9968766363250444, 0.011048673341560419]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9988227064336015, 0.028313521570218755, 0.00018435699405992148]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [1.2588801608059421e-09, 0.030019998886300244, 0.9878803001388118]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9965860613026095, 0.016285988009024, 6.77229846692246e-05]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [4.344543827179963e-10, 0.01135948557301183, 0.9504057730717616]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [1.9522693743496538e-09, 0.05401747554558697, 0.9860625029992682]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.0132139468752029, 0.9969562242384403, 0.012292171013355249]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [0.0009537220631409489, 0.9791843577381758, 0.019369548365135002]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9943930205401662, 0.014972735993432433, 5.096903919131728e-05]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [3.097845550712538e-10, 0.006492899627100703, 0.9044360369394739]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [5.032367979178633e-10, 0.015135846700494476, 0.943178032345935]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [4.911113967878171e-10, 0.01335214516828945, 0.9458997866538498]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9911462135459608, 0.021010077582725262, 2.9954734357547545e-05]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9974843571104451, 0.012822695934211259, 9.842750007292935e-05]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.012039550384702407, 0.9959588816156244, 0.010099167447123575]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.013585087869411, 0.9974832825131357, 0.020615305247931204]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.023773311726306208, 0.9983809321160361, 0.023897111439718302]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.990959834594562, 0.010862329736073305, 3.974991744440855e-05]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [4.162139136852183e-07, 0.40078576184363823, 0.4465048199199858]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9944251502360966, 0.009651070479966986, 5.3843047911358596e-05]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.011520096947211015, 0.9952003646892219, 0.008335280389092693]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.999108690751896, 0.021044385705652986, 0.0002386315753635155]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9957829711903806, 0.021173762903339003, 5.946756430971583e-05]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [2.1273399673414557e-10, 0.005132522207087102, 0.893393417266345]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [3.2946935432311823e-10, 0.005337587230607728, 0.8801897119862566]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [2.2006753147284482e-10, 0.005681935304680501, 0.8831650749652099]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [2.782831880699592e-10, 0.003813222398678163, 0.8416087020878804]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9976733997488889, 0.01577429072727014, 0.00010541331402608298]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.005826706497394887, 0.9897118927949233, 0.004092211103661004]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.994381752988263, 0.009650337339709904, 5.304349250365214e-05]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [4.622246244381323e-10, 0.010961199778729173, 0.9326987933189606]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9969421923949157, 0.01287869817043901, 8.825123282982059e-05]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.0009755856923384405, 0.9871838900072546, 0.03536903563517812]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.025186724532739704, 0.9982259033632109, 0.024882830208223242]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [6.360966052613554e-10, 0.0097669067415326, 0.8698322460892339]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.006463504065146058, 0.9948809462958649, 0.013850108722790105]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9991290938887487, 0.009661779163873415, 0.0002834727894497558]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.021400240480502788, 0.9984559781346981, 0.02070519543675211]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9966739273430067, 0.013336944926933573, 7.847682849527043e-05]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.00256336831573582, 0.9928564795926172, 0.03850077158996033]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9912862695196242, 0.007734910656006923, 3.9257448798510305e-05]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.014926816048711045, 0.997683735818618, 0.021578725812792358]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9975957713840174, 0.011964352979028264, 0.00011155796683849212]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.0173420835862862, 0.9974698930902691, 0.01692293786112725]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [8.686975691861916e-10, 0.01553236702428197, 0.9537683993510561]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.005753345379513504, 0.9895111918584331, 0.004354093304372254]
Epoch 4000: RMSE = 0.12296081340279962
Epoch 4100: RMSE = 0.12888871165869328
Epoch 4200: RMSE = 0.12712210715178662
Epoch 4300: RMSE = 0.12403182603129287
Epoch 4400: RMSE = 0.12047894584025969
Epoch 4500: RMSE = 0.12463071966241049
Epoch 4600: RMSE = 0.11525715988754873
Epoch 4700: RMSE = 0.1274930907816941
Epoch 4800: RMSE = 0.12003367726266592
Epoch 4900: RMSE = 0.11556677115186023
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9996718974224296, 0.008476905125142594, 0.0007219318278576993]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [4.318841805975528e-10, 0.010963162302234937, 0.9581758622613351]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9983195145785118, 0.012794162737096186, 0.00014270657785035905]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [9.833877746422106e-10, 0.02161746222890614, 0.9834596053969581]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [4.3774082839419025e-10, 0.012977230656227705, 0.9555133288275851]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [2.2147572451499565e-10, 0.0035714081654471866, 0.8858408712871831]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.01041059373822978, 0.998078249639918, 0.00781172564460567]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.009272840096615165, 0.9970759302846199, 0.005877258650746682]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.01044755545487099, 0.9980976089860653, 0.008748817052457426]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [8.980820545643078e-10, 0.025380227767904297, 0.9799075815898576]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9985039947596184, 0.012258916877883932, 0.0001660844255155774]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [3.4264291197266023e-10, 0.00733130356543591, 0.9352658187608412]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.0019742081334820786, 0.9929303317567839, 0.010878161250461546]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [3.3270401526602593e-10, 0.005530258513381535, 0.8894203014215278]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9923317180231465, 0.01572570490675847, 4.1176252502619704e-05]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [6.247426325090309e-06, 0.8084904091777076, 0.1046954282348344]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9947318404808692, 0.01603414805711518, 5.3465405127778225e-05]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9965932758116245, 0.014815272601282212, 7.517640383877951e-05]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9994309958794194, 0.017071672302154606, 0.00033515161247791203]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9945278759394204, 0.007548959468962931, 5.6237588803735603e-05]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9996849059190904, 0.01926268580805878, 0.0006397860949313109]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9988426287288831, 0.0120431779741068, 0.00019818306670267295]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9874300145599464, 0.007130277424645259, 4.854663811364415e-05]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [9.77070882687088e-10, 0.019928645791670804, 0.9838531144261179]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [3.5224931773426195e-10, 0.008495310938705354, 0.9506103710427682]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9967248898765517, 0.00936746025052417, 8.236526835055085e-05]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.004644438608686651, 0.9935857064149629, 0.0030732671590931]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9990232944881962, 0.00848227040764967, 0.00027446048316822784]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.9998462798212165, 0.016419620111296242, 0.0014817949321312839]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9978916325193623, 0.011866521886658423, 0.00011706978189899289]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.015603747670202972, 0.9985397322071459, 0.013320529650017168]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [2.5059330039980615e-10, 0.004844411295078686, 0.9048480346337319]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [3.7226113655452914e-07, 0.4406006424166657, 0.39547134458276484]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [3.9483439316366225e-10, 0.00431290079763473, 0.8127678042696074]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.004701098250826299, 0.993759277027741, 0.0028643208156624573]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.009763019751415495, 0.9975395476351593, 0.007040592320210852]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.017600697759186285, 0.9990723971160915, 0.014474761155207861]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9968054883111308, 0.00717958358085714, 8.696962693226386e-05]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9987432797961014, 0.01173576897597108, 0.00018894662208230025]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [0.0010144869784859433, 0.9883254897814185, 0.012909783370347708]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [2.0063659029311923e-10, 0.004577840049356784, 0.8953434510986109]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.012314427128555622, 0.9985640601254204, 0.013982333115963242]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.00872887116748589, 0.9977521417160543, 0.008279912503572535]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.014190293064805919, 0.9984744937713887, 0.011826722177703714]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [5.576890506982792e-09, 0.03772947692474032, 0.7483015525832262]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [3.9109487054593833e-07, 0.4331628934716922, 0.397553572785238]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [3.6550411968204917e-10, 0.00815462405219373, 0.9336596878664475]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.008037862873415699, 0.9959134074372701, 0.004907158404375382]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9981939276332527, 0.020400098606243654, 0.00012715417534377727]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9970938615258226, 0.008880795972065772, 8.924966443050142e-05]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [4.986397350897324e-10, 0.01600817672865755, 0.9653561720947795]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [5.897866164643832e-10, 0.013003907147881806, 0.9688765190269533]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.996716772327728, 0.009377553931461235, 8.214122622210225e-05]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.00979658174163446, 0.997879220499614, 0.008272429493888867]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9939381153757176, 0.010816089408020884, 5.274185292750832e-05]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [9.302742486573342e-11, 0.0020748695397314115, 0.7721562531722749]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [6.025570872834238e-10, 0.010682084258149488, 0.9562526960059629]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [6.97087057328887e-10, 0.014044139069279678, 0.9722407127694572]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9939946340165117, 0.021428559844268804, 3.904272109163051e-05]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [5.377890108218687e-10, 0.01505227466203479, 0.9676059618133653]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [1.644717325172324e-10, 0.0027824046764432634, 0.8745433837164612]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [4.712841376137254e-10, 0.012076433059552518, 0.9634606059003528]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [1.7811101882123965e-10, 0.004282767478627112, 0.8836437955719895]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9986124963523155, 0.015448358922434293, 0.00016078524510601438]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.0010384527857227432, 0.992671063390835, 0.024390201690452397]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [3.7458147603003174e-10, 0.009644106906888585, 0.9472944861602411]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [3.6820983938715277e-10, 0.007162033791333978, 0.9504538690321505]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [7.735751350654882e-09, 0.09584963925387485, 0.9720716102526155]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.020489375142660624, 0.9989196286366858, 0.017733939053350436]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.018659627041162422, 0.9988245754518345, 0.017403118382299663]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9989007414053078, 0.011216839847208228, 0.00022715287681187278]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9967250595634173, 0.009373787262906929, 8.257291822096521e-05]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.9981703766019153, 0.008443815515201873, 0.00014258714387158393]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [1.6056409176878869e-09, 0.021542616113920655, 0.8945053705106497]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9899425928899643, 0.015852902527775676, 2.7818117483462686e-05]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.004218429683924251, 0.9967311681806571, 0.01950110519541924]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [3.3554079761165287e-06, 0.6772463741730144, 0.0991023746482284]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [1.7240606053792938e-10, 0.003889400399804376, 0.8932105723673907]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [3.8576852483162935e-10, 0.011021916872988072, 0.9446648823166903]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9983200421090963, 0.015729586196163336, 0.0001264950217305087]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.012640827124051443, 0.9984838364649877, 0.011089179932148054]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [5.164663097506151e-10, 0.01080832204660908, 0.9545330233905204]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9951019995704774, 0.00928474529021041, 5.738831654443486e-05]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [2.269001229017151e-10, 0.004352040747893529, 0.91599980780276]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [1.4988576099152378e-09, 0.03977009297886304, 0.9864694154710323]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.009450386296816818, 0.9974549177742216, 0.00735720229542597]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9959667923951812, 0.00901761531644246, 7.600579873837575e-05]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9962103264894578, 0.011943525272893168, 6.458922704968292e-05]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9974541019287463, 0.010320300912618069, 9.778895012008002e-05]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.019333074769685134, 0.9990126376641848, 0.017005450847435263]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9955173898413044, 0.013877002294621897, 6.051696073315418e-05]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9993840523152315, 0.012141818465899823, 0.0004081100834582919]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.014175975100705729, 0.9987216093867594, 0.014662036156536757]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9995722029327159, 0.02656524625427093, 0.00048040495261023313]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.0187235141051073, 0.9988292372878194, 0.01677744339126936]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.004960226718518766, 0.9973746962383823, 0.021618918264224032]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [6.604325291177865e-10, 0.012765114642712622, 0.9727726027641743]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [1.0209050125518944e-09, 0.022704010849701254, 0.9880404089839394]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9977058983416515, 0.021014033808067066, 9.860923346803218e-05]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.006445295883837529, 0.9919929982411063, 0.0027090398024949606]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9994727553419263, 0.016363861870229958, 0.0004523200557180727]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [2.2428369706338236e-10, 0.006067486867546869, 0.924877523108693]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.007639598062001561, 0.9968304289771708, 0.006188786067297876]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9989334853546985, 0.015036744689092359, 0.00021356599810354025]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.022739624473536224, 0.9991330471487391, 0.02209819574001805]
Epoch 5000: RMSE = 0.12463121217491074
Epoch 5100: RMSE = 0.11657746426385333
Epoch 5200: RMSE = 0.11302490071471717
Epoch 5300: RMSE = 0.11240511667039886
Epoch 5400: RMSE = 0.12533627371237535
Epoch 5500: RMSE = 0.11171351393928339
Epoch 5600: RMSE = 0.11246461250853955
Epoch 5700: RMSE = 0.11164475217696884
Epoch 5800: RMSE = 0.11159170302636919
Epoch 5900: RMSE = 0.11129068059832913
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [3.672518126902104e-10, 0.006837199295164142, 0.9608877265069958]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.01597921801029139, 0.9991003797596237, 0.013649421721738893]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [5.015187665260933e-10, 0.008076013170022572, 0.9711735051095469]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.003969586771903554, 0.9950647102422961, 0.002502275630764195]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [3.1310934257345373e-10, 0.00443870499443837, 0.953721405296937]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.008431269766427453, 0.9983841519802324, 0.006730831201910981]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9980757562373314, 0.008007351047346678, 0.00013909744924242413]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9983719350676948, 0.007537804454038687, 0.0001594142484392342]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [1.5138064003473443e-10, 0.002650627619000848, 0.8909559790634066]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [3.2522235040819493e-10, 0.00677396391638601, 0.9484138602638681]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.002957300248912775, 0.9961935793325165, 0.00747075257742075]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9998750049264156, 0.015687104602977973, 0.0016750158994368462]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9980765241834977, 0.008007041243731429, 0.0001392123635786207]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [3.4356529952667254e-06, 0.6936514456758814, 0.0864054142907519]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.008061905538339072, 0.9980476906815346, 0.005974507652274995]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [1.6221108959376291e-10, 0.0016407450981853237, 0.8567483666070537]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9998315087173367, 0.021943312535051163, 0.0012592814514061008]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [7.644577381493174e-10, 0.01599750363067511, 0.9812210289844259]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.01651189624653565, 0.9992442426989853, 0.013768223757397696]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9986232909192938, 0.008754359067799254, 0.00018082959257880433]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [1.9279022167013149e-10, 0.0026947491565161227, 0.9205109349015246]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [4.1397302567598797e-08, 0.21007394164203802, 0.9503419041250145]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9997722808479453, 0.013452561200645727, 0.0010738527316107626]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [5.616387025494401e-10, 0.007909281258508946, 0.9743290777725719]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.007952105672599763, 0.997748941436112, 0.004772827155265729]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [5.144223070230286e-10, 0.006681489696649142, 0.958841226263788]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.0056026676944338485, 0.9936688905395946, 0.0021849913935549706]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.00014524430191424008, 0.977916879118213, 0.03280555431386114]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9967266198921345, 0.014058026898355735, 8.481589718726973e-05]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9908844297714714, 0.006153088078610661, 6.277460750674892e-05]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.997972043296981, 0.012857124480902668, 0.000124788122379749]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9972730965119454, 0.012041644000089977, 9.785502625058559e-05]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9997608615121706, 0.01408438255987918, 0.0008216217724310903]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.002377851956878974, 0.996954086254449, 0.014679916304338592]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [8.308890890207553e-10, 0.012463097891041795, 0.9849203183915936]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.008311915661564936, 0.99810473843287, 0.005779633412622214]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9992443968915431, 0.010343765504139485, 0.0003324663267070838]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9997110159205559, 0.01000053947867014, 0.0008851044446641252]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9993822886329926, 0.009796900802207345, 0.00039218317871053107]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.012085273167075491, 0.9988254209475194, 0.009715897449164729]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.009125519098278195, 0.9985723654900149, 0.007054255710355853]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.9999371910803341, 0.013139732153569172, 0.003753546704106443]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [8.682979345626753e-10, 0.014122072826677488, 0.9887359458254522]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [2.995097657096664e-10, 0.005289606133498316, 0.9537661648843603]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [8.362414509593873e-10, 0.01355752028731351, 0.9844349680458413]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9991635723940799, 0.013327664168027784, 0.00025630165056845117]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.015004420714194862, 0.9992858721642549, 0.011907179819134056]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.013321363086157577, 0.998885252681341, 0.010829618567931253]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [2.9858864858122285e-10, 0.004658906174105132, 0.9388777524627184]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9990599111747787, 0.01753019181217933, 0.000245773376036176]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9994412981564673, 0.010068267261551305, 0.0004172883478647111]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9929467747002841, 0.014039361018267188, 3.798597816814021e-05]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [1.786761262029029e-05, 0.8610706446556203, 0.04045918128888942]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.010810080988031368, 0.9988392325653958, 0.00897006277992271]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [3.7229668627486473e-10, 0.008122219830286739, 0.958068652070256]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9991290065422694, 0.010858216225956627, 0.0002766964723307461]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9994652553284425, 0.009349931620781064, 0.0004734574562077276]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [2.001144621444149e-10, 0.0023339843587827512, 0.889743991122827]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9977853563816816, 0.010279482622895885, 0.00010899436575084139]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [1.4656956996742564e-10, 0.002404007735825707, 0.8982866317707283]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [3.7993613696689563e-10, 0.004315770734097624, 0.8850322771990754]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.019462222331522394, 0.9993370443760663, 0.017848700546061715]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9994869897408566, 0.01261182897258993, 0.0004489321402170537]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [1.2707300877199157e-09, 0.024878876618997358, 0.9871919865276942]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [2.1304064657680193e-10, 0.0030157772218603594, 0.9101375346211203]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9961357870237731, 0.018984571588830403, 5.929265859722354e-05]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.00653110978412333, 0.9975735144134338, 0.004978686046835294]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.00891460187018805, 0.9985379839086977, 0.006323453334409963]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.004352608512280607, 0.9979360145541898, 0.017599154681310385]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [1.703690228559212e-10, 0.0028150902463172462, 0.9028527368952245]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [5.927646806684291e-10, 0.008749832576676593, 0.9738659256402749]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [3.192131248173621e-10, 0.006009729275978349, 0.9501280127284587]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9960284360378043, 0.00939870510213564, 7.796858511701048e-05]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9988617066710803, 0.01010477298981727, 0.00021709183515699495]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.99902290719408, 0.0070909431781005835, 0.0002674916535686626]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.0068884766201003965, 0.9968465807425881, 0.003976019509792262]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.996562250370755, 0.0064866336773596765, 8.7586575767557e-05]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.017496791268687306, 0.999174366015253, 0.014353322665225664]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.00633480143752737, 0.9980537714737543, 0.007172103113992392]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [8.974573412867516e-06, 0.880996068404132, 0.16419986229571049]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [7.906243354147937e-11, 0.0012878373571330874, 0.7832434727789341]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.007655545224686002, 0.9984814083017379, 0.01241405424347742]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9969843388751098, 0.007986162085230914, 9.13065602240524e-05]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [4.007473071664036e-10, 0.007525303709887665, 0.9656066037767569]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [4.240128524070955e-10, 0.009988811436224966, 0.9675878738752588]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9998517353682314, 0.006793095634583158, 0.0016384446549413216]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9987841632083381, 0.018124936756857345, 0.00018608042822184013]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [4.986079370119428e-10, 0.005397676943914828, 0.8700646423675569]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [1.9070085278916176e-10, 0.003757446788626154, 0.9289403497235323]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9975447852783625, 0.007725508841108111, 0.00012263589465531607]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [6.0198483864650065e-09, 0.034256100712874535, 0.7181341171979206]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [5.218844804803725e-09, 0.0368686464128663, 0.8479963687548001]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.004017022810841231, 0.9951661786686266, 0.00235770808629828]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [4.5729773039795787e-10, 0.009387594076082058, 0.9698277301068905]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9980749768151319, 0.008019308467822365, 0.00013896825255760596]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [3.1178254397074356e-10, 0.005087003875753671, 0.9382621881786033]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.012292793756672494, 0.9990289381413532, 0.01186473063107197]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [1.3978704927974265e-10, 0.0017270476424107017, 0.8823622662203721]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [4.234028129089168e-10, 0.006560453154130393, 0.9579145282035934]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9947873961581616, 0.01392092781977951, 5.8865603336248684e-05]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.007583358069613343, 0.9985960551380827, 0.012930050727412209]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9993336818556106, 0.01303791647230203, 0.00034021214310296427]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9995080762848502, 0.007025221928790069, 0.0005532048400383763]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9981736454191736, 0.006060773025789772, 0.00015277448392674766]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.016094416028339418, 0.9991096199355999, 0.014123551944277838]
Epoch 6000: RMSE = 0.11297741243893059
Epoch 6100: RMSE = 0.1110519085660311
Epoch 6200: RMSE = 0.11158808978946166
Epoch 6300: RMSE = 0.11074917270297253
Epoch 6400: RMSE = 0.11111263152775701
Epoch 6500: RMSE = 0.11096862689440672
Epoch 6600: RMSE = 0.11324195481526333
Epoch 6700: RMSE = 0.11113580928823455
Epoch 6800: RMSE = 0.1108266908059809
Epoch 6900: RMSE = 0.11093559408923734
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9999095823645576, 0.0057073107759631906, 0.0027477701687790607]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.009548880256010058, 0.9990866106885361, 0.007710828618929995]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9978190924033379, 0.007195789167718823, 0.00012556458098020537]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9971861347494743, 0.017532181973471294, 8.121917862159105e-05]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [1.875399643828285e-10, 0.002492127332757348, 0.9127472153877155]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.015451776727485465, 0.9993496810903326, 0.012385332206457367]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [7.644875351647702e-10, 0.011706585948711311, 0.9890354910767349]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [1.6788091453127346e-10, 0.0031028481708448553, 0.9307306198703924]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9925047027479005, 0.0055561737226669655, 7.235023986248871e-05]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [1.1749708992504531e-05, 0.9138372471442955, 0.13671010754306362]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9986333296010312, 0.00717732947777148, 0.00019478523571683863]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.017212875976189642, 0.9994794817344603, 0.015416243665851675]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.006741395143563794, 0.9987908042430073, 0.0107574869339866]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9994366467287904, 0.009606935560801836, 0.00043299775105970484]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9995189692050755, 0.009106922984669894, 0.000529929107777667]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9996103461961214, 0.008579992924733446, 0.0006316328575641279]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [5.218851611181591e-10, 0.007241273802655442, 0.9746416906307194]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.999963424160063, 0.011016157685137518, 0.0066050186930807575]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.0061120154767813194, 0.9974882278431002, 0.0034053597997965317]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9976490756914227, 0.012863446418150123, 0.00011786679912469121]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9984700305074011, 0.009257098326722179, 0.0001584913544917519]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.014105657915484956, 0.9992938699496042, 0.011695501244691226]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [4.415182198333801e-10, 0.006701102434974117, 0.9716776843366319]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [1.696423615024369e-10, 0.002222892856145012, 0.9224898038504142]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [1.230259424509992e-10, 0.001426603843824375, 0.8843523977422034]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9986335892328321, 0.007176911440803573, 0.00019492842285847434]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [1.2900860329233489e-10, 0.00198706800547598, 0.9015452342491354]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9985638990716901, 0.0116546364124442, 0.00017601763183637014]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.999680339423689, 0.006092649191020442, 0.0008616433944821481]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [3.232934981748124e-10, 0.005671662025792309, 0.9616697214674039]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9992454675051385, 0.008981296061029338, 0.0003310537495346196]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9999302301696179, 0.013337275516052417, 0.003082599232810436]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.011156311162036789, 0.9992509212594681, 0.010096474042756627]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9998214119549196, 0.008635125803781332, 0.0014604660487171055]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9944861946002966, 0.012993102848494384, 4.7429646565651384e-05]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9996853096173921, 0.011048576190244163, 0.0007472770884901024]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9993961401759266, 0.015707255076389766, 0.00038773971737952504]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [3.470784581267482e-08, 0.18701515196191454, 0.9509242809267914]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.003759022381160363, 0.9983057865811232, 0.015506252650333735]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.007490574396103293, 0.9987358850035764, 0.00575247105170855]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.014574562621393382, 0.9994038739701394, 0.011876406267353068]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9974722950775874, 0.005826835303642836, 0.00011812729780378508]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9995927714491788, 0.011454781335080778, 0.0005663694846888853]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [2.8053640864162145e-10, 0.004952821143983085, 0.9517289845609602]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [3.344938738035292e-09, 0.02595727605500923, 0.8615439243144671]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.007327304245736981, 0.9985065526655292, 0.004976610508378371]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9998685877159084, 0.011548292278386699, 0.0019107212676948717]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [2.92302936442048e-10, 0.003265308990082653, 0.8929268484208445]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [6.725158269091929e-10, 0.01318796743566278, 0.9818298013675529]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.0066954500475983865, 0.9988831814853759, 0.011159376511448703]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9969854766365459, 0.008538086760348876, 0.00010111895118663264]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [1.4995111152908988e-10, 0.0023208612962558393, 0.9060847424118122]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.00016465329438989748, 0.9840088251098806, 0.027113799144457093]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.007867796249326112, 0.9988460492112503, 0.005465903648491184]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.999354587104027, 0.006225920137517711, 0.00041017913163086305]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9980635051136013, 0.010932457155814288, 0.00013787203846667604]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [2.8683450361496453e-10, 0.0056128090925674735, 0.9495079095906747]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9988886450918838, 0.0067042989473661005, 0.00023446258981583404]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.00806335154688527, 0.9988770395448867, 0.006081304392056941]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [3.733086915592939e-10, 0.008239284516437712, 0.9686769296940445]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.005756894889245813, 0.998086231108105, 0.004300591354585556]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [1.314705825435847e-10, 0.0012773521112438603, 0.8627535717483371]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [2.060667272701183e-09, 0.0154572788228687, 0.7789808633118323]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9987223296931032, 0.005373519238500694, 0.00021812751166384276]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.0033528440597414247, 0.9974730226111277, 0.005886812130682174]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9996662395044971, 0.00813753552078418, 0.0007719429015011494]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.007109634280767074, 0.9984565358571361, 0.005106133936314046]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9990953463316354, 0.007753911820236516, 0.00027815757561531535]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [2.602888120602928e-10, 0.0038247066148186988, 0.9405373845057836]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [4.4365746748176877e-10, 0.005448038515130039, 0.9601446802919652]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [2.7561748175037815e-10, 0.00367107575599361, 0.9545449526148584]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.01066070278430759, 0.999075159149521, 0.008360797027131068]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9998646542025111, 0.012095660694757789, 0.0014920615969101067]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9982233067765444, 0.006933006009800231, 0.0001686483717885982]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [2.9356190809228767e-10, 0.003378034428558985, 0.8873164790325673]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9999093249637762, 0.018820824473758834, 0.0024158299924218153]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.005452114960570903, 0.9984307154348216, 0.006253231036734923]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.002154023818593099, 0.9975685956893664, 0.01271377438463521]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.998633605193642, 0.007170105601427719, 0.00019496182762924742]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [2.636542009657239e-10, 0.004368662489496262, 0.9550082561084564]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9996547327531238, 0.008782110834846382, 0.0006879441680681233]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [2.725999974424581e-10, 0.004174555510321835, 0.9396233546331062]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [3.276770876658589e-10, 0.006703422904393828, 0.9592061577747375]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [1.0561527416769281e-05, 0.8590011764354233, 0.05293937999584916]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9992202897326757, 0.016281986727467664, 0.0002944261875624919]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [3.527767454795081e-10, 0.006211648707509475, 0.9666255529916892]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [1.332450210105361e-10, 0.002194124980968836, 0.8929043302461429]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [4.944127430316297e-10, 0.006542333203624154, 0.9750787143690522]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [2.434083786746074e-05, 0.9048382415100134, 0.03182806005120328]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [7.314795028064364e-10, 0.010322789100355475, 0.9853639360402867]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9959607352041719, 0.012842968073244249, 7.447258289243503e-05]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.014216402111256165, 0.9993000917180945, 0.012126593621991857]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [4.0257957090703954e-10, 0.007757487569984101, 0.9704992932986898]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.007068486677466172, 0.9982067931168285, 0.004089624664085545]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.0035069184759646296, 0.9960959736629106, 0.002134345421435553]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9994814281037974, 0.01179355879585538, 0.00042087279463671074]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.003552522286009364, 0.9961564618640505, 0.002012952451626147]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [1.1180399921927015e-09, 0.020618047612832875, 0.9876462704212962]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [7.361896879807418e-10, 0.011231197796262391, 0.9848908283614871]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.011752818557303929, 0.999123143841279, 0.009331823742211542]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [6.954436392341774e-11, 0.001062020879735213, 0.7892129503733936]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.005061713472845112, 0.9948239994720368, 0.001856551387672318]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [1.73208511117075e-10, 0.0019101162877339425, 0.8933464967977904]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.013255051300553938, 0.9994385507678112, 0.01023988329079584]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [3.715823314524353e-10, 0.005417464909480194, 0.958589189197533]
Epoch 7000: RMSE = 0.1108683948605092
Epoch 7100: RMSE = 0.11104054978634256
Epoch 7200: RMSE = 0.11093548787952184
Epoch 7300: RMSE = 0.11075510238167231
Epoch 7400: RMSE = 0.11076088926381279
Epoch 7500: RMSE = 0.11100613860681449
Epoch 7600: RMSE = 0.11097253811124194
Epoch 7700: RMSE = 0.1106175634933806
Epoch 7800: RMSE = 0.11065146019124493
Epoch 7900: RMSE = 0.11053023780879447
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.011976369169323566, 0.9995569089579934, 0.00904529332845451]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [2.585225244496864e-10, 0.005406197185956918, 0.9490294269079784]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [1.5291062942350684e-10, 0.0021353269268118108, 0.9222427044710511]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.01553787281161907, 0.9995893078037263, 0.013634196561931704]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [3.3669065202978656e-10, 0.007939726644026281, 0.9683898155577817]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9994031845773182, 0.015117237402258963, 0.00038453820372268604]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9999043810461156, 0.010722354332339465, 0.002133731340271493]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9964713441030055, 0.012305437408883822, 8.326863442589234e-05]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [4.4590948110658503e-10, 0.006296352615027883, 0.9749268829322578]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9996315673624261, 0.008281734819490742, 0.0006929334067601587]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.0024846396610620314, 0.9983762226590233, 0.010375271416164195]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9981851667597336, 0.006723004295256976, 0.00014884327025728816]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.9994920892239167, 0.005664554456985241, 0.0005194126141638923]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.006412084888201132, 0.9985695441464777, 0.0035793574720451382]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [1.1364804171922739e-08, 0.09739955710562719, 0.9645068761649237]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9999376920558689, 0.016660729323194675, 0.003558704229609675]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [1.1631127446151266e-10, 0.0012152803608711151, 0.8622155953921572]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [5.3043520240135525e-06, 0.8771341848980431, 0.15734018140249184]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [1.201413800878736e-10, 0.00211171851537629, 0.8912644216028333]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [4.706687475695842e-10, 0.0069606226678309005, 0.974319884417273]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9998639245469554, 0.007710989176567186, 0.0019182473566256482]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [2.4687164659194826e-10, 0.0040364489083001736, 0.9386555096952864]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.010100330362614303, 0.9994102784936576, 0.008855480640018386]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9997501128484458, 0.005469190770409904, 0.00109969109562878]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.0031621212485012963, 0.996900303108519, 0.0018575206528587077]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [6.597497936324109e-10, 0.009933399084262216, 0.9851280555055456]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.999740217707075, 0.007923661634505887, 0.000915230033245796]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [2.48568091399885e-10, 0.003537449774889869, 0.9539618253133528]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.999530517566638, 0.014542763933241908, 0.000496515629499691]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.006918781366835118, 0.9992018489278138, 0.009341479268313501]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9930724968591927, 0.005242026691987554, 7.41551512818214e-05]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.007296008983303184, 0.9991169522972981, 0.005314013520390254]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [1.3521056587298835e-10, 0.0022364219628995046, 0.904505896351903]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.009627071561357085, 0.9992716363901556, 0.007338005497300231]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [3.9816346076978137e-10, 0.006441006061482937, 0.9713576323141383]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9988071038582802, 0.010944216687414096, 0.00020875080813174935]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [3.21074289276344e-10, 0.003800407682641606, 0.8783802951499462]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9995622304400277, 0.008788029860512048, 0.0005557978439371917]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [3.6307369376013173e-10, 0.007463878181592485, 0.9700980853138451]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [3.704806184635473e-05, 0.9517067704296116, 0.03033099208088153]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.0036159570730341572, 0.9982487633962996, 0.0048628771954528265]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [4.110913375776654e-10, 0.005365069335734014, 0.9593285146730468]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [3.181531789073691e-10, 0.005980049829446794, 0.9662161239481435]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [1.1091239034047922e-10, 0.0013710962637220182, 0.8832342596362581]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [6.066286940635589e-10, 0.012716359556062103, 0.9815687240145519]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [1.5138150361989943e-10, 0.002983652459723521, 0.9301192668717502]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [2.531511776994334e-10, 0.004775991193556796, 0.9511910112330938]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.006415396148984006, 0.9987819945602647, 0.004482351554570272]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9996109278130599, 0.01077543362733075, 0.0005613553294654453]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [6.268432109568669e-11, 0.0010216171524395057, 0.7871360963225773]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.9999731760306507, 0.009545693408137013, 0.009067620248456332]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9996976506556433, 0.010404888192702227, 0.0007654900999512941]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.012736674319040358, 0.9994428830255715, 0.010315273152460483]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [3.354935894879128e-10, 0.005222626227982973, 0.9584159273970048]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [4.3254574162277126e-05, 0.9490675690446796, 0.022871591379751404]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [3.491231324614816e-09, 0.024768507948411428, 0.7336014025716552]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9999328188691304, 0.004957095378828827, 0.003726058208195897]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.013957639348612771, 0.9994868685508269, 0.010944608751705769]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.0052600555213593815, 0.9988215698348675, 0.005396238952229285]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [1.0078522674310053e-09, 0.019844922280259292, 0.9876081820468255]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9984005767384292, 0.010296034076231772, 0.00016498487179243806]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9973708842869858, 0.008078800670115191, 0.00011309691851507787]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.01316531637351851, 0.9995304793300223, 0.010489946697847353]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.0055317325477717465, 0.9979978508295769, 0.0029869185430355517]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [6.894564718613293e-10, 0.01126121242649207, 0.9890343777284542]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.006608636538005246, 0.9988230122386028, 0.0043818221043195925]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [3.394635385180578e-09, 0.028378324399608884, 0.8538693827850715]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9951737225881397, 0.012480860524564041, 5.2779036564966486e-05]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9997636168747714, 0.010000427868722099, 0.0009990192265946203]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9997031366295638, 0.007762291218825122, 0.000831853900272901]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.00020601230686794465, 0.9895280008807747, 0.021908801856566233]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9997482757600074, 0.0073271047524530964, 0.0010282603211183147]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9989646972879133, 0.0049417828437822164, 0.0002674495343812892]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.998059461492852, 0.012174959406293893, 0.00014155657323948668]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9976680619165326, 0.016753171399287448, 9.700225960226035e-05]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.006765947451329137, 0.9990046180306303, 0.005075493032710334]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.012837625809107057, 0.9994484332828101, 0.010718651500645117]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [2.3776725753352194e-10, 0.004209951434314204, 0.9550099001698443]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.00463892847537162, 0.9957802988801934, 0.0016066806157183668]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9988723129859175, 0.006657840660668876, 0.0002337793141359255]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [2.915450389114269e-10, 0.005451722129865705, 0.9616552146299343]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.010608494123715999, 0.9993082530752234, 0.008240001865704049]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [2.35283693173301e-10, 0.00369492395495728, 0.9405801383534809]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9999051373005531, 0.010243799074274832, 0.002669782548479562]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.00710349025421979, 0.9990899139261491, 0.004815620389577572]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [2.6377552287929706e-10, 0.0031702689121030298, 0.8923799051522947]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [2.955007460869558e-10, 0.006460342202365356, 0.9592524049890271]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [1.1631660630461762e-10, 0.0019096663127362377, 0.9017154955766453]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.0036477624920515273, 0.9987164343209561, 0.013489739690250744]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9987686568800126, 0.008625286757314901, 0.0001959090545574409]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.005198422811066638, 0.9984932885315373, 0.0037871730284113722]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.998872814584606, 0.006657494665449703, 0.00023405288376335663]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9988728815797905, 0.006657442758034495, 0.0002340678618492403]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [1.5273534941798347e-10, 0.0018102586159026053, 0.8937457388051623]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9999508353860778, 0.011688519189695438, 0.004419163809922618]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9994042908839635, 0.008258152061824712, 0.0004176337412316637]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9985124470059898, 0.006465384213241942, 0.00019865251333969612]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9991096096451036, 0.0061749516268727, 0.0002904195355961839]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.008620824559814775, 0.9992794399670002, 0.006798652604647355]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.005495003670233613, 0.998962406129847, 0.009887074774076162]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [6.639593956672459e-10, 0.010810552120957015, 0.9847803788556035]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9978645340918328, 0.005440263360038247, 0.00013732511883433028]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9992977627153287, 0.007114014466036924, 0.00035781603405131386]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.0032067909200290405, 0.9969436843541238, 0.0017598134997778285]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [1.6909507436278188e-10, 0.0023965078759163295, 0.9124532544045337]
Epoch 8000: RMSE = 0.11069870969866026
Epoch 8100: RMSE = 0.11072271269193963
Epoch 8200: RMSE = 0.11069304125543228
Epoch 8300: RMSE = 0.11069276907771426
Epoch 8400: RMSE = 0.11057308511518109
Epoch 8500: RMSE = 0.11051256440642251
Epoch 8600: RMSE = 0.1105888764101017
Epoch 8700: RMSE = 0.11047947781661435
Epoch 8800: RMSE = 0.11055756631523868
Epoch 8900: RMSE = 0.11056865778220419
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9997422656367237, 0.0071312338530369725, 0.0009469223763311064]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [1.0173929960386328e-10, 0.0013743977237046592, 0.8834173677361734]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.012092909034233766, 0.9996192662322342, 0.009416150776590571]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.002898210570679007, 0.9974720323090931, 0.001666201448720507]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [2.9190157582146255e-10, 0.005994392307094795, 0.9663244900448157]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9994698972174391, 0.007700021690625035, 0.00046261417231578027]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [1.1018934789471228e-10, 0.0021166900520015004, 0.8919708286259239]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [2.1847970693909563e-05, 0.9422619332356685, 0.033072118420472287]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9992048938622684, 0.005760344366024595, 0.0003202029165708838]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.004308880420536186, 0.9964743869235732, 0.0014125230864805113]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.011004022028590094, 0.9996408557274574, 0.008136717690095963]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [2.3715917991752136e-10, 0.00542204563189655, 0.9489201825225961]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9996790911834813, 0.007644866692638572, 0.0007869366709470358]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.006521869175816875, 0.999261352303339, 0.0043122623430891195]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.999744238843262, 0.009573587493324933, 0.0008984420737949736]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [5.564474622595542e-10, 0.012746759960547596, 0.9816314465967942]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [3.689870041816605e-10, 0.005295093147303019, 0.9598047695738496]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [6.092606216422501e-10, 0.010837683440411989, 0.9847584005540119]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9986209709466034, 0.006106074952049944, 0.00020961849966611332]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9999216118393477, 0.009653492850714384, 0.002593584603518014]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [3.653125846816379e-10, 0.006458286017820178, 0.9715357497714029]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.0062153313992996785, 0.9993400787180701, 0.008533249846193441]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.005086984777843726, 0.9983612406692736, 0.002659715510105454]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [2.441558410195612e-10, 0.003217098930624996, 0.8910331887501671]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9998823497323783, 0.007000302937919004, 0.002204637671365482]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [2.573807531372337e-10, 0.003465527101086122, 0.8838270717761469]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [2.4485700713841927e-05, 0.9375948413769227, 0.025194459313772313]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9999597129568121, 0.010425665214943683, 0.005378402724187044]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [3.089230408224858e-10, 0.007964582749770266, 0.9683830542598187]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [1.0776839703449363e-10, 0.001231882827911833, 0.8615215083374707]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.9999204959133982, 0.009238882978356082, 0.003160933264866922]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [1.187775281059448e-05, 0.9414486457658909, 0.11239840366122907]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.005885568832568468, 0.999011247576823, 0.004014809355568016]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9996671683084211, 0.009976767746191112, 0.0006497066338790053]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.9990637290386494, 0.004607822867421922, 0.0002895720659831133]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9989589854539994, 0.006267406358955053, 0.0002470173817028461]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.005781483513686365, 0.9992390543273918, 0.008459269963265795]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [2.3204345084018836e-10, 0.0047880650527658, 0.9509394566426029]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [2.0628414365201194e-08, 0.15866187487743666, 0.9537928895893616]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [2.5614680495334834e-09, 0.02527678699260403, 0.8591928786215378]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [2.6750784636420525e-10, 0.005458440738059487, 0.961131238408363]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.006690684259275685, 0.9992830800000683, 0.00478512180450357]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9994792424722967, 0.014199696337798587, 0.00043464360912814725]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.009737876947682808, 0.9994386424082231, 0.007363290479119813]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.014276534217433951, 0.9996672068443198, 0.012232977361382477]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.003377856274072462, 0.9989538762024901, 0.012090730936177527]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.004803733454692304, 0.9990386789120234, 0.0048404124688462115]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [1.3888007368517826e-10, 0.0029870986364301305, 0.9299171248249023]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.011694463841485031, 0.9995479405927967, 0.00925860787252681]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.0030877289965252795, 0.9984925225782819, 0.004498161046577464]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [1.5275761750840894e-09, 0.014672470264110917, 0.7790085458613757]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.0117841529048385, 0.9995526935721851, 0.009612134551527454]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.012820192367664997, 0.999583406254687, 0.009813838446733315]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.007913244837646462, 0.9994150389937762, 0.006106281018335738]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [2.980497651371949e-10, 0.0051057515442299375, 0.9587331736462551]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [2.71097264735566e-10, 0.006467646832751187, 0.9588073586733463]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9999433671217111, 0.004386315806791596, 0.004395578182376459]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [2.2805024186202e-10, 0.003541820567947917, 0.9541897908024938]
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9966281374582238, 0.011914036851465604, 8.453712396451642e-05]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9974867471343324, 0.0077308080149509, 0.00011452264929164431]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9996153143391214, 0.008150968257168597, 0.0006257301468273579]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9989608995188869, 0.0062623975934514445, 0.0002479798959603587]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.005908191991000775, 0.9988280709385557, 0.003191832639539745]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9995870341694335, 0.013629290835412908, 0.0005586230758011054]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [6.326159015381627e-10, 0.011273319720591143, 0.9889522004666406]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9985356289741778, 0.009794822952933612, 0.00017636390655714431]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.004764613659975375, 0.9987765414062957, 0.003383222624569478]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9983282737272297, 0.006360561306611592, 0.0001578880758484506]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [1.402416087318413e-10, 0.002138117360063631, 0.9219452708885155]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.002941683882445124, 0.9975020752363515, 0.0015702201154001833]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [3.3308977207640567e-10, 0.007472616113923381, 0.970203121818222]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.006060372540464014, 0.9990443930165304, 0.003925814413924102]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9988902547293987, 0.008135829926036488, 0.00021302225338908345]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.006206508519440935, 0.9991919447090696, 0.004546162041489575]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9978622755929447, 0.016153048893895648, 0.00010339998068227686]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9997786123052185, 0.007245365005632572, 0.001067410159945613]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [4.0910412608481965e-10, 0.00630525796973795, 0.9748303105710112]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.9999773884638011, 0.008434255929215338, 0.010698777239791043]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.00014132911711347545, 0.988935124275564, 0.022279015878597704]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9931345313382405, 0.0050134691179069355, 7.122826385461202e-05]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.002115582005781501, 0.9985861897353634, 0.009652741284811474]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [1.0670577298634394e-10, 0.0019118692654255395, 0.9007709458973789]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.00883351881299245, 0.9994089180626461, 0.006626389360190439]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.9995508546206104, 0.005226121733679146, 0.0005797085466078376]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.999949563852268, 0.0149821863092182, 0.004384588497115945]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [1.240260795227862e-10, 0.0022389747172125306, 0.9050768173532128]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9997832314146426, 0.006694598835758471, 0.001182427139116518]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [5.7474250644982606e-11, 0.001022523731258204, 0.7875944261419681]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [9.240232825614707e-10, 0.019853465798054595, 0.9875915839956586]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9980194815220998, 0.005139004821182723, 0.00014464521195595053]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9989624328560387, 0.006260941731526841, 0.00024876733836604015]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [4.318256972420338e-10, 0.006968405295243463, 0.974614200151725]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [2.1814828566521214e-10, 0.004214574389027938, 0.9548683335106427]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9993868421588176, 0.006606292744823356, 0.000405323653881627]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.009110298666793139, 0.9995149519962034, 0.008082327159039593]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [2.2490783636085567e-10, 0.004020249091580548, 0.9394751634177129]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [6.053213869796349e-10, 0.009944494636401582, 0.9853034436945396]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [2.1348435749036414e-10, 0.003669698792842018, 0.9406107291781662]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9982232876728917, 0.011643130512374473, 0.00015156251870809115]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [1.551339728975508e-10, 0.002398736477833897, 0.9127151824219941]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9989011009288301, 0.01039807147811668, 0.00022292316774179052]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [1.3721888549891167e-10, 0.0017854967257776846, 0.8939600428346084]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9997806082108643, 0.0049874176350592, 0.0012410429008861811]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.9997962982955978, 0.00918643296691086, 0.0011467129307810805]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.9954142350757875, 0.01210197024594171, 5.359783126521368e-05]
Epoch 9000: RMSE = 0.1102378161335869
Epoch 9100: RMSE = 0.11029193690956837
Epoch 9200: RMSE = 0.11034166589664739
Epoch 9300: RMSE = 0.11034743411876662
Epoch 9400: RMSE = 0.11040736629088763
Epoch 9500: RMSE = 0.11036176955473874
Epoch 9600: RMSE = 0.11031319616496159
Epoch 9700: RMSE = 0.11023658353958062
Epoch 9800: RMSE = 0.11040314082574437
Epoch 9900: RMSE = 0.11034267790680215
Sample [5.1 3.3 1.7 0.5] Expected [1. 0. 0.] Produced [0.9966588382939737, 0.011568400053828599, 8.217104961515339e-05]
Sample [4.8 3.4 1.9 0.2] Expected [1. 0. 0.] Produced [0.995484397106908, 0.011766347152666586, 5.229335590440569e-05]
Sample [4.6 3.6 1.  0.2] Expected [1. 0. 0.] Produced [0.9999492754748833, 0.003932511309162793, 0.004842526045961125]
Sample [5.4 3.9 1.3 0.4] Expected [1. 0. 0.] Produced [0.999928896565781, 0.00841447643829425, 0.003485870370241646]
Sample [5.5 3.5 1.3 0.2] Expected [1. 0. 0.] Produced [0.9998909928936245, 0.006425001316915187, 0.002334211410993019]
Sample [5.7 2.8 4.1 1.3] Expected [0. 1. 0.] Produced [0.005775091593913138, 0.9993258045330418, 0.004112687229500582]
Sample [6.1 2.6 5.6 1.4] Expected [0. 0. 1.] Produced [1.4415593921124023e-10, 0.0024153804737866683, 0.9114695374581776]
Sample [4.8 3.4 1.6 0.2] Expected [1. 0. 0.] Produced [0.9989490028310387, 0.007717143737695133, 0.00021898221382844036]
Sample [5.8 2.7 4.1 1. ] Expected [0. 1. 0.] Produced [0.005630897268177698, 0.9992018898417273, 0.003552420363391817]
Sample [5.1 3.8 1.6 0.2] Expected [1. 0. 0.] Produced [0.9996974910091689, 0.0092937713498622, 0.0007025682889809544]
Sample [6.3 3.4 5.6 2.4] Expected [0. 0. 1.] Produced [2.5191776205289213e-10, 0.006511440080853681, 0.9585093596691661]
Sample [5.6 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.0022895007494977797, 0.998939852494061, 0.008326661636362084]
Sample [7.1 3.  5.9 2.1] Expected [0. 0. 1.] Produced [3.394526374709155e-10, 0.0064925250255672105, 0.9712659928833645]
Sample [6.3 2.8 5.1 1.5] Expected [0. 0. 1.] Produced [2.095444903987132e-10, 0.00305389060051374, 0.8927036858268022]
Sample [5.9 3.2 4.8 1.8] Expected [0. 1. 0.] Produced [5.377543953924156e-06, 0.9113454079469036, 0.13492190073220187]
Sample [5.7 2.6 3.5 1. ] Expected [0. 1. 0.] Produced [0.0047377794947221, 0.998619356596627, 0.0023892257740739096]
Sample [5.7 2.5 5.  2. ] Expected [0. 0. 1.] Produced [9.914841810096828e-11, 0.0019259420354903111, 0.9000063531763738]
Sample [6.3 2.5 4.9 1.5] Expected [0. 1. 0.] Produced [1.2931067231071374e-10, 0.0018201457210929126, 0.8920113422644452]
Sample [6.7 3.1 4.7 1.5] Expected [0. 1. 0.] Produced [0.013290878192589586, 0.9997226347144659, 0.011078535652583646]
Sample [4.6 3.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9992499493498724, 0.005409989257353271, 0.0003303609289536489]
Sample [6.  2.9 4.5 1.5] Expected [0. 1. 0.] Produced [0.005967380678247141, 0.9994612949616214, 0.007621227577341087]
Sample [5.  2.3 3.3 1. ] Expected [0. 1. 0.] Produced [0.002691162773006353, 0.9978783403766145, 0.0014942808917810294]
Sample [7.6 3.  6.6 2.1] Expected [0. 0. 1.] Produced [5.661559604151083e-10, 0.010902256809445123, 0.9844957775183074]
Sample [5.1 3.5 1.4 0.3] Expected [1. 0. 0.] Produced [0.9997022203986191, 0.007114226922161603, 0.0008289969513192527]
Sample [5.5 4.2 1.4 0.2] Expected [1. 0. 0.] Produced [0.9999643173162841, 0.009411894666255537, 0.005982384790988203]
Sample [6.1 2.9 4.7 1.4] Expected [0. 1. 0.] Produced [0.003814255365964489, 0.9992433656849598, 0.010247186599015289]
Sample [6.2 2.8 4.8 1.8] Expected [0. 0. 1.] Produced [2.3814526174793746e-10, 0.0034962369388371614, 0.8817273209268983]
Sample [6.5 3.  5.2 2. ] Expected [0. 0. 1.] Produced [1.9985068738591525e-10, 0.0037202812101062408, 0.9392659669018191]
Sample [4.6 3.4 1.4 0.3] Expected [1. 0. 0.] Produced [0.9994315429927375, 0.006183322112616455, 0.00042647242995529994]
Sample [5.5 2.5 4.  1.3] Expected [0. 1. 0.] Produced [0.0030877799614949034, 0.998807735412963, 0.003968639466615327]
Sample [5.6 3.  4.1 1.3] Expected [0. 1. 0.] Produced [0.006063546954435167, 0.9993824841304141, 0.0038876266381228865]
Sample [6.7 2.5 5.8 1.8] Expected [0. 0. 1.] Produced [2.1191711461905158e-10, 0.0035682950618870805, 0.9535904298870612]
Sample [5.7 3.8 1.7 0.3] Expected [1. 0. 0.] Produced [0.9996120120458983, 0.012868407428851744, 0.0005800827953031378]
Sample [6.1 2.8 4.7 1.2] Expected [0. 1. 0.] Produced [0.005261580051086267, 0.9993513310681502, 0.007754689246935853]
Sample [5.7 2.9 4.2 1.3] Expected [0. 1. 0.] Produced [0.006226985201556002, 0.9994022274281131, 0.004329457953541826]
Sample [6.  3.4 4.5 1.6] Expected [0. 1. 0.] Produced [0.010236817143565462, 0.9997000087857663, 0.007347703632409339]
Sample [6.4 2.8 5.6 2.2] Expected [0. 0. 1.] Produced [2.0271420650354325e-10, 0.004246439179107402, 0.9540138719201561]
Sample [4.9 2.4 3.3 1. ] Expected [0. 1. 0.] Produced [0.002734043219955074, 0.9979020221368265, 0.0014089005928428853]
Sample [6.7 3.1 4.4 1.4] Expected [0. 1. 0.] Produced [0.01192508354864947, 0.9996521486375316, 0.008869601916371615]
Sample [5.9 3.  4.2 1.5] Expected [0. 1. 0.] Produced [0.00736488602803621, 0.9995120325061218, 0.005515013858010574]
Sample [5.8 2.8 5.1 2.4] Expected [0. 0. 1.] Produced [1.2904226692504852e-10, 0.003009305006943511, 0.929329342747899]
Sample [6.6 3.  4.4 1.4] Expected [0. 1. 0.] Produced [0.010874738389014918, 0.9996227076431802, 0.008387181310202252]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9989964273508601, 0.005931310435615047, 0.0002489832058786612]
Sample [7.4 2.8 6.1 1.9] Expected [0. 0. 1.] Produced [3.801623167586525e-10, 0.006352512261322337, 0.9744941780317391]
Sample [5.6 2.9 3.6 1.3] Expected [0. 1. 0.] Produced [0.005512911525880443, 0.9990127799734864, 0.0028657075058051555]
Sample [5.1 3.8 1.9 0.4] Expected [1. 0. 0.] Produced [0.9979479728729851, 0.015621804356865629, 0.00010431019777174972]
Sample [6.5 3.  5.8 2.2] Expected [0. 0. 1.] Produced [2.4856781010475743e-10, 0.005498845763895594, 0.9608214594825635]
Sample [5.  3.  1.6 0.2] Expected [1. 0. 0.] Produced [0.9975020829073962, 0.007437164739215363, 0.00011071669105226346]
Sample [7.2 3.6 6.1 2.5] Expected [0. 0. 1.] Produced [5.170198382506916e-10, 0.012821143926998487, 0.9813677189709895]
Sample [5.  3.2 1.2 0.2] Expected [1. 0. 0.] Produced [0.9997951883258036, 0.004598014464206062, 0.0013019074186423815]
Sample [5.1 3.4 1.5 0.2] Expected [1. 0. 0.] Produced [0.9995001172464422, 0.0072307517837791385, 0.0004783757352860692]
Sample [7.9 3.8 6.4 2. ] Expected [0. 0. 1.] Produced [1.31746952098474e-08, 0.1272668794784986, 0.9587055613057488]
Sample [5.4 3.  4.5 1.5] Expected [0. 1. 0.] Produced [0.0002147799336417404, 0.9934447933204055, 0.017169545723697503]
Sample [4.5 2.3 1.3 0.3] Expected [1. 0. 0.] Produced [0.9929870865988142, 0.00482254160493602, 6.578711358488417e-05]
Sample [6.  3.  4.8 1.8] Expected [0. 0. 1.] Produced [1.4473612103314842e-09, 0.015333083919050094, 0.774126251776506]
Sample [5.6 2.8 4.9 2. ] Expected [0. 0. 1.] Produced [1.0235760398182991e-10, 0.002126592125580609, 0.891038020547345]
Sample [7.3 2.9 6.3 1.8] Expected [0. 0. 1.] Produced [4.012805134765492e-10, 0.007011423271984722, 0.974271164371098]
Sample [4.4 3.  1.3 0.2] Expected [1. 0. 0.] Produced [0.999115941863278, 0.004319829762628931, 0.00029945025618969206]
Sample [5.7 4.4 1.5 0.4] Expected [1. 0. 0.] Produced [0.9999558900954852, 0.013614375477237331, 0.004948407230490737]
Sample [6.1 3.  4.6 1.4] Expected [0. 1. 0.] Produced [0.008537122365786482, 0.9995974210690817, 0.007288986230595117]
Sample [5.  3.6 1.4 0.2] Expected [1. 0. 0.] Produced [0.9997982086417521, 0.00668798543817757, 0.001149962525318188]
Sample [6.4 3.2 4.5 1.5] Expected [0. 1. 0.] Produced [0.011246179303501646, 0.9996818233900735, 0.008534766399556205]
Sample [5.6 2.7 4.2 1.3] Expected [0. 1. 0.] Produced [0.004458494872504139, 0.9991951886912651, 0.0044026833491148405]
Sample [6.7 3.1 5.6 2.4] Expected [0. 0. 1.] Produced [2.712193337510942e-10, 0.006022861396650845, 0.9660538665968115]
Sample [5.1 2.5 3.  1.1] Expected [0. 1. 0.] Produced [0.004049899816043509, 0.9969704162119222, 0.0012555517881300878]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9989986672736815, 0.005927285234597911, 0.00024995517931636395]
Sample [7.7 2.6 6.9 2.3] Expected [0. 0. 1.] Produced [5.878725172743975e-10, 0.01134159497374499, 0.9888685605946143]
Sample [6.4 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.009052329473556924, 0.9995310646922767, 0.0066874589712726]
Sample [5.8 4.  1.2 0.2] Expected [1. 0. 0.] Produced [0.9999796048642072, 0.007557746619192034, 0.011693932739312822]
Sample [6.3 2.5 5.  1.9] Expected [0. 0. 1.] Produced [1.3029144752773703e-10, 0.0021510917839554816, 0.9213797357401918]
Sample [5.  3.4 1.6 0.4] Expected [1. 0. 0.] Produced [0.9985925847850065, 0.009362537666624753, 0.00017828610880491448]
Sample [4.9 2.5 4.5 1.7] Expected [0. 0. 1.] Produced [5.3386590042455224e-11, 0.001028718003314663, 0.7859360385108841]
Sample [6.9 3.1 5.4 2.1] Expected [0. 0. 1.] Produced [2.7610083867106804e-10, 0.005126807824318625, 0.9586471321184813]
Sample [6.  2.2 5.  1.5] Expected [0. 0. 1.] Produced [9.451433155843301e-11, 0.0013809219814286693, 0.8831274917433675]
Sample [6.2 2.2 4.5 1.5] Expected [0. 1. 0.] Produced [9.607721547795317e-11, 0.0012001154124691, 0.8624017207570694]
Sample [6.6 2.9 4.6 1.3] Expected [0. 1. 0.] Produced [0.010962953062578717, 0.9996268340197546, 0.0086982761701022]
Sample [6.4 3.2 5.3 2.3] Expected [0. 0. 1.] Produced [2.1510583776255637e-10, 0.004799587098604529, 0.9507077691129607]
Sample [6.2 3.4 5.4 2.3] Expected [0. 0. 1.] Produced [2.1997313063387422e-10, 0.0054402835801206735, 0.9482802329866278]
Sample [5.1 3.8 1.5 0.3] Expected [1. 0. 0.] Produced [0.9997675762861127, 0.00887831245539804, 0.0009695714669074027]
Sample [5.  3.5 1.3 0.3] Expected [1. 0. 0.] Produced [0.9998010231392952, 0.006176175647068521, 0.001262919002038017]
Sample [6.4 3.1 5.5 1.8] Expected [0. 0. 1.] Produced [2.0886727203884577e-10, 0.004044202641091064, 0.9384439808654136]
Sample [6.2 2.9 4.3 1.3] Expected [0. 1. 0.] Produced [0.008212040159766518, 0.9995064051219956, 0.005987843673491804]
Sample [4.9 3.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9989956374243367, 0.005929354932732005, 0.0002488803382287493]
Sample [5.4 3.4 1.7 0.2] Expected [1. 0. 0.] Produced [0.9989284107261661, 0.009941921814506803, 0.0002210359576659922]
Sample [7.7 3.8 6.7 2.2] Expected [0. 0. 1.] Produced [8.585750122060257e-10, 0.019973958091129458, 0.9873769372268457]
Sample [6.3 3.3 6.  2.5] Expected [0. 0. 1.] Produced [2.870486459056851e-10, 0.007999749666490676, 0.9678980086160315]
Sample [5.1 3.5 1.4 0.2] Expected [1. 0. 0.] Produced [0.9997616219285904, 0.006607023949912321, 0.0010031095235019456]
Sample [4.8 3.  1.4 0.3] Expected [1. 0. 0.] Produced [0.9986572105814331, 0.005798919280780635, 0.00020841124986212988]
Sample [4.4 2.9 1.4 0.2] Expected [1. 0. 0.] Produced [0.9980773432221661, 0.004884441115869034, 0.0001438163479391278]
Sample [5.2 4.1 1.5 0.1] Expected [1. 0. 0.] Produced [0.9999306410884206, 0.008780698264824766, 0.0028883634118856967]
Sample [7.7 2.8 6.7 2. ] Expected [0. 0. 1.] Produced [5.62503480052668e-10, 0.010006044184473224, 0.9850421887887703]
Sample [5.5 2.3 4.  1.3] Expected [0. 1. 0.] Produced [1.2738873421858104e-05, 0.9158363043714719, 0.028783780131896846]
Sample [5.4 3.7 1.5 0.2] Expected [1. 0. 0.] Produced [0.999812468113034, 0.008517287168250981, 0.0012212039430242662]
Sample [5.8 2.7 3.9 1.2] Expected [0. 1. 0.] Produced [0.005468671306238277, 0.9991723111465881, 0.0036372443660735785]
Sample [5.6 2.5 3.9 1.1] Expected [0. 1. 0.] Produced [0.0044312365319643405, 0.9989792252798837, 0.0030540550159872293]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.] Produced [1.1523418836529318e-10, 0.0022537229928423245, 0.9039445285703478]
Sample [5.4 3.9 1.7 0.4] Expected [1. 0. 0.] Produced [0.9995181079473356, 0.013410902421432792, 0.0004597581795021624]
Sample [4.7 3.2 1.3 0.2] Expected [1. 0. 0.] Produced [0.9995800804953048, 0.004863045712192664, 0.0006059713739307977]
Sample [5.2 3.5 1.5 0.2] Expected [1. 0. 0.] Produced [0.9996405436890077, 0.007621420978231933, 0.0006546361072136217]
Sample [6.5 3.2 5.1 2. ] Expected [0. 0. 1.] Produced [1.979681525864254e-09, 0.0226241885390035, 0.8653082695693991]
Sample [6.7 3.3 5.7 2.5] Expected [0. 0. 1.] Produced [3.095093975603254e-10, 0.007521071299090166, 0.9699218406623118]
Sample [5.  3.5 1.6 0.6] Expected [1. 0. 0.] Produced [0.9982904282178817, 0.011183091479041484, 0.00015260424634167845]
Sample [4.6 3.1 1.5 0.2] Expected [1. 0. 0.] Produced [0.9983888149761815, 0.00605079656143084, 0.00015880855660258935]
Sample [5.5 2.6 4.4 1.2] Expected [0. 1. 0.] Produced [1.3574556256552648e-05, 0.9312000024799948, 0.035612751765601816]
Sample [7.2 3.  5.8 1.6] Expected [0. 0. 1.] Produced [3.3774682726325644e-10, 0.0052652566445501705, 0.9595808537521041]
Epoch 10000: RMSE = 0.11031701533140159
Sample [4.9 3.  1.4 0.2] Expected [1. 0. 0.], Produced [0.9990176859932002, 0.005468757568010362, 0.00027598901074757273]
Sample [5.  3.4 1.5 0.2] Expected [1. 0. 0.], Produced [0.9994584198690605, 0.007077475609720273, 0.0004384203729691915]
Sample [4.8 3.  1.4 0.1] Expected [1. 0. 0.], Produced [0.9991541042796491, 0.0049588540712404575, 0.0003068508698106765]
Sample [4.3 3.  1.1 0.1] Expected [1. 0. 0.], Produced [0.9997119624277663, 0.0030409025069682415, 0.0009066730425349897]
Sample [5.1 3.7 1.5 0.4] Expected [1. 0. 0.], Produced [0.999627927011089, 0.009235599206301084, 0.0006395196184996566]
Sample [5.2 3.4 1.4 0.2] Expected [1. 0. 0.], Produced [0.9997188372615506, 0.0065307190790483026, 0.0008783737798155217]
Sample [4.7 3.2 1.6 0.2] Expected [1. 0. 0.], Produced [0.9980987736901761, 0.00723745219169027, 0.00012994599900046679]
Sample [4.8 3.1 1.6 0.2] Expected [1. 0. 0.], Produced [0.9977312075145299, 0.007258721199551181, 0.00011481430771194357]
Sample [5.4 3.4 1.5 0.4] Expected [1. 0. 0.], Produced [0.9993789177730701, 0.008954919853175867, 0.00042242846287149924]
Sample [4.4 3.2 1.3 0.2] Expected [1. 0. 0.], Produced [0.9994670036276743, 0.004561978361982053, 0.0004663664311524259]
Sample [5.3 3.7 1.5 0.2] Expected [1. 0. 0.], Produced [0.999797355036522, 0.008330517737551116, 0.0011206575965036786]
Sample [5.  3.3 1.4 0.2] Expected [1. 0. 0.], Produced [0.999577496297363, 0.006057537360515637, 0.0005890789244705544]
Sample [7.  3.2 4.7 1.4] Expected [0. 1. 0.], Produced [0.015806112340148185, 0.9997469254602747, 0.012275532869591446]
Sample [6.9 3.1 4.9 1.5] Expected [0. 1. 0.], Produced [0.015520000156184894, 0.9997631266721634, 0.01378456025927166]
Sample [6.5 2.8 4.6 1.5] Expected [0. 1. 0.], Produced [0.008084708590544562, 0.9995508981339399, 0.010263728114122793]
Sample [5.7 2.8 4.5 1.3] Expected [0. 1. 0.], Produced [0.0017902385502819708, 0.9985261740538621, 0.008189953843368118]
Sample [6.3 3.3 4.7 1.6] Expected [0. 1. 0.], Produced [0.012026788092910119, 0.9997380345880617, 0.009628541255508431]
Sample [5.2 2.7 3.9 1.4] Expected [0. 1. 0.], Produced [0.003690787748727396, 0.9990437063354344, 0.0031343599771344916]
Sample [5.  2.  3.5 1. ] Expected [0. 1. 0.], Produced [0.0011358149732236223, 0.9962818398069967, 0.0021762021236933792]
Sample [6.  2.2 4.  1. ] Expected [0. 1. 0.], Produced [0.004561653828299335, 0.9988771281225771, 0.0037847907158328146]
Sample [6.1 2.8 4.  1.3] Expected [0. 1. 0.], Produced [0.006882352611282109, 0.9993511172249655, 0.004832315153616881]
Sample [6.8 2.8 4.8 1.4] Expected [0. 1. 0.], Produced [0.010442864034655197, 0.9996281299536812, 0.01228103662999873]
Sample [6.7 3.  5.  1.7] Expected [0. 1. 0.], Produced [8.791486043307379e-05, 0.987675150368665, 0.0915847966933843]
Sample [5.5 2.4 3.8 1.1] Expected [0. 1. 0.], Produced [0.003892414754347515, 0.9988136164540489, 0.0027521650572730804]
Sample [5.5 2.4 3.7 1. ] Expected [0. 1. 0.], Produced [0.0038168429397492667, 0.9986826294113009, 0.002436865978788059]
Sample [6.  2.7 5.1 1.6] Expected [0. 1. 0.], Produced [1.2191229517536956e-10, 0.002079565047945785, 0.8959780211134314]
Sample [6.3 2.3 4.4 1.3] Expected [0. 1. 0.], Produced [8.239020379577071e-06, 0.8882291711437637, 0.07479769906723151]
Sample [5.8 2.6 4.  1.2] Expected [0. 1. 0.], Produced [0.005328175665486441, 0.9991799098861998, 0.003840054818358324]
Sample [5.7 3.  4.2 1.2] Expected [0. 1. 0.], Produced [0.006465558963491503, 0.9994084665925939, 0.004064723000273637]
Sample [6.3 2.9 5.6 1.8] Expected [0. 0. 1.], Produced [1.8937900636368855e-10, 0.003662733778267718, 0.9381757670564874]
Sample [6.4 2.7 5.3 1.9] Expected [0. 0. 1.], Produced [1.6600235193315673e-10, 0.002949031602667436, 0.9357205591983411]
Sample [6.8 3.  5.5 2.1] Expected [0. 0. 1.], Produced [2.5210757476320944e-10, 0.0048126198680354905, 0.9587661454881111]
Sample [6.5 3.  5.5 1.8] Expected [0. 0. 1.], Produced [2.0956266465056011e-10, 0.003893227560648978, 0.9415994059630489]
Sample [6.9 3.2 5.7 2.3] Expected [0. 0. 1.], Produced [3.180851155352196e-10, 0.0068166486733246935, 0.9691258736959113]
Sample [6.3 2.7 4.9 1.8] Expected [0. 0. 1.], Produced [1.4220681108135202e-10, 0.002304247313891951, 0.9110880780298952]
Sample [6.7 3.3 5.7 2.1] Expected [0. 0. 1.], Produced [2.928371560478008e-10, 0.0062956535575647244, 0.9610525805258079]
Sample [7.2 3.2 6.  1.8] Expected [0. 0. 1.], Produced [3.8637074879277897e-10, 0.006987750328758818, 0.9686598002129304]
Sample [6.1 3.  4.9 1.8] Expected [0. 0. 1.], Produced [4.74541720003209e-10, 0.006578386391817294, 0.8567784009524848]
Sample [6.4 2.8 5.6 2.1] Expected [0. 0. 1.], Produced [1.997178517499061e-10, 0.004055051725042888, 0.9512210092116615]
Sample [7.7 3.  6.1 2.3] Expected [0. 0. 1.], Produced [5.038187070609881e-10, 0.009282931618392751, 0.9834801669717521]
Sample [6.9 3.1 5.1 2.3] Expected [0. 0. 1.], Produced [3.453725624116596e-10, 0.006047394502052754, 0.9525956341624187]
Sample [5.8 2.7 5.1 1.9] Expected [0. 0. 1.], Produced [1.152342890348327e-10, 0.0022542302202514337, 0.9041586203309742]
Sample [6.8 3.2 5.9 2.3] Expected [0. 0. 1.], Produced [3.2752339289636674e-10, 0.007418057791575034, 0.9705495882353287]
Sample [6.7 3.  5.2 2.3] Expected [0. 0. 1.], Produced [2.2140357165988892e-10, 0.004359669490292857, 0.9555341453918877]
Sample [5.9 3.  5.1 1.8] Expected [0. 0. 1.], Produced [1.3730900757655028e-10, 0.0027022122117812152, 0.9034404252848545]
Final Test RMSE =  0.11861390836185241
Sample [1.51] Expected [0.99815247] Produced [0.8462566822871772]
Sample [0.51] Expected [0.48817725] Produced [0.7393441042871504]
Sample [1.24] Expected [0.945784] Produced [0.8219587001257624]
Sample [0.15] Expected [0.14943813] Produced [0.6884830810740974]
Sample [0.65] Expected [0.60518641] Produced [0.7560638651429042]
Sample [0.89] Expected [0.77707175] Produced [0.784396246280299]
Sample [1.49] Expected [0.99673775] Produced [0.8434228843081207]
Sample [0.66] Expected [0.61311685] Produced [0.7572667907376518]
Sample [1.39] Expected [0.98370081] Produced [0.8347584229774515]
Sample [0.29] Expected [0.28595223] Produced [0.7077408710449729]
Sample [0.71] Expected [0.65183377] Produced [0.7626078898585724]
Sample [1.43] Expected [0.99010456] Produced [0.837590770236583]
Sample [0.59] Expected [0.55636102] Produced [0.7476134572035715]
Sample [0.96] Expected [0.81919157] Produced [0.7913081236569306]
Sample [0.9] Expected [0.78332691] Produced [0.784681056449429]
Epoch 0: RMSE = 0.22092823353370347
Epoch 100: RMSE = 0.20737405877653659
Epoch 200: RMSE = 0.20681416788909088
Epoch 300: RMSE = 0.20636086518068347
Epoch 400: RMSE = 0.20598131900476457
Epoch 500: RMSE = 0.2056555738202346
Epoch 600: RMSE = 0.20537054267767332
Epoch 700: RMSE = 0.20511643392936876
Epoch 800: RMSE = 0.20488476746861142
Epoch 900: RMSE = 0.20467656467508455
Sample [0.96] Expected [0.81919157] Produced [0.7267439601658656]
Sample [1.39] Expected [0.98370081] Produced [0.7746891859173357]
Sample [1.24] Expected [0.945784] Produced [0.7597453612371401]
Sample [1.43] Expected [0.99010456] Produced [0.7798282831149722]
Sample [0.71] Expected [0.65183377] Produced [0.696765540496891]
Sample [1.49] Expected [0.99673775] Produced [0.7861094771951169]
Sample [0.29] Expected [0.28595223] Produced [0.6362580579478968]
Sample [0.65] Expected [0.60518641] Produced [0.6880657299566457]
Sample [1.51] Expected [0.99815247] Produced [0.787404835747141]
Sample [0.51] Expected [0.48817725] Produced [0.6686206038854086]
Sample [0.15] Expected [0.14943813] Produced [0.61325404653792]
Sample [0.9] Expected [0.78332691] Produced [0.7195063145892552]
Sample [0.66] Expected [0.61311685] Produced [0.6882114706696217]
Sample [0.89] Expected [0.77707175] Produced [0.7182221012168835]
Sample [0.59] Expected [0.55636102] Produced [0.6784265238975624]
Epoch 1000: RMSE = 0.2044831391655721
Epoch 1100: RMSE = 0.20430992057571268
Epoch 1200: RMSE = 0.2041446497254187
Epoch 1300: RMSE = 0.20398776432557297
Epoch 1400: RMSE = 0.20383819565802969
Epoch 1500: RMSE = 0.20368241175252877
Epoch 1600: RMSE = 0.20354895358923183
Epoch 1700: RMSE = 0.20340671609196212
Epoch 1800: RMSE = 0.20326062014073634
Epoch 1900: RMSE = 0.20310762951511693
Sample [0.96] Expected [0.81919157] Produced [0.7285468441779313]
Sample [0.66] Expected [0.61311685] Produced [0.6889072085548855]
Sample [0.51] Expected [0.48817725] Produced [0.6662400173566781]
Sample [1.43] Expected [0.99010456] Produced [0.7785731567563696]
Sample [0.65] Expected [0.60518641] Produced [0.687326218052298]
Sample [0.9] Expected [0.78332691] Produced [0.7209722686301183]
Sample [0.15] Expected [0.14943813] Produced [0.6064922517211417]
Sample [1.49] Expected [0.99673775] Produced [0.7834624205509878]
Sample [0.71] Expected [0.65183377] Produced [0.6951906486046109]
Sample [1.51] Expected [0.99815247] Produced [0.7857869300467387]
Sample [0.89] Expected [0.77707175] Produced [0.7197947895164114]
Sample [1.24] Expected [0.945784] Produced [0.7603988943517711]
Sample [0.29] Expected [0.28595223] Produced [0.6312348871624789]
Sample [1.39] Expected [0.98370081] Produced [0.7750036769646975]
Sample [0.59] Expected [0.55636102] Produced [0.6787397953771419]
Epoch 2000: RMSE = 0.20294676143187815
Epoch 2100: RMSE = 0.20276571404067514
Epoch 2200: RMSE = 0.2025656063798833
Epoch 2300: RMSE = 0.20233416455935269
Epoch 2400: RMSE = 0.2020571289162423
Epoch 2500: RMSE = 0.2017205833819861
Epoch 2600: RMSE = 0.20129250326238116
Epoch 2700: RMSE = 0.2007317640442163
Epoch 2800: RMSE = 0.19997474055461634
Epoch 2900: RMSE = 0.19891122790627536
Sample [1.39] Expected [0.98370081] Produced [0.7764283516127238]
Sample [0.15] Expected [0.14943813] Produced [0.5879741280239336]
Sample [0.71] Expected [0.65183377] Produced [0.6925781395489858]
Sample [0.59] Expected [0.55636102] Produced [0.6727062727886839]
Sample [1.49] Expected [0.99673775] Produced [0.7844711739572628]
Sample [0.65] Expected [0.60518641] Produced [0.6830471357201554]
Sample [0.29] Expected [0.28595223] Produced [0.6161377518595448]
Sample [0.89] Expected [0.77707175] Produced [0.7180011058954532]
Sample [0.51] Expected [0.48817725] Produced [0.6578476433499705]
Sample [0.9] Expected [0.78332691] Produced [0.7190358323051166]
Sample [1.24] Expected [0.945784] Produced [0.7598593271777724]
Sample [0.96] Expected [0.81919157] Produced [0.7277592207493716]
Sample [1.43] Expected [0.99010456] Produced [0.7792210974072284]
Sample [1.51] Expected [0.99815247] Produced [0.7869024604642558]
Sample [0.66] Expected [0.61311685] Produced [0.6854026257457645]
Epoch 3000: RMSE = 0.1973619534368371
Epoch 3100: RMSE = 0.19504009947162634
Epoch 3200: RMSE = 0.19149803217772954
Epoch 3300: RMSE = 0.1861949585939492
Epoch 3400: RMSE = 0.17880626263644747
Epoch 3500: RMSE = 0.1696121887019182
Epoch 3600: RMSE = 0.15946863140673634
Epoch 3700: RMSE = 0.1492593874800518
Epoch 3800: RMSE = 0.13950854560645734
Epoch 3900: RMSE = 0.13040772481474966
Sample [0.29] Expected [0.28595223] Produced [0.4704287100395309]
Sample [0.65] Expected [0.60518641] Produced [0.6429275669265708]
Sample [1.24] Expected [0.945784] Produced [0.8125454815693554]
Sample [0.15] Expected [0.14943813] Produced [0.3962967750131608]
Sample [0.51] Expected [0.48817725] Produced [0.5804381712591794]
Sample [0.9] Expected [0.78332691] Produced [0.7313437397421598]
Sample [1.49] Expected [0.99673775] Produced [0.8505457014019565]
Sample [1.43] Expected [0.99010456] Produced [0.8427496907236798]
Sample [1.39] Expected [0.98370081] Produced [0.8372373994103391]
Sample [1.51] Expected [0.99815247] Produced [0.8535746335046214]
Sample [0.59] Expected [0.55636102] Produced [0.6178644302265563]
Sample [0.66] Expected [0.61311685] Produced [0.647371396574761]
Sample [0.96] Expected [0.81919157] Produced [0.7495034971871213]
Sample [0.71] Expected [0.65183377] Produced [0.6673038644772302]
Sample [0.89] Expected [0.77707175] Produced [0.7293049788118522]
Epoch 4000: RMSE = 0.12196592527675398
Epoch 4100: RMSE = 0.11413224790089858
Epoch 4200: RMSE = 0.1068463880317882
Epoch 4300: RMSE = 0.1000684634574708
Epoch 4400: RMSE = 0.09377891555381179
Epoch 4500: RMSE = 0.08796997329757526
Epoch 4600: RMSE = 0.08263827870205061
Epoch 4700: RMSE = 0.07777453310388285
Epoch 4800: RMSE = 0.07336250453347774
Epoch 4900: RMSE = 0.06937922772742613
Sample [0.9] Expected [0.78332691] Produced [0.7627182029509086]
Sample [1.24] Expected [0.945784] Produced [0.8626144309555336]
Sample [0.59] Expected [0.55636102] Produced [0.5896211704768506]
Sample [1.39] Expected [0.98370081] Produced [0.8882384027111145]
Sample [0.71] Expected [0.65183377] Produced [0.6677414638263336]
Sample [1.51] Expected [0.99815247] Produced [0.9038146861802445]
Sample [0.51] Expected [0.48817725] Produced [0.5307264434520153]
Sample [1.43] Expected [0.99010456] Produced [0.8938681957209502]
Sample [1.49] Expected [0.99673775] Produced [0.9015214155468925]
Sample [0.89] Expected [0.77707175] Produced [0.7588699919100775]
Sample [0.96] Expected [0.81919157] Produced [0.786368654386128]
Sample [0.29] Expected [0.28595223] Produced [0.355743174590351]
Sample [0.66] Expected [0.61311685] Produced [0.6369783007693995]
Sample [0.15] Expected [0.14943813] Produced [0.2524125674515163]
Sample [0.65] Expected [0.60518641] Produced [0.6302486186852979]
Epoch 5000: RMSE = 0.06579389158171621
Epoch 5100: RMSE = 0.06257449822615631
Epoch 5200: RMSE = 0.05968701512979189
Epoch 5300: RMSE = 0.057098025400956216
Epoch 5400: RMSE = 0.054776572451711227
Epoch 5500: RMSE = 0.05269343564687673
Epoch 5600: RMSE = 0.05082209651115551
Epoch 5700: RMSE = 0.049138964707740006
Epoch 5800: RMSE = 0.04762261191329023
Epoch 5900: RMSE = 0.04625427767532893
Sample [1.43] Expected [0.99010456] Produced [0.9154492566172141]
Sample [1.24] Expected [0.945784] Produced [0.8855071033746342]
Sample [0.96] Expected [0.81919157] Produced [0.8055289888132032]
Sample [0.66] Expected [0.61311685] Produced [0.6338283250951607]
Sample [0.29] Expected [0.28595223] Produced [0.30539398298623804]
Sample [0.71] Expected [0.65183377] Produced [0.6702159550085844]
Sample [0.51] Expected [0.48817725] Produced [0.5071067031383151]
Sample [0.89] Expected [0.77707175] Produced [0.7749539980436815]
Sample [0.59] Expected [0.55636102] Produced [0.5773613398363345]
Sample [1.39] Expected [0.98370081] Produced [0.9101840992296949]
Sample [0.65] Expected [0.60518641] Produced [0.6259304111263819]
Sample [1.51] Expected [0.99815247] Produced [0.9245687746011505]
Sample [1.49] Expected [0.99673775] Produced [0.9224596758856158]
Sample [0.9] Expected [0.78332691] Produced [0.7797032266413754]
Sample [0.15] Expected [0.14943813] Produced [0.19714962657256996]
Epoch 6000: RMSE = 0.04501732338220306
Epoch 6100: RMSE = 0.04389701891297512
Epoch 6200: RMSE = 0.0428802450781045
Epoch 6300: RMSE = 0.041955684715661246
Epoch 6400: RMSE = 0.04111311880885614
Epoch 6500: RMSE = 0.040343817760145416
Epoch 6600: RMSE = 0.03963971390502395
Epoch 6700: RMSE = 0.0389938673517404
Epoch 6800: RMSE = 0.03840014412610228
Epoch 6900: RMSE = 0.037853240847557056
Sample [0.96] Expected [0.81919157] Produced [0.8150088360004953]
Sample [0.9] Expected [0.78332691] Produced [0.7880956590171729]
Sample [1.24] Expected [0.945784] Produced [0.8963964750682011]
Sample [1.49] Expected [0.99673775] Produced [0.9322393773118686]
Sample [0.51] Expected [0.48817725] Produced [0.4955652362369392]
Sample [0.59] Expected [0.55636102] Produced [0.571375010996495]
Sample [0.66] Expected [0.61311685] Produced [0.632079759419406]
Sample [0.29] Expected [0.28595223] Produced [0.2826646809855017]
Sample [0.89] Expected [0.77707175] Produced [0.7831733963105384]
Sample [0.71] Expected [0.65183377] Produced [0.6713649299288269]
Sample [1.51] Expected [0.99815247] Produced [0.9342139310460141]
Sample [0.15] Expected [0.14943813] Produced [0.17439284016546103]
Sample [1.43] Expected [0.99010456] Produced [0.9255416219333926]
Sample [1.39] Expected [0.98370081] Produced [0.9205410722675252]
Sample [0.65] Expected [0.60518641] Produced [0.6238021362513197]
Epoch 7000: RMSE = 0.03734831408864892
Epoch 7100: RMSE = 0.036880985718007846
Epoch 7200: RMSE = 0.03644753390613427
Epoch 7300: RMSE = 0.036044657926593816
Epoch 7400: RMSE = 0.03566928342579839
Epoch 7500: RMSE = 0.03531882780468651
Epoch 7600: RMSE = 0.03499082292046063
Epoch 7700: RMSE = 0.03468329555645669
Epoch 7800: RMSE = 0.03439424640260137
Epoch 7900: RMSE = 0.03412202756898418
Sample [0.59] Expected [0.55636102] Produced [0.5675748063317949]
Sample [0.96] Expected [0.81919157] Produced [0.8202227856981124]
Sample [0.71] Expected [0.65183377] Produced [0.6716860850588702]
Sample [1.51] Expected [0.99815247] Produced [0.9398019173059404]
Sample [0.65] Expected [0.60518641] Produced [0.6220848925112572]
Sample [1.49] Expected [0.99673775] Produced [0.9378444137255326]
Sample [1.39] Expected [0.98370081] Produced [0.9264506987226895]
Sample [0.51] Expected [0.48817725] Produced [0.4888930435746556]
Sample [0.66] Expected [0.61311685] Produced [0.630764199882866]
Sample [0.9] Expected [0.78332691] Produced [0.7925560267303491]
Sample [0.89] Expected [0.77707175] Produced [0.7874820808642967]
Sample [1.43] Expected [0.99010456] Produced [0.9313513652917942]
Sample [0.15] Expected [0.14943813] Produced [0.16394641349844932]
Sample [1.24] Expected [0.945784] Produced [0.9025721134997632]
Sample [0.29] Expected [0.28595223] Produced [0.27145341958302016]
Epoch 8000: RMSE = 0.033865102439339885
Epoch 8100: RMSE = 0.03362220568380085
Epoch 8200: RMSE = 0.03339195656176124
Epoch 8300: RMSE = 0.033173508540793706
Epoch 8400: RMSE = 0.032965704340531816
Epoch 8500: RMSE = 0.03276771448064472
Epoch 8600: RMSE = 0.032578706319940046
Epoch 8700: RMSE = 0.03239804929232367
Epoch 8800: RMSE = 0.03222501197652723
Epoch 8900: RMSE = 0.032059067428610824
Sample [0.29] Expected [0.28595223] Produced [0.26574538566792805]
Sample [0.66] Expected [0.61311685] Produced [0.6297554365791701]
Sample [0.89] Expected [0.77707175] Produced [0.7900004145807926]
Sample [0.9] Expected [0.78332691] Produced [0.7951266776116899]
Sample [1.43] Expected [0.99010456] Produced [0.9351667085008907]
Sample [0.71] Expected [0.65183377] Produced [0.6715667625895625]
Sample [0.15] Expected [0.14943813] Produced [0.15903450331676486]
Sample [0.96] Expected [0.81919157] Produced [0.823237338876233]
Sample [1.39] Expected [0.98370081] Produced [0.9302885427956162]
Sample [1.49] Expected [0.99673775] Produced [0.9415783476023973]
Sample [0.59] Expected [0.55636102] Produced [0.565104982328469]
Sample [0.65] Expected [0.60518641] Produced [0.620827015810789]
Sample [1.24] Expected [0.945784] Produced [0.9064988137678367]
Sample [0.51] Expected [0.48817725] Produced [0.48492445039946397]
Sample [1.51] Expected [0.99815247] Produced [0.9435094028904429]
Epoch 9000: RMSE = 0.03189968515257683
Epoch 9100: RMSE = 0.03174641443323411
Epoch 9200: RMSE = 0.03159871375090271
Epoch 9300: RMSE = 0.03145636783183089
Epoch 9400: RMSE = 0.03131888278642771
Epoch 9500: RMSE = 0.031185945161159266
Epoch 9600: RMSE = 0.031057344746032632
Epoch 9700: RMSE = 0.03093271627212521
Epoch 9800: RMSE = 0.030811864087030866
Epoch 9900: RMSE = 0.03069456539293422
Sample [0.89] Expected [0.77707175] Produced [0.7914295251898394]
Sample [1.39] Expected [0.98370081] Produced [0.9330212968864408]
Sample [1.49] Expected [0.99673775] Produced [0.9442479096665707]
Sample [0.15] Expected [0.14943813] Produced [0.156908389246318]
Sample [1.24] Expected [0.945784] Produced [0.9092506319045309]
Sample [0.29] Expected [0.28595223] Produced [0.2628947779154661]
Sample [0.51] Expected [0.48817725] Produced [0.4826767358107992]
Sample [0.59] Expected [0.55636102] Produced [0.563599776307486]
Sample [0.9] Expected [0.78332691] Produced [0.7967701604494919]
Sample [0.96] Expected [0.81919157] Produced [0.8252725949538092]
Sample [1.43] Expected [0.99010456] Produced [0.9379055747492083]
Sample [0.71] Expected [0.65183377] Produced [0.6713964121369079]
Sample [1.51] Expected [0.99815247] Produced [0.9461701870424215]
Sample [0.66] Expected [0.61311685] Produced [0.6289027368319564]
Sample [0.65] Expected [0.60518641] Produced [0.6199003043652676]
Epoch 10000: RMSE = 0.030580513915159042
Sample [0.] Expected [0.], Produced [0.08357152839705025]
Sample [0.01] Expected [0.00999983], Produced [0.08730161194651304]
Sample [0.02] Expected [0.01999867], Produced [0.0911811772805994]
Sample [0.03] Expected [0.0299955], Produced [0.09521424559132491]
Sample [0.04] Expected [0.03998933], Produced [0.09940476567993092]
Sample [0.05] Expected [0.04997917], Produced [0.10375659514255088]
Sample [0.06] Expected [0.05996401], Produced [0.1082734805199376]
Sample [0.07] Expected [0.06994285], Produced [0.11295903646353979]
Sample [0.08] Expected [0.07991469], Produced [0.11781672398550944]
Sample [0.09] Expected [0.08987855], Produced [0.12284982787642386]
Sample [0.1] Expected [0.09983342], Produced [0.12806143339146892]
Sample [0.11] Expected [0.1097783], Produced [0.13345440232337039]
Sample [0.12] Expected [0.11971221], Produced [0.13903134859825536]
Sample [0.13] Expected [0.12963414], Produced [0.14479461354862125]
Sample [0.14] Expected [0.13954311], Produced [0.1507462410353854]
Sample [0.16] Expected [0.15931821], Produced [0.16322112291007126]
Sample [0.17] Expected [0.16918235], Produced [0.16974675554579818]
Sample [0.18] Expected [0.17902957], Produced [0.17646545965044502]
Sample [0.19] Expected [0.18885889], Produced [0.18337742740132504]
Sample [0.2] Expected [0.19866933], Produced [0.1904824127290647]
Sample [0.21] Expected [0.2084599], Produced [0.19777971148758672]
Sample [0.22] Expected [0.21822962], Produced [0.2052681433459235]
Sample [0.23] Expected [0.22797752], Produced [0.2129460356636371]
Sample [0.24] Expected [0.23770263], Produced [0.22081120960659498]
Sample [0.25] Expected [0.24740396], Produced [0.22886096875059075]
Sample [0.26] Expected [0.25708055], Produced [0.2370920904066696]
Sample [0.27] Expected [0.26673144], Produced [0.24550081988390096]
Sample [0.28] Expected [0.27635565], Produced [0.2540828678827756]
Sample [0.3] Expected [0.29552021], Produced [0.27174709677834724]
Sample [0.31] Expected [0.30505864], Produced [0.2808180495067023]
Sample [0.32] Expected [0.31456656], Produced [0.2900398833247356]
Sample [0.33] Expected [0.32404303], Produced [0.29940571616145034]
Sample [0.34] Expected [0.33348709], Produced [0.3089081883812737]
Sample [0.35] Expected [0.34289781], Produced [0.31853948477326177]
Sample [0.36] Expected [0.35227423], Produced [0.32829135995797665]
Sample [0.37] Expected [0.36161543], Produced [0.3381551670564708]
Sample [0.38] Expected [0.37092047], Produced [0.34812188942236016]
Sample [0.39] Expected [0.38018842], Produced [0.35818217519659923]
Sample [0.4] Expected [0.38941834], Produced [0.3683263744061322]
Sample [0.41] Expected [0.39860933], Produced [0.3785445782929099]
Sample [0.42] Expected [0.40776045], Produced [0.3888266605295695]
Sample [0.43] Expected [0.4168708], Produced [0.39916231995301266]
Sample [0.44] Expected [0.42593947], Produced [0.409541124427727]
Sample [0.45] Expected [0.43496553], Produced [0.41995255543735743]
Sample [0.46] Expected [0.44394811], Produced [0.43038605299600075]
Sample [0.47] Expected [0.45288629], Produced [0.44083106047008463]
Sample [0.48] Expected [0.46177918], Produced [0.4512770689074161]
Sample [0.49] Expected [0.47062589], Produced [0.46171366048187923]
Sample [0.5] Expected [0.47942554], Produced [0.4721305506799532]
Sample [0.52] Expected [0.49688014], Produced [0.49286499698906044]
Sample [0.53] Expected [0.50553334], Produced [0.503163005882756]
Sample [0.54] Expected [0.51413599], Produced [0.5134022893310786]
Sample [0.55] Expected [0.52268723], Produced [0.5235737952531027]
Sample [0.56] Expected [0.5311862], Produced [0.5336688140850979]
Sample [0.57] Expected [0.53963205], Produced [0.5436790041359071]
Sample [0.58] Expected [0.54802394], Produced [0.5535964138299582]
Sample [0.6] Expected [0.56464247], Produced [0.5731231476694715]
Sample [0.61] Expected [0.57286746], Produced [0.5827186749637103]
Sample [0.62] Expected [0.58103516], Produced [0.592193850498117]
Sample [0.63] Expected [0.58914476], Produced [0.6015428960293115]
Sample [0.64] Expected [0.59719544], Produced [0.6107604908581105]
Sample [0.67] Expected [0.62098599], Produced [0.6375782264727267]
Sample [0.68] Expected [0.62879302], Produced [0.6462259373421896]
Sample [0.69] Expected [0.63653718], Produced [0.6547223977426593]
Sample [0.7] Expected [0.64421769], Produced [0.6630649637748071]
Sample [0.72] Expected [0.65938467], Produced [0.679279895964649]
Sample [0.73] Expected [0.66686964], Produced [0.6871489910439291]
Sample [0.74] Expected [0.67428791], Produced [0.6948576188028465]
Sample [0.75] Expected [0.68163876], Produced [0.7024050600714596]
Sample [0.76] Expected [0.68892145], Produced [0.7097909315181306]
Sample [0.77] Expected [0.69613524], Produced [0.7170151676970042]
Sample [0.78] Expected [0.70327942], Produced [0.7240780027627995]
Sample [0.79] Expected [0.71035327], Produced [0.7309799520056254]
Sample [0.8] Expected [0.71735609], Produced [0.737721793346076]
Sample [0.81] Expected [0.72428717], Produced [0.744304548918081]
Sample [0.82] Expected [0.73114583], Produced [0.7507294668540893]
Sample [0.83] Expected [0.73793137], Produced [0.7569980033743672]
Sample [0.84] Expected [0.74464312], Produced [0.7631118052696397]
Sample [0.85] Expected [0.75128041], Produced [0.7690726928541534]
Sample [0.86] Expected [0.75784256], Produced [0.774882643454599]
Sample [0.87] Expected [0.76432894], Produced [0.7805437754892981]
Sample [0.88] Expected [0.77073888], Produced [0.7860583331816983]
Sample [0.91] Expected [0.78950374], Produced [0.8017465874201471]
Sample [0.92] Expected [0.79560162], Produced [0.8066993091517888]
Sample [0.93] Expected [0.80161994], Produced [0.8115180777270141]
Sample [0.94] Expected [0.8075581], Produced [0.8162056099867219]
Sample [0.95] Expected [0.8134155], Produced [0.8207646612010423]
Sample [0.97] Expected [0.82488571], Produced [0.8295084766743481]
Sample [0.98] Expected [0.83049737], Produced [0.833698860440824]
Sample [0.99] Expected [0.83602598], Produced [0.8377719862953404]
Sample [1.] Expected [0.84147098], Produced [0.8417306707856457]
Sample [1.01] Expected [0.84683184], Produced [0.8455777210652213]
Sample [1.02] Expected [0.85210802], Produced [0.8493159290730178]
Sample [1.03] Expected [0.85729899], Produced [0.852948066258743]
Sample [1.04] Expected [0.86240423], Produced [0.8564768788259124]
Sample [1.05] Expected [0.86742323], Produced [0.8599050834646278]
Sample [1.06] Expected [0.87235548], Produced [0.8632353635460901]
Sample [1.07] Expected [0.8772005], Produced [0.8664703657510885]
Sample [1.08] Expected [0.88195781], Produced [0.8696126971051548]
Sample [1.09] Expected [0.88662691], Produced [0.8726649223936582]
Sample [1.1] Expected [0.89120736], Produced [0.8756295619308443]
Sample [1.11] Expected [0.89569869], Produced [0.8785090896576483]
Sample [1.12] Expected [0.90010044], Produced [0.8813059315440349]
Sample [1.13] Expected [0.90441219], Produced [0.884022464272586]
Sample [1.14] Expected [0.9086335], Produced [0.8866610141810848]
Sample [1.15] Expected [0.91276394], Produced [0.8892238564429047]
Sample [1.16] Expected [0.91680311], Produced [0.8917132144650748]
Sample [1.17] Expected [0.9207506], Produced [0.8941312594849762]
Sample [1.18] Expected [0.92460601], Produced [0.8964801103476999]
Sample [1.19] Expected [0.92836897], Produced [0.8987618334471482]
Sample [1.2] Expected [0.93203909], Produced [0.9009784428150125]
Sample [1.21] Expected [0.935616], Produced [0.9031319003427689]
Sample [1.22] Expected [0.93909936], Produced [0.9052241161228206]
Sample [1.23] Expected [0.9424888], Produced [0.9072569488958664]
Sample [1.25] Expected [0.94898462], Produced [0.9111516469578399]
Sample [1.26] Expected [0.95209034], Produced [0.9130169782490994]
Sample [1.27] Expected [0.95510086], Produced [0.9148298599962581]
Sample [1.28] Expected [0.95801586], Produced [0.9165919038174227]
Sample [1.29] Expected [0.96083506], Produced [0.9183046742806436]
Sample [1.3] Expected [0.96355819], Produced [0.9199696898048828]
Sample [1.31] Expected [0.96618495], Produced [0.9215884235934002]
Sample [1.32] Expected [0.9687151], Produced [0.9231623045934076]
Sample [1.33] Expected [0.97114838], Produced [0.9246927184764099]
Sample [1.34] Expected [0.97348454], Produced [0.9261810086341582]
Sample [1.35] Expected [0.97572336], Produced [0.9276284771856349]
Sample [1.36] Expected [0.9778646], Produced [0.9290363859909245]
Sample [1.37] Expected [0.97990806], Produced [0.930405957668253]
Sample [1.38] Expected [0.98185353], Produced [0.9317383766108555]
Sample [1.4] Expected [0.98544973], Produced [0.9342963088163903]
Sample [1.41] Expected [0.9871001], Produced [0.9355240088329667]
Sample [1.42] Expected [0.98865176], Produced [0.9367189316114313]
Sample [1.44] Expected [0.99145835], Produced [0.9390144464794674]
Sample [1.45] Expected [0.99271299], Produced [0.9401169593492644]
Sample [1.46] Expected [0.99386836], Produced [0.9411905384231679]
Sample [1.47] Expected [0.99492435], Produced [0.9422360685635435]
Sample [1.48] Expected [0.99588084], Produced [0.9432544060553751]
Sample [1.5] Expected [0.99749499], Produced [0.9452127906019123]
Sample [1.52] Expected [0.99871014], Produced [0.9470720037184513]
Sample [1.53] Expected [0.99916795], Produced [0.9479662824866548]
Sample [1.54] Expected [0.99952583], Produced [0.9488379540580768]
Sample [1.55] Expected [0.99978376], Produced [0.9496876981931897]
Sample [1.56] Expected [0.99994172], Produced [0.9505161725541886]
Sample [1.57] Expected [0.99999968], Produced [0.9513240134241274]
Final Test RMSE =  0.0307881768889552
Sample [0. 1.] Expected [1.] Produced [0.6930856650285606]
Sample [1. 0.] Expected [1.] Produced [0.6873732059072744]
Sample [1. 1.] Expected [0.] Produced [0.770416031354799]
Sample [0. 0.] Expected [0.] Produced [0.5964893265057839]
Epoch 0: RMSE = 0.5341517339201144
Epoch 100: RMSE = 0.5204991353158508
Epoch 200: RMSE = 0.5124516592864964
Epoch 300: RMSE = 0.5080192302113431
Epoch 400: RMSE = 0.5056755594149629
Epoch 500: RMSE = 0.5044727660518492
Epoch 600: RMSE = 0.5038675602427772
Epoch 700: RMSE = 0.5035675938811758
Epoch 800: RMSE = 0.5034204661090242
Epoch 900: RMSE = 0.503348388387591
Sample [1. 0.] Expected [1.] Produced [0.5121364713739336]
Sample [0. 0.] Expected [0.] Produced [0.4305991134335366]
Sample [1. 1.] Expected [0.] Produced [0.5844437655643154]
Sample [0. 1.] Expected [1.] Produced [0.5017127338098377]
Epoch 1000: RMSE = 0.5033118153310845
Epoch 1100: RMSE = 0.5032926006954491
Epoch 1200: RMSE = 0.5032821947617665
Epoch 1300: RMSE = 0.5032756623663495
Epoch 1400: RMSE = 0.5032707849214815
Epoch 1500: RMSE = 0.5032679192432298
Epoch 1600: RMSE = 0.503265008644242
Epoch 1700: RMSE = 0.5032625647470581
Epoch 1800: RMSE = 0.5032602303658636
Epoch 1900: RMSE = 0.5032588887953807
Sample [1. 0.] Expected [1.] Produced [0.5066816844286622]
Sample [0. 0.] Expected [0.] Produced [0.42696475949018353]
Sample [0. 1.] Expected [1.] Produced [0.49628870274825276]
Sample [1. 1.] Expected [0.] Produced [0.5776532677983459]
Epoch 2000: RMSE = 0.5032569510453377
Epoch 2100: RMSE = 0.503255213615092
Epoch 2200: RMSE = 0.5032542601330892
Epoch 2300: RMSE = 0.5032530345675309
Epoch 2400: RMSE = 0.5032518981353054
Epoch 2500: RMSE = 0.5032508229171373
Epoch 2600: RMSE = 0.5032499907895063
Epoch 2700: RMSE = 0.5032490277667135
Epoch 2800: RMSE = 0.5032483880277627
Epoch 2900: RMSE = 0.5032477532833112
Sample [0. 0.] Expected [0.] Produced [0.42711418789083516]
Sample [0. 1.] Expected [1.] Produced [0.4953813541610076]
Sample [1. 1.] Expected [0.] Produced [0.5763259283260511]
Sample [1. 0.] Expected [1.] Produced [0.5062259427281096]
Epoch 3000: RMSE = 0.5032471814430893
Epoch 3100: RMSE = 0.503246494173387
Epoch 3200: RMSE = 0.5032458228588135
Epoch 3300: RMSE = 0.5032456090209819
Epoch 3400: RMSE = 0.503245007351583
Epoch 3500: RMSE = 0.5032447204783205
Epoch 3600: RMSE = 0.5032441136740763
Epoch 3700: RMSE = 0.5032439077211162
Epoch 3800: RMSE = 0.5032434421568209
Epoch 3900: RMSE = 0.5032432259532624
Sample [1. 1.] Expected [0.] Produced [0.5756704012577651]
Sample [0. 1.] Expected [1.] Produced [0.49465292757316126]
Sample [0. 0.] Expected [0.] Produced [0.4273711513853133]
Sample [1. 0.] Expected [1.] Produced [0.5064460499149603]
Epoch 4000: RMSE = 0.5032429028748461
Epoch 4100: RMSE = 0.5032424718483732
Epoch 4200: RMSE = 0.5032422987006494
Epoch 4300: RMSE = 0.5032418588247279
Epoch 4400: RMSE = 0.5032417047340906
Epoch 4500: RMSE = 0.5032413291210935
Epoch 4600: RMSE = 0.5032409762303929
Epoch 4700: RMSE = 0.5032408265586655
Epoch 4800: RMSE = 0.5032405488061557
Epoch 4900: RMSE = 0.5032396631022489
Sample [0. 0.] Expected [0.] Produced [0.42771534652161053]
Sample [1. 0.] Expected [1.] Produced [0.5069367273781775]
Sample [0. 1.] Expected [1.] Produced [0.4951930106457016]
Sample [1. 1.] Expected [0.] Produced [0.5762978333040005]
Epoch 5000: RMSE = 0.503239778244546
Epoch 5100: RMSE = 0.5032395812859166
Epoch 5200: RMSE = 0.5032392888396053
Epoch 5300: RMSE = 0.5032389340761916
Epoch 5400: RMSE = 0.5032385494075707
Epoch 5500: RMSE = 0.5032385675497014
Epoch 5600: RMSE = 0.5032382347577323
Epoch 5700: RMSE = 0.5032376703010333
Epoch 5800: RMSE = 0.5032371017510001
Epoch 5900: RMSE = 0.5032372997991925
Sample [0. 0.] Expected [0.] Produced [0.4278489332086934]
Sample [1. 0.] Expected [1.] Produced [0.5073180919937962]
Sample [1. 1.] Expected [0.] Produced [0.5754317474201295]
Sample [0. 1.] Expected [1.] Produced [0.4939585278997478]
Epoch 6000: RMSE = 0.5032369818358008
Epoch 6100: RMSE = 0.503236063331421
Epoch 6200: RMSE = 0.5032362046787194
Epoch 6300: RMSE = 0.5032359621861161
Epoch 6400: RMSE = 0.5032354327248377
Epoch 6500: RMSE = 0.5032352493411532
Epoch 6600: RMSE = 0.5032346020782231
Epoch 6700: RMSE = 0.5032342925942344
Epoch 6800: RMSE = 0.5032339877104496
Epoch 6900: RMSE = 0.5032335637034301
Sample [1. 1.] Expected [0.] Produced [0.5751246500564404]
Sample [1. 0.] Expected [1.] Produced [0.5074240111874149]
Sample [0. 1.] Expected [1.] Produced [0.49386355717732716]
Sample [0. 0.] Expected [0.] Produced [0.4282539764684222]
Epoch 7000: RMSE = 0.5032333044849684
Epoch 7100: RMSE = 0.5032327428053743
Epoch 7200: RMSE = 0.5032321395055782
Epoch 7300: RMSE = 0.5032314534264317
Epoch 7400: RMSE = 0.5032310113765583
Epoch 7500: RMSE = 0.5032305537853242
Epoch 7600: RMSE = 0.5032306099506008
Epoch 7700: RMSE = 0.5032300883032081
Epoch 7800: RMSE = 0.5032294792416014
Epoch 7900: RMSE = 0.5032285759397929
Sample [0. 0.] Expected [0.] Produced [0.42800103777008563]
Sample [1. 1.] Expected [0.] Produced [0.5745858203899332]
Sample [0. 1.] Expected [1.] Produced [0.49234788683269]
Sample [1. 0.] Expected [1.] Produced [0.5081554605713773]
Epoch 8000: RMSE = 0.5032284452151011
Epoch 8100: RMSE = 0.5032280229984656
Epoch 8200: RMSE = 0.503227472530339
Epoch 8300: RMSE = 0.5032268436345875
Epoch 8400: RMSE = 0.503225804572739
Epoch 8500: RMSE = 0.5032255535224273
Epoch 8600: RMSE = 0.5032246265316727
Epoch 8700: RMSE = 0.5032244312340796
Epoch 8800: RMSE = 0.5032238717084828
Epoch 8900: RMSE = 0.5032229142478267
Sample [0. 1.] Expected [1.] Produced [0.4929662915203513]
Sample [0. 0.] Expected [0.] Produced [0.42855944760189063]
Sample [1. 0.] Expected [1.] Produced [0.5094652684654758]
Sample [1. 1.] Expected [0.] Produced [0.5758122173793988]
Epoch 9000: RMSE = 0.503222220839981
Epoch 9100: RMSE = 0.5032217182558794
Epoch 9200: RMSE = 0.5032205097585745
Epoch 9300: RMSE = 0.5032200949234616
Epoch 9400: RMSE = 0.5032193940586707
Epoch 9500: RMSE = 0.5032188923345887
Epoch 9600: RMSE = 0.5032180041136224
Epoch 9700: RMSE = 0.5032167500953962
Epoch 9800: RMSE = 0.5032162602086067
Epoch 9900: RMSE = 0.5032151229559724
Sample [1. 0.] Expected [1.] Produced [0.5098753641638398]
Sample [0. 1.] Expected [1.] Produced [0.4927590255530032]
Sample [1. 1.] Expected [0.] Produced [0.5760866422228905]
Sample [0. 0.] Expected [0.] Produced [0.42837744585285314]
Epoch 10000: RMSE = 0.5032143231950295
"""




