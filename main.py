"""
Build a NNData class that will help us better manage our training and
testing data.
"""
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
import numpy as np
import random
from math import floor
import math


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


class MultiLinkNode(ABC):
    """
    Abstract Class that will be the starting point for the FFBPNeurode class.
    """
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
        """ Print ID of current node and neighbors."""
        neighbors_upstream = [id(x) for x in
                              self._neighbors[MultiLinkNode.Side.UPSTREAM]]
        neighbors_downstream = [id(x) for x in
                                self._neighbors[MultiLinkNode.Side.DOWNSTREAM]]
        return f"Current Node ID:{id(self)}\n" \
               f"Upstream Neighbor ID's: {neighbors_upstream}\n" \
               f"Downstream Neighbor ID's: {neighbors_downstream}"

    @abstractmethod
    def _process_new_neighbor(self, node, side: Enum):
        """
        Method that will be created in child class Neurode.
        :param node:
        :param side:
        :return:
        """
        pass

    def reset_neighbors(self, nodes: list, side: Enum):
        """
        Reset (or set) the nodes that link into this node either upstream or
        downstream.
        :param nodes:
        :param side:
        :return:
        """
        if side == MultiLinkNode.Side.UPSTREAM:

            # Copy nodes into Upstream
            nodes_copy = nodes.copy()
            self._neighbors[MultiLinkNode.Side.UPSTREAM].extend(nodes_copy)

            # Call Process New Neighbor method
            for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
                self._process_new_neighbor(node=node,
                                           side=MultiLinkNode.Side.UPSTREAM)

            # Calculate Reference Value
            self._reference_value[MultiLinkNode.Side.UPSTREAM] = 2 ** len(
                self._neighbors[MultiLinkNode.Side.UPSTREAM]) - 1

        elif side == MultiLinkNode.Side.DOWNSTREAM:

            # Copy nodes into Downstream
            nodes_copy = nodes.copy()
            self._neighbors[MultiLinkNode.Side.DOWNSTREAM].extend(nodes_copy)

            # Call Process New Neighbor method
            for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._process_new_neighbor(node=node,
                                           side=MultiLinkNode.Side.DOWNSTREAM)

                # Calculate Reference Value
                self._reference_value[MultiLinkNode.Side.DOWNSTREAM] = 2 ** len(
                    self._neighbors[MultiLinkNode.Side.DOWNSTREAM]) - 1


class Neurode(MultiLinkNode):

    def __init__(self, node_type: LayerType, learning_rate: float = .05):
        """
        This class is inherited from class MultiLinkNode. Allows us associate nodes
        with neighboring nodes and check them check-in after reporting they have
        data.
        :param node_type:
        :param learning_rate:
        """
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}
        super(Neurode, self).__init__()

    @property
    def value(self):
        """
        Return current value of neurode.
        :return:
        """
        return self._value

    @property
    def node_type(self):
        """Get node type. This is one of the LayerType elements"""
        return self._node_type

    @property
    def learning_rate(self):
        """Get learning rate."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, new_learning_rate: float):
        """
        Set learning rate parameter.
        :param new_learning_rate:
        :return:
        """
        self._learning_rate = new_learning_rate

    def _process_new_neighbor(self, node: 'Neurode', side: Enum):
        """
        Process new node neighbors.
        :param node:
        :param side:
        :return:
        """
        if side == MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.uniform(0, 1)
        else:
            pass

    def _check_in(self, node: 'Neurode', side: MultiLinkNode.Side):
        """
        Method called when neighboring nodes check-in with data.
        :param node:
        :param side:
        :return:
        """

        # Up-stream/down-stream Sides
        up_stream = MultiLinkNode.Side.UPSTREAM
        down_stream = MultiLinkNode.Side.DOWNSTREAM

        # Check in process for up-stream nodes
        if side == MultiLinkNode.Side.UPSTREAM:
            node_index = self._neighbors[up_stream].index(node)
            new_report = 2 ** node_index
            self._reporting_nodes[MultiLinkNode.Side.UPSTREAM] = \
                self._reporting_nodes[up_stream] | new_report
            upstream_reference_value = self._reference_value[
                up_stream]

            # Validate if all up-stream nodes have checked in
            if upstream_reference_value == self._reporting_nodes[up_stream]:
                self._reporting_nodes[up_stream] = 0
                return True
            else:
                return False

        # Check-in process for down-stream nodes
        elif side == down_stream:
            node_index = self._neighbors[down_stream].index(node)
            new_report = 2 ** node_index
            self._reporting_nodes[down_stream] = \
                self._reporting_nodes[down_stream] | new_report
            downstream_reference_value = self._reference_value[down_stream]

            # Validate if all nodes have checked in
            if downstream_reference_value == self._reporting_nodes[down_stream]:
                self._reporting_nodes[down_stream] = 0
                return True
            else:
                return False

    def get_weight(self, node: 'Neurode'):
        """
        Look up node in weights dictionary to find associated weight.
        :param node:
        :return:
        """
        return self._weights[node]


class FFNeurode(Neurode):
    """
    Calculates weighted values from upstream nodes which are then bound to a
    range 0-1 through the sigmoid function. This weighted sum value is then
    communicated to all downstream neighboring nodes.
    """
    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + math.exp(-value))

    def _calculate_value(self):
        """
        Calculate the weighted sum of upstream nodes' values. Pass this result
        to the method 'sigmoid' and store returned value into _value attribute.
        :return:
        """

        # Calculate weighted values of upstream nodes
        weighted_sum_list = []
        for k, v in self._weights.items():
            weighted_value = k.value*v
            weighted_sum_list.append(weighted_value)

        # Sum list of weighted values
        weighted_sum = sum(weighted_sum_list)

        # Set _value attribute
        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self):
        """
        Runs the method 'data_ready_upstream' for each neighboring downstream
        node.
        :return:
        """

        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """
        Upstream Nodes will call this method once they have data ready.
        :param node:
        :return:
        """

        # If node has data call methods 'calculate value' and 'fire downstream'
        if self._check_in(node, side=MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value):
        """
        Directly set the value of an input layer neurode.
        :param input_value:
        :return:
        """

        self._value = input_value
        self._fire_downstream()


class BPNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self):
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        return value*(1 - value)

    def _calculate_delta(self, expected_value=None):
        """
        Calculate the delta between the node value and the expected Value. If
        node type is of 'Output' then calculate 1st way. Else
        calculate another way. TESTING.
        :param expected_value:
        :return:
        """

        # Calculate delta for Output nodes
        if self._node_type == LayerType.OUTPUT:
            self._delta = (expected_value -
                           self.value)*self._sigmoid_derivative(self.value)

        # Calculate delta for hidden/input nodes
        else:
            weighted_downstream_deltas = []

            # Loop downstream nodes to get weighted downstream delta
            for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                weighted_delta = node.get_weight(self)*node.delta
                weighted_downstream_deltas.append(weighted_delta)

            # Sum weighted deltas and set self._delta to value
            weighted_sum_downstream_deltas = sum(weighted_downstream_deltas)
            hidden_delta = weighted_sum_downstream_deltas*self._sigmoid_derivative(self.value)
            self._delta = hidden_delta

    def data_ready_downstream(self, node):
        """
        Downstream Nodes will call this method once they have data ready.
        :param node:
        :return:
        """

        # If node has data call methods '_calculate_delta' and '_fire_upstream'
        if self._check_in(node, side=MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """
        Directly set the value of an output layer neurode.
        :param expected_value:
        :return:
        """
        self._calculate_delta(expected_value=expected_value)

        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(node=self)

    def adjust_weights(self, node, adjustment):
        """
        Called by an upstream node, using node reference to add adjustment to
        appropriate entry of self._weights.
        :param node:
        :param adjustment:
        :return:
        """
        self._weights[node] = adjustment

    def _update_weights(self):
        """
        Iterate through its downstream neighbors, and use adjust_weights to
        request an adjustment to the weight given to this node's data.
        :return:
        """

        # Loop through downstream neighbors and use method 'adjust_weights'
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            value_upstream_node = self.value
            downstream_delta = node.delta
            downstream_learning_rate = node.learning_rate
            weight = node.get_weight(self)
            adjustment = weight + value_upstream_node*downstream_delta*downstream_learning_rate
            node.adjust_weights(node=self, adjustment=adjustment)

    def _fire_upstream(self):
        """
        Runs the method 'data_ready_upstream' for each neighboring downstream
        node.
        :return:
        """

        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:

    class EmptyListError(Exception):
        """
        This custom exception is raised if our list is empty.
        """

        def __init__(self, message):
            self.message = message

    def __init__(self):
        self._head = None
        self._curr = None
        self._prev = None

    def move_forward(self):
        if self._curr is None:
            return None
        else:
            self._curr = self._curr.next

        if self._curr is None:
            return None
        else:
            return self._curr.data

    def move_back(self):
        pass

    def reset_to_head(self):
        """
        Reset node to head node.
        :return:
        """
        self._curr = self._head
        if self._curr is None:
            return None
        else:
            return self._curr.data

    def reset_to_tail(self):
        pass

    def add_to_head(self, data):
        """
        insert item at the beginning of the doubly linked list.
        :param data:
        :return:
        """
        new_node = Node(data)
        new_node.next = self._head
        self._head = new_node
        self.reset_to_head()

    def add_after_cur(self, data):
        if self._curr is None:
            self.add_to_head(data)
            return
        new_node = Node(data)
        new_node.next = self._curr.next
        self._curr.next = new_node

    def remove_from_head(self):
        if self._head is None:
            return None
        ret_val = self._head.data
        self._head = self._head.next
        self.reset_to_head()
        return ret_val

    def remove_after_cur(self):
        if self._curr is None or self._curr.next is None:
            return None
        ret_val = self._curr.next.data
        self._curr = self._curr.next.next
        return ret_val

    def get_current_data(self):
        if self._head is None:
            raise DoublyLinkedList.EmptyListError("This is an empty list.")
        else:
            if self._curr is None:
                raise DoublyLinkedList.EmptyListError("This is an empty list.")
            else:
                return self._curr.data


class LayerList:
    pass


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


def dll_test():
    my_list = DoublyLinkedList()
    try:
        my_list.get_current_data()
    except DoublyLinkedList.EmptyListError:
        print("Pass")
    else:
        print("Fail 1")


    for a in range(3):
        my_list.add_to_head(a)
    if my_list.get_current_data() != 2:
        print("Error")

    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail 2")
    my_list.move_forward()


    try:
        my_list.move_forward()
    except IndexError:
        print("Pass")
    else:
        print("Fail 3")
    if my_list.get_current_data() != 0:
        print("Fail 4")
    my_list.move_back()
    my_list.remove_after_cur()
    if my_list.get_current_data() != 1:
        print("Fail 5")
    my_list.move_back()
    if my_list.get_current_data() != 2:
        print("Fail")
    try:
        my_list.move_back()
    except IndexError:
        print("Pass")
    else:
        print("Fail 6")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail 7")


def testing():
    my_list = DoublyLinkedList()
    for a in range(3):
        print(a)
        my_list.add_to_head(a)
    print(my_list.get_current_data())
    my_list.move_forward()
    print(my_list.get_current_data())
    print(my_list.get_current_data())
    print(my_list.get_current_data())


if __name__ == '__main__':
    dll_test()

