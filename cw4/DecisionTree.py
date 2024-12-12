import math
from collections import Counter


class Node:

        def __init__(self, attribute=None, label=None):

            self.attribute = attribute
            self.label = label
            self.children = {}
            self.is_leaf = label is not None

        def add_child(self, value, child_node):
            self.children[value] = child_node

        def predict(self, sample):

            if self.is_leaf:
                return self.label

            value = sample.get(self.attribute)

            if value in self.children:
                return self.children[value].predict(sample)

            return self.label

class DecisionTree:

    def __init__(self):
        self.root = None

    def train_tree(self, data, labels, attributes):
        self.root = self._build_tree(data, labels, attributes)

    def _build_tree(self, data, labels, available_attributes):

        if len(set(labels)) == 1:
            return Node(label=labels[0])

        if not available_attributes:
            most_common_label = Counter(labels).most_common(1)[0][0]
            return Node(label=most_common_label)

        best_attribute = self._find_best_attribute(data, labels, available_attributes)
        root = Node(attribute=best_attribute)

        attribute_values = set(sample[best_attribute] for sample in data)

        for value in attribute_values:

            filtered_data = [
                sample for sample in data if sample[best_attribute] == value
            ]

            filtered_labels = [
                label for sample, label in zip(data, labels) if sample[best_attribute] == value
            ]

            if not filtered_data:
                most_common_label = Counter(labels).most_common(1)[0][0]
                root.add_child(value, Node(label=most_common_label))
            else:
                remaining_attributes = [
                    attr for attr in available_attributes if attr != best_attribute
                ]
                root.add_child(value, self._build_tree(filtered_data, filtered_labels, remaining_attributes))

        return root

    def predict(self, sample):
        return self.root.predict(sample)

    def _find_best_attribute(self, data, labels, attributes):

        best_attribute = None
        highest_gain = -float('inf')

        for attribute in attributes:
            gain = self._calculate_information_gain(data, labels, attribute)
            if gain > highest_gain:
                highest_gain = gain
                best_attribute = attribute

        return best_attribute

    def _calculate_information_gain(self, data, labels, attribute):

        total_entropy = self._calc_entropy(labels)
        attribute_values = set(sample[attribute] for sample in data)

        weighted_entropy = 0

        for value in attribute_values:
            subset_labels = [
                label for sample, label in zip(data, labels) if sample[attribute] == value
            ]

            weighted_entropy += (len(subset_labels) / len(labels)) * self._calc_entropy(subset_labels)

        return total_entropy - weighted_entropy

    @staticmethod
    def _calc_entropy(labels):

        label_counts = Counter(labels)
        entropy = 0

        for count in label_counts.values():
            probability = count / len(labels)
            entropy -= probability * math.log2(probability)

        return entropy