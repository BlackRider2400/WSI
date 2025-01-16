import csv
import random
from statistics import stdev
import os

from sklearn.metrics import confusion_matrix, accuracy_score
from DecisionTree import DecisionTree
import statistics


def load_data(file_path):

    data = []
    labels = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) > 0:
                labels.append(row[-1])
                data.append({f"attr{i}" : value for i, value in enumerate(row[:-1], start=1)})

    return data, labels

def split_data(data, labels, train_ratio=0.6):

    combined = list(zip(data, labels))
    random.shuffle(combined)
    data, labels = zip(*combined)

    split_point = int(len(data) * train_ratio)
    train_data, train_labels = data[:split_point], labels[:split_point]
    test_data, test_labels = data[split_point:], labels[split_point:]

    return train_data, train_labels, test_data, test_labels

def main():
    accuracies = []
    cm_list = []
    for _ in range(25):
    #data, labels = load_data("breast+cancer/breast-cancer.data")
    #data, labels = load_data("mushroom/agaricus-lepiota.data")
    #data, labels = load_data("mushroom/agaricus-lepiota-short.data")
        os.system("python zadanie.py")
        data, labels = load_data("output.csv")
        tree = DecisionTree()
        attributes = [f"attr{i}" for i in range(1, len(data[0]) + 1)]

        train_data, train_labels, test_data, test_labels = split_data(data, labels)
        tree.train_tree(train_data, train_labels, attributes)
        predictions = [tree.predict(sample) for sample in test_data]
        predictions = [pred if pred is not None else "unknown" for pred in predictions]

        accuracy = accuracy_score(test_labels, predictions)
        accuracies.append(accuracy)

        cm = confusion_matrix(test_labels, predictions)

        cm_list.append(cm)

    average_accuracy = sum(accuracies) / len(accuracies)


    print(f"Average Accuracy: {average_accuracy * 100:.2f}%")
    print(f"Min Accuracy: {min(accuracies) * 100:.2f}%")
    print(f"Max Accuracy: {max(accuracies) * 100:.2f}%")
    print(f"StdDev Accuracy: {stdev(accuracies) * 100:.2f}%")

    for i in cm_list:
        print(i)


if __name__ == "__main__":
    main()