from DecisionTree import DecisionTree

data = [
    {"color": "red", "shape": "round"},
    {"color": "green", "shape": "oval"},
    {"color": "red", "shape": "oval"},
    {"color": "green", "shape": "round"},
]
labels = ["apple", "pear", "apple", "pear"]
attributes = ["color", "shape"]

training_data = data[:3]
training_labels = labels[:3]
testing_data = data[3:]
testing_labels = labels[3:]

tree = DecisionTree()
tree.train_tree(training_data, training_labels, attributes)

for sample, true_label in zip(testing_data, testing_labels):
    prediction = tree.predict(sample)
    print(f"Sample: {sample}, True Label: {true_label}, Prediction: {prediction}")
