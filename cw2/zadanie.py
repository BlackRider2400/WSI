from cec2017.functions import f2, f13
import random
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import math

DOWN_RANGE = -100
UP_RANGE = 100
ITERATIONS = 10000

def compare_children(x1, x2, function):
    if function(x1) <= function(x2):
        return True
    return False

def generate_parents(quantity):
    parents = []
    for i in range(quantity):
        parent = []
        for j in range(10):
            parent.append(random.uniform(DOWN_RANGE, UP_RANGE))
        parents.append(parent)
    return parents

def generate_children(list_of_children, function, sigma):
    children = []
    length = len(list_of_children)
    for i in range(0, length // 2):
        rand_index = random.randint(0, len(list_of_children) - 1)
        if compare_children(list_of_children[0], list_of_children[rand_index], function):
            children.append((list_of_children[0].copy()))
            children.append((list_of_children[0].copy()))
        else:
            children.append(list_of_children[rand_index].copy())
            children.append(list_of_children[rand_index].copy())

    for i in range(len(children)):
        for j in range(10):
            children[i][j] += random.uniform(-sigma, sigma)
            children[i][j] = max(DOWN_RANGE, min(UP_RANGE, children[i][j]))

    return children

def start_simulation(function, sigma, quantity):
    parents_list = generate_parents(quantity)
    children_list = parents_list
    for i in range(ITERATIONS // quantity):
        children_list = generate_children(children_list.copy(), function, sigma)

    children_list.sort(key=lambda x: (function(x)))

    with open("zadanie.csv", "a") as file:
        file.write(f"{function.__name__},{sigma}, {quantity}, {function(children_list[0])}\n")

    return {
        "function": function.__name__,
        "sigma": sigma,
        "quantity": quantity,
        "best_value": function(children_list[0])
    }

def run_simulation(params):
    return start_simulation(*params)

if __name__ == '__main__':
    with open("zadanie.csv", "w") as file:
        file.write("function, sigma, quantity, best_value\n")

    sigmas = [4, 3, 2, 1]
    quantities = [32, 64, 128, 256, 512]
    functions = [f2, f13]

    # for q in quantities:
    #     for s in sigmas:
    #         start_simulation(f2, sigma=s, quantity=q)
    tasks = [(f, s, q) for q in quantities for s in sigmas for f in functions for _ in range(25)]
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_simulation, tasks))


    data = {}

    for f in functions:
        data[f.__name__] = {}
        for q in quantities:
            data[f.__name__][q] = {}
            for s in sigmas:
                data[f.__name__][q][s] = []

    for r in results:
        data[r["function"]][r["quantity"]][r["sigma"]].append(r["best_value"])

    for func in data.keys():
        for quantity in data[func].keys():
            for sigma in data[func][quantity].keys():
                values = np.array(data[func][quantity][sigma])
                data[func][quantity][sigma] = {
                    "max": np.max(values),
                    "mean": np.mean(values),
                    "min": np.min(values),
                    "std_dev": np.std(values)
                }

    for func in data.keys():
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = []
        min_values, mean_values, max_values= [], [], []

        # Iterate over population (quantity) and sigma
        for quantity in data[func].keys():
            for sigma in data[func][quantity].keys():
                stats = data[func][quantity][sigma]
                labels.append(f"σ={sigma}, N={quantity}, std={round(stats['std_dev'], 2)}")
                min_values.append(stats['min'])
                mean_values.append(stats['mean'])
                max_values.append(stats['max'])

        # Set bar positions on the x-axis
        x = np.arange(len(labels))  # Positions for each group
        width = 0.2  # Width of each bar

        # Create bars for each statistic
        ax.bar(x - width * 1.5, min_values, width, label='Min')
        ax.bar(x - width / 2, mean_values, width, label='Mean')
        ax.bar(x + width * 1.5, max_values, width, label='Max')

        # Add labels, title, and legend
        ax.set_xlabel('Sigma and Population and stddev')
        ax.set_ylabel('Values')
        ax.set_title(f'Statistics for {func}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.show()