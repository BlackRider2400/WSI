from cec2017.functions import f2, f13
import random
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np

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

    sigmas = [1, 2, 4, 8]
    quantities = [16, 32, 64, 128, 256]
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
        labels = []
        min_values, mean_values, max_values, std_dev_values = [], [], [], []

        for quantity in data[func].keys():
            for sigma in data[func][quantity].keys():
                stats = data[func][quantity][sigma]
                labels.append(f"σ={sigma}, N={quantity}")
                min_values.append(stats['min'])
                mean_values.append(stats['mean'])
                max_values.append(stats['max'])
                std_dev_values.append(stats['std_dev'])

        x = np.arange(len(labels)) * 2

        bar_width = 1
        figure_size = (20, 8)

        fig_min, ax_min = plt.subplots(figsize=figure_size)
        ax_min.bar(x, min_values, width=bar_width, label='Min')
        ax_min.set_xlabel('Sigma and Population')
        ax_min.set_ylabel('Min Values')
        ax_min.set_title(f'Min Values for {func}')
        ax_min.set_xticks(x)
        ax_min.set_xticklabels(labels, rotation=45, ha='right')
        for i, v in enumerate(min_values):
            ax_min.text(x[i], v + 0.05 * max(min_values), f"{v:.2e}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        fig_min.savefig(f"{func}_min_values.png")
        plt.close(fig_min)

        fig_mean, ax_mean = plt.subplots(figsize=figure_size)
        ax_mean.bar(x, mean_values, width=bar_width, label='Mean')
        ax_mean.set_xlabel('Sigma and Population')
        ax_mean.set_ylabel('Mean Values')
        ax_mean.set_title(f'Mean Values for {func}')
        ax_mean.set_xticks(x)
        ax_mean.set_xticklabels(labels, rotation=45, ha='right')
        for i, v in enumerate(mean_values):
            ax_mean.text(x[i], v + 0.05 * max(mean_values), f"{v:.2e}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        fig_mean.savefig(f"{func}_mean_values.png")
        plt.close(fig_mean)

        fig_max, ax_max = plt.subplots(figsize=figure_size)
        ax_max.bar(x, max_values, width=bar_width, label='Max')
        ax_max.set_xlabel('Sigma and Population')
        ax_max.set_ylabel('Max Values')
        ax_max.set_title(f'Max Values for {func}')
        ax_max.set_xticks(x)
        ax_max.set_xticklabels(labels, rotation=45, ha='right')
        for i, v in enumerate(max_values):
            ax_max.text(x[i], v + 0.05 * max(max_values), f"{v:.2e}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        fig_max.savefig(f"{func}_max_values.png")
        plt.close(fig_max)

        fig_std, ax_std = plt.subplots(figsize=figure_size)
        ax_std.bar(x, std_dev_values, width=bar_width, label='Std Dev')
        ax_std.set_xlabel('Sigma and Population')
        ax_std.set_ylabel('Standard Deviation')
        ax_std.set_title(f'Standard Deviation for {func}')
        ax_std.set_xticks(x)
        ax_std.set_xticklabels(labels, rotation=45, ha='right')
        for i, v in enumerate(std_dev_values):
            ax_std.text(x[i], v + 0.05 * max(std_dev_values), f"{v:.2e}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        fig_std.savefig(f"{func}_std_dev.png")
        plt.close(fig_std)

    with open("zadanie_plot_data.csv", "w") as file:
        file.write("function, quantity, sigma, min, avg, max, std\n")

        for func in data.keys():
            for quantity in data[func].keys():
                for sigma in data[func][quantity].keys():
                    stats = data[func][quantity][sigma]
                    file.write(f"{func},{quantity},{sigma},{stats['min']},{stats['mean']},{stats['max']},{stats['std_dev']}\n")
