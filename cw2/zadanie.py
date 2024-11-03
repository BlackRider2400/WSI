from cec2017.functions import f2, f13
import random
import concurrent.futures
import matplotlib as plt

DOWN_RANGE = -100
UP_RANGE = 100
ITERATIONS = 50000

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
