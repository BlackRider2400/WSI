import numpy as np


def brute_force(list_mass, list_items_val, max_mass):
    items = []
    for i in range(len(list_mass)):
        items.append((list_mass[i], list_items_val[i]))

    all_options = []
    for i in range(len(list_mass)**2):
        binary_format = bin(i)[2:]
        binary_format = binary_format.rjust(len(list_mass), "0")
        all_options.append(binary_format)

    max_val = -1
    best_mass = -1
    best_option = ""

    for i in all_options:
        sum_of_values = 0
        mass = 0
        for j in range(len(i)):
            if i[j] == "1":
                sum_of_values += int(list_items_val[j])
                mass += int(list_mass[j])
                if mass > max_mass:
                    break

        if mass <= max_mass and sum_of_values > max_val:
            max_val = sum_of_values
            best_mass = mass
            best_option = i

    out_put = []

    for i in range(len(list_mass)):
        if best_option[i] == "1":
            out_put.append((list_mass[i], list_items_val[i]))

    return out_put


def heuristics(list_mass, list_items_val, max_mass):
    items = []
    for i in range(len(list_mass)):
        items.append((list_mass[i], list_items_val[i]))

    items.sort(key=lambda x: (x[1] / x[0]), reverse=True)

    mass = 0

    out_put = []

    for i in items:
        if i[0] + mass <= max_mass:
            out_put.append(i)
            mass += i[0]


    return out_put


if __name__ == '__main__':
    m = [8, 3, 5, 2]
    p = [16, 8, 9, 6]
    M = np.sum(m) / 2

    print(brute_force(m, p, M))
    print(heuristics(m, p, M))

