import numpy as np
import time
import random


def brute_force(list_mass, list_items_val, max_mass):
    items = []
    for i in range(len(list_mass)):
        items.append((list_mass[i], list_items_val[i]))

    max_val = -1
    best_mass = -1
    best_option = ""

    for i in range(2**(len(list_mass))):
        binary_format = bin(i)[2:]
        binary_format = binary_format.rjust(len(list_mass), "0")

        sum_of_values = 0
        mass = 0
        for j in range(len(binary_format)):
            if binary_format[j] == "1":
                sum_of_values += int(list_items_val[j])
                mass += int(list_mass[j])

            if mass > max_mass:
                break

        if mass <= max_mass and sum_of_values > max_val:
            max_val = sum_of_values
            best_mass = mass
            best_option = binary_format

    out_put = []

    for i in range(len(list_mass)):
        if best_option[i] == "1":
            out_put.append((list_mass[i], list_items_val[i]))

    return {"out_put": out_put, "mass": best_mass, "value": max_val}


def heuristics(list_mass, list_items_val, max_mass):
    items = []
    for i in range(len(list_mass)):
        items.append((list_mass[i], list_items_val[i]))

    items.sort(key=lambda x: (x[1] / x[0]), reverse=True)

    mass = 0
    value = 0

    out_put = []

    for i in items:
        if i[0] + mass <= max_mass:
            out_put.append(i)
            mass += i[0]
            value += i[1]

    return {"out_put": out_put, "mass": mass, "value": value}


def test_data(list_mass, list_items_val):
    max_mass = np.sum(list_mass) / 2
    start_time_bruteforce = time.process_time()
    data_bruteforce = brute_force(list_mass, list_items_val, max_mass)
    end_time_bruteforce = time.process_time()
    start_time_heuristics = time.process_time()
    data_heuristics = heuristics(list_mass, list_items_val, max_mass)
    end_time_heuristics = time.process_time()

    time_heuristics = "{0:02f}s".format(end_time_heuristics - start_time_heuristics)
    time_bruteforce = "{0:02f}s".format(end_time_bruteforce - start_time_bruteforce)

    with open("zadanie1.csv", "a") as file:
            file.write(f"{len(list_mass)},{max_mass},{time_bruteforce},{data_bruteforce.get('mass')},{data_bruteforce.get('value')},"
                       f"{time_heuristics},{data_heuristics.get('mass')},{data_heuristics.get('value')}\n")

if __name__ == '__main__':
    max_int = 20
    m = [8, 3, 5, 2]
    p = [16, 8, 9, 6]

    with open("zadanie1.csv", "w") as file:
        file.write("items count,max mass,bruteforce time,bruteforce mass,bruteforce value,heuristics time,heuristics mass,heuristics value\n")

    test_data(m, p)

    for i in range(25):
        for j in range(5, 26):
            m = []
            p = []
            for k in range(j):
                m.append(random.randint(1, max_int))
                p.append(random.randint(1, max_int))
            test_data(m, p)
