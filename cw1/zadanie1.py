import numpy as np
import time
import random


def brute_force(list_mass, list_items_val, max_mass):
    items = []
    for i in range(len(list_mass)):
        items.append((list_mass[i], list_items_val[i]))

    all_options = []
    for i in range(2**(len(list_mass))):
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

    return out_put, best_mass, max_val


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


    return out_put, mass, value


def test_data(list_mass, list_items_val):
    items = []
    for i in range(len(list_mass)):
        items.append((list_mass[i], list_items_val[i]))
    M = np.sum(m) / 2
    with open("zadanie1.txt", "a") as file:
        file.write(f"Test for {len(m)} elements (max mass: {M}):\n")
        file.write(str(items) + "\n")
        start_time_bruteforce = time.process_time()
        data = brute_force(list_mass, list_items_val, M)
        end_time_bruteforce = time.process_time()
        file.write(f"Bruteforce, mass: {data[1]} value: {data[2]}\n")
        file.write(str(data[0]))
        file.write("\n")
        start_time_heuristics = time.process_time()
        data = heuristics(list_mass, list_items_val, M)
        end_time_heuristics = time.process_time()
        file.write(f"Heuristics, mass: {data[1]} value: {data[2]}\n")
        file.write(str(data[0]))
        file.write("\n--------------------------------------\n")
        file.write("Bruteforce time: " + "{0:02f}s".format(end_time_bruteforce - start_time_bruteforce))
        file.write("\n")
        file.write("Heuristics time: " + "{0:02f}s".format(end_time_heuristics - start_time_heuristics))
        file.write("\n--------------------------------------\n")



if __name__ == '__main__':
    m = [8, 3, 5, 2]
    p = [16, 8, 9, 6]

    with open("zadanie1.txt", "w") as file:
        file.write("Zadanie1:\n")

    test_data(m, p)

    m = []
    p = []

    for i in range(100):
        m.append(random.randint(1, 30))
        p.append(random.randint(1, 30))

    test_data(m, p)

