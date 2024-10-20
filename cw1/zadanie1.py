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

    print(f"Finished calculating for {len(list_mass)}.")


if __name__ == '__main__':
    max_int = 50
    m = [8, 3, 5, 2]
    p = [16, 8, 9, 6]

    with open("zadanie1.txt", "w") as file:
        file.write("Zadanie1:\n")

    test_data(m, p)

    m = []
    p = []

    for i in range(10):
        m.append(random.randint(1, max_int))
        p.append(random.randint(1, max_int))

    test_data(m, p)

    m = []
    p = []

    for i in range(15):
        m.append(random.randint(1, max_int))
        p.append(random.randint(1, max_int))

    test_data(m, p)

    m = []
    p = []

    for i in range(20):
        m.append(random.randint(1, max_int))
        p.append(random.randint(1, max_int))

    test_data(m, p)

    m = []
    p = []

    for i in range(21):
        m.append(random.randint(1, max_int))
        p.append(random.randint(1, max_int))

    test_data(m, p)

    m = []
    p = []

    for i in range(22):
        m.append(random.randint(1, max_int))
        p.append(random.randint(1, max_int))

    test_data(m, p)

    m = []
    p = []

    for i in range(23):
        m.append(random.randint(1, max_int))
        p.append(random.randint(1, max_int))

    test_data(m, p)

    m = []
    p = []

    for i in range(24):
        m.append(random.randint(1, max_int))
        p.append(random.randint(1, max_int))

    test_data(m, p)

    m = []
    p = []

    for i in range(25):
        m.append(random.randint(1, max_int))
        p.append(random.randint(1, max_int))

    test_data(m, p)

