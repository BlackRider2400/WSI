import numpy as np


# Definicja przedmiotów
m = np.array([8, 3, 5, 2])  # masa przedmiotów
p = np.array([16, 8, 9, 6])  # wartość przedmiotów
M = np.sum(m) / 2  # maksymalna masa plecaka

def brute_force():
    all_options = []
    for i in range(len(m)):
        binary_format = bin(i)[2:]
        all_options.append(binary_format)

    max_val = -1
    best_mass = -1
    best_option = ""

    for i in all_options:
        sum = 0
        mass = 0
        for j in range(len(i)):
            if i[j] == "1":
                sum += int(p[j])
                mass += int(m[i])
                if mass > M:
                    break

        if mass <= M and sum > max_val:
            max_val = sum
            best_mass = mass
            best_option = i

brute_force()

