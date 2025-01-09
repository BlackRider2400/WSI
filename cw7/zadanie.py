import numpy as np
import json
import csv

def load_network(file_path):
    with open(file_path, 'r') as f:
        network = json.load(f)
    return network

def sample_from_distribution(probs):
    return np.random.choice([True, False], p=[probs[0], probs[1]])

def generate_data(network, num_samples):
    data = []
    for _ in range(num_samples):
        sample = {}
        for node, config in network.items():
            if 'parents' not in config:
                if 'probs' not in config:
                    return
                sample[node] = sample_from_distribution(config['probs'])
            else:
                parent_values = tuple(bool(sample[parent]) for parent in config['parents'])
                parent_values_str = str(parent_values) if len(parent_values) > 1 else str(parent_values[0])
                if parent_values_str not in config['conditional_probs']:
                    return
                sample[node] = sample_from_distribution(config['conditional_probs'][parent_values_str])
        data.append(sample)
    return data

def save_data(data, output_file):
    if not data:
        print("There is no data to save.")
        return
    headers = list(data[0].keys())

    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)

def run_network(input_file, output_file, num_samples):
    network = load_network(input_file)
    data = generate_data(network, num_samples)
    save_data(data, output_file)

def main():
    run_network("network.json", "output.csv", 1000)

if __name__ == "__main__":
    main()
