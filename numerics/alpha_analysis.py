#!/usr/bin/env python
import json
import matplotlib.pyplot as plt

base_dir = "/Users/matt/scratch/tsp/"
datasets = ["alpha_0_005/1", "alpha_0_01/correct_data", "alpha_0_05/1", "alpha_0_10/1"]
alphas = [0.005, 0.01, 0.05, 0.1]
interactions = range(1,101)
means = {}
stds = {}
for ix in range(len(datasets)):
    with open(base_dir + datasets[ix], 'r') as f:
        json_string = f.read()
        x = json.loads(json_string)
        mean_data = []
        std_data = []
        for k,v in x.items():
            mean_data.append((int(k), v[1]))
            std_data.append((int(k), v[2]))
        means[alphas[ix]] = mean_data
        stds[alphas[ix]] = std_data
fig = plt.figure()
colors = ['r', 'b', 'g', 'black']
for ix in range(len(alphas)):
    alpha = alphas[ix]
    print("alpha = ", alpha)
    x, y = zip(*means[alpha])
    _, y_error = zip(*stds[alpha])
    plt.errorbar(x, y, y_error, marker='x', linestyle='none', label="{:.3f}".format(alpha), color=colors[ix])
plt.legend(loc = 'upper left')
plt.title("Error vs interaction for 20d Harmonic Osc. w/ 1-qubit bath")
plt.xlabel("Interaction Number")
plt.ylabel("Schatten-2 error")
plt.show()
plt.savefig(base_dir + "first_results.png")
