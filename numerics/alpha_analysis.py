#!/usr/bin/env python
import json
import matplotlib.pyplot as plt

base_dir = "/scratch/n/nawiebe/hagan/tsp/error_v_interaction/"
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
            mean_data.append((k, v[1]))
            std_data.append((k, v[2]))
        means[alphas[ix]] = mean_data
        stds[alphas[ix]] = std_data
fig = plt.figure()
for alpha in alphas:
    x, means = means[alpha]
    _, stds = stds[alpha]
    plt.errorbar(x, means, stds, label="{:.3f}".format(alpha))
plt.legend(loc = 'upper left')
plt.title("Schatten-2 error vs interaction number for 20d Harmonic Osc. w/ single qubit bath")
plt.xlabel("interaction number")
plt.ylabel("Schatten-2 error")
plt.savefig("/scratch/n/nawiebe/hagan/tsp/error_v_interaction/first_results.png")
