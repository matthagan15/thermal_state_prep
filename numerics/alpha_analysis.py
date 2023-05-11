#!/usr/bin/env python
import json
from os import walk
from os.path import isfile, join
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

def get_experiments():
    base = input("what base directory to use: ")
    for root, runs, files in walk(base):
        return (base, runs)

def analyze_run(directory):
    files = []
    for _, _, f in walk(directory):
        files = f
    x_vals = set()
    means = {}
    errors = {}
    for file in files:
        try:
            # print('file:', file)
            # print('parsed: ', int(file))
            seed = int(file)
            path = join(directory, file)
            print('path:', path)
            with open(path) as f:
                j = json.load(f)
                for (k,v) in j.items():
                    x = int(k)
                    x_vals.add(x)
                    if x in means:
                        means[x].insert((v[0], v[1]))
                    else:
                        means[x] = [(v[0], v[1])]
                    if x in errors:
                        errors[x].insert((v[0], v[2]))
                    else:
                        errors[x] = [(v[0], v[2])]
        except Exception as e:
            pass
            # print("not a data file:", file)
            # print('exception: ', e)
    processed = {}
    for x in x_vals:
        tmp = means[x]
        avg = 0.
        std = 0.
        tot_samples = 0
        for u,v in tmp:
            tot_samples += u
        for u, v in tmp:
            avg += float(u) * v / float(tot_samples)
        for n, s in errors[x]:
            std += s * np.sqrt(float(n) / tot_samples)
        processed[x] = (avg, std)
    print('processed: ', processed)

if __name__ == '__main__':
    x,y = get_experiments()
    for dir in y:
        analyze_run(join(x, dir))

def all_in_one():
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
