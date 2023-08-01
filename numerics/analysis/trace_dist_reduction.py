import json
from os import walk
from os.path import isfile, join
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

def load_data():
    filename = "/Users/matt/repos/thermal_state_prep/numerics/tmp/single_qbit_out.json"
    with open(filename) as f:
        json_data =json.load(f)
        print("keys:")
        print(json_data.keys())
        alphas = set()
        betas = set()
        gammas = set()
        means = {}
        stds = {}
        data_list = json_data["data"]
        for pt in data_list:
            a = pt["alpha"]
            b = pt["beta_sys"]
            c = pt["env_gap"]
            alphas.add(a)
            betas.add(b)
            gammas.add(c)
            means[(a, b, c)] = pt["mean"]
            stds[(a, b, c)] = pt["std"]
        return (alphas, betas, gammas, means, stds)

def plot_gap(alpha, beta, gammas, means, stds):
    """assumes means and stds are dictionaries that map (a,b,c) -> mean. """
    x = list(gammas)
    y = []
    yerr = []
    for i in x:
        y.append(means[(alpha, beta, i)])
        yerr.append(stds[(alpha, beta, i)])
    fig = plt.figure()
    # plt.yscale('log', nonposy='clip')
    plt.errorbar(x, y, yerr, label="alpha = {:.4f}\nbeta_sys = {:.4f}".format(alpha, beta), fmt="r+")
    plt.legend(loc = 'lower right')
    plt.title("Trace distance after 1 interaction, env_beta = 0.5, system gap = 1.0")
    plt.xlabel("Environment Gap")
    plt.ylabel("Normalized trace distance ideal to output")
    plt.show()

def plot_alpha(alphas, beta, gamma, means, stds):
    """assumes means and stds are dictionaries that map (a,b,c) -> mean. """
    x = list(alphas)
    y = []
    yerr = []
    for i in x:
        y.append(means[(i, beta, gamma)])
        yerr.append(stds[(i, beta, gamma)])
    fig = plt.figure()
    # plt.yscale('log', nonposy='clip')
    plt.xscale('log')
    plt.errorbar(x, y, yerr, label="env_gap = {:.4f}\nbeta_sys = {:.4f}".format(gamma, beta), fmt="r+")
    plt.legend(loc = 'upper right')
    plt.title("Trace distance after 1 interaction, env_beta = 0.5, system gap = 1.0")
    plt.xlabel("alpha")
    plt.ylabel("Normalized trace distance ideal to output")
    plt.show()

if __name__ == "__main__":
    (alphas, betas, gammas, means, stds) = load_data()
    print('alphas:\n')
    print(alphas)
    print('betas:\n')
    print(betas)
    print('gammas:\n')
    print(gammas)
    # print('means:\n')
    # print(means)
    # plot_gap(min(alphas), min(betas), gammas, means, stds)
    # plot_gap(0.0031622776601683794, min(betas), gammas, means, stds)
    plot_gap(min(alphas), min(betas), gammas, means, stds)
    plot_alpha(alphas, min(betas), 1.0015384615384615, means, stds)
