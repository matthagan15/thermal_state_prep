import json
from os import walk
from os.path import isfile, join
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

filepath = "/Users/matt/repos/thermal_state_prep/numerics/cluster_data/single_shot_harmonic/results.json"

def get_fixed_params(inputs):
    alphas, beta_envs, beta_syss, times = set(), set(), set(), set()
    for (alpha, beta_env, beta_sys, time) in inputs:
        alphas.add(alpha)
        beta_envs.add(beta_env)
        beta_syss.add(beta_sys)
        times.add(time)
    print("alphas")
    print(alphas)
    print("beta_envs")
    print(beta_envs)
    print("beta_syss")
    print(beta_syss)
    print("times")
    print(times)
    

with open(filepath) as f:
    j = json.load(f)
    inputs = j["inputs"]
    outputs = j["outputs"]
    num_samples = j["num_samples"]
    dim_sys = j["dim_sys"]
    label = j["label"]
    if len(inputs) != len(outputs):
        raise Exception("inputs and outputs are not the same length.")
    print("len of inputs: ", len(inputs))
    fixed_params = get_fixed_params(inputs)
    variable = "beta_sys"

    if variable == "beta_sys":
        # The format of this is [(alpha, beta_env, time)]
        n = 240000 - 1
        fixed_params = [(inputs[n][0], inputs[n][1], inputs[n][3])]
        fig = plt.figure()
        colors = ['r', 'b', 'g', 'black']
        color_count = 0
        for (alpha_0, beta_env_0, time_0) in fixed_params:
            print("alpha, beta_env, time", alpha_0, beta_env_0, time_0)
            x = []
            y = []
            yerr = []
            for ix in range(len(inputs)):
                (alpha, beta_env, beta_sys, time) = inputs[ix]
                (original_dist, mean_dist, std_dist, dist_of_mean) = outputs[ix]
                if alpha == alpha_0 and beta_env == beta_env_0 and time == time_0:
                    x.append(beta_sys)
                    y.append(mean_dist)
                    yerr.append(std_dist)

                    print("beta_sys, mean_dist = ", beta_sys, ", ", mean_dist)
            plt.errorbar(x, y, yerr, marker='x', linestyle='none', label="a={:.3f}, b_e={:.2f}, t={:.2f}".format(alpha_0, beta_env_0, time_0), color=colors[color_count])
            color_count += 1
        plt.legend()
        plt.show()
        
    