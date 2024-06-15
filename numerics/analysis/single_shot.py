import json
from os import walk
from os.path import isfile, join
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

filepath = "/Users/matt/repos/thermal_state_prep/numerics/data/single_shot_harmonic/results.json"


with open(filepath) as f:
    j = json.load(f)
    inputs = j["inputs"]
    outputs = j["outputs"]
    num_samples = j["num_samples"]
    dim_sys = j["dim_sys"]
    label = j["label"]
    if len(inputs) != len(outputs):
        raise Exception("inputs and outputs are not the same length.")
    
    variable = "beta_env"

    if variable == "beta_env":
        # The format of this is [(alpha, beta_env, time)]
        fixed_params = [(inputs[0][0], inputs[0][1], inputs[0][3]), (inputs[10][0], inputs[10][1], inputs[10][3])]
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
            plt.errorbar(x, y, yerr, marker='x', linestyle='none', label="a={:.2f}, b_e={:.2f}, t={:.2f}".format(alpha_0, beta_env_0, time_0), color=colors[color_count])
            color_count += 1
        plt.show()
        
    