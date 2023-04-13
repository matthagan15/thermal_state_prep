#!/usr/bin/env python

base_dir = "/scratch/n/nawiebe/hagan/tsp/error_v_interaction/"
datasets = ["alpha_0_005/1", "alpha_0_01/correct_data", "alpha_0_05/1", "alpha_0_10/1"]
for data_file in datasets:
    with open(base_dir + data_file, 'r') as f:
        json_string = f.read()
        x = json.loads(json_string)
        print("file: ", base_dir + data_file)
        for k,v in x.items():
            print("k = ", k)
            print("v = ", v)
