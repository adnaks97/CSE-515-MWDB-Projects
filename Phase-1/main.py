import os
import pandas as pd
import pickle as pkl
from task1 import normalize, get_lengths, quantize, windowing


def run_task1(data_dir, w, s, r):
    data_dir = os.path.abspath(data_dir)
    files = [f for f in os.listdir(data_dir) if ".csv" in f]
    for f in files:
        data = pd.read_csv(os.path.join(data_dir, f), header=None)
        data = normalize(data)
        lengths = get_lengths(r)
        data = quantize(data, lengths)
        all_features = windowing(data, w, s, f)
        pkl.dump(all_features,open(os.path.join(data_dir,f.split(".")[0]+".wrd"), "wb"))


if __name__ == "__main__":
    dir_name = "Z"
    w = 10
    r = 3
    s = 5
    run_task1(dir_name, w, s, r)
