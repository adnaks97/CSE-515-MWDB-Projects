import numpy as np
import os
import pandas as pd
import pickle as pkl
from scipy.stats import norm


class Task1:
    def __init__(self, dir, r, w, s):
        self.data_dir = os.path.abspath(dir)
        self.r = r
        self.w = w
        self.s = s
        self.files = sorted([f for f in os.listdir(self.data_dir) if ".csv" in f])
        self.execute()

    def execute(self):
        for f in self.files:
            data = pd.read_csv(os.path.join(self.data_dir, f), header=None)
            data = self.normalize(data)
            lengths = self.get_lengths()
            data = self.quantize(data, lengths)
            all_features = self.windowing(data, f)
            pkl.dump(all_features, open(os.path.join(self.data_dir, "task1", f.split(".")[0] + ".wrd"), "wb"))

    def normalize(self, df, lower=-1, upper=1):
        df = df.T
        df = (df - df.min()) * (upper - lower) / (df.max() - df.min()) + lower
        return df.T

    def get_lengths(self, mean=0, std=0.25):
        x = np.linspace(-1, 1, 200)
        gaussian = norm(mean, std).pdf(x)
        band_points = np.concatenate((np.linspace(-1, 0, self.r+1), np.linspace(0, 1, self.r+1)[1:]))
        lengths = [0]
        for i in range(len(band_points)-1):
            nx = x[(x >= band_points[i]) & (x < band_points[i+1])]
            ny = norm(mean, std).pdf(nx)
            lengths.append(2*ny.sum()/gaussian.sum())
        return lengths

    def quantize(self, data, lengths):
        lengths = np.array(lengths)
        bins = -1 + np.cumsum(np.array(lengths))
        bins[-1] = 1
        mid_points = np.round(np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]), decimals=4)
        for i in range(len(data)):
            ft = data.iloc[i, :]
            ft_digitized = np.digitize(ft, bins) - 1
            ft_digitized[ft_digitized == len(mid_points)] = len(mid_points)-1
            ft_quantized = mid_points[ft_digitized]
            data.iloc[i, :] = ft_quantized
        return data

    def windowing(self, df, f):
        all_feats = []
        for i in range(df.shape[0]):
            for j in range(0, df.shape[1], self.s):
                idx = (int(f.split(".")[0]), i, j)
                win = df.values[i, j:j+self.w].tolist()
                all_feats.append((idx, win))
        return all_feats


if __name__ == "__main__":
    dir = "Z"
    r = 3
    w = 3
    s = 3
    task1 = Task1(dir, r, w, s)