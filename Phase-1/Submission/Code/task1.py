import numpy as np
import os
import pandas as pd
import pickle as pkl
from scipy.stats import norm
from pathlib import Path


class Task1:
    """
    Class to perform task1
    """
    def __init__(self, dir, r, w, s):
        """
        Initializes and runs the task1 by taking the requred parameters
        :param dir: the directory where we can find csv files
        :param r: the value for number of bands
        :param w: the size of each window
        :param s: the shift size for each window
        """
        self.data_dir = os.path.abspath(dir)
        self.out_dir = os.path.join(str(Path(self.data_dir).parent), "SampleOutputs", "task1")
        self.r = r
        self.w = w
        self.s = s
        self.files = sorted([f for f in os.listdir(self.data_dir) if ".csv" in f])
        self.execute()

    def execute(self):
        """
        This is the main function that runs the task1 by calling all the steps to be performed for each file
        :return:
        """
        # loop through each file
        for f in self.files:
            # open the csv
            data = pd.read_csv(os.path.join(self.data_dir, f), header=None)
            # normalize the data
            data = self.normalize(data)
            # calculate the lengths of bands using gaussian
            lengths = self.get_lengths()
            # quantize the data
            data = self.quantize(data, lengths)
            # convert the data into windows/words
            all_features = self.windowing(data, f)
            # save the words in new files
            pkl.dump(all_features, open(os.path.join(self.data_dir, f.split(".")[0] + ".wrd"), "wb"))
            pkl.dump(all_features, open(os.path.join(self.out_dir, f.split(".")[0] + ".wrd"), "wb"))

    def normalize(self, df, lower=-1, upper=1):
        """
        Performs normalization taking new range as input
        :param df: dataframe
        :param lower: new min
        :param upper: new max
        :return: dataframe normalized
        """
        df = df.T
        df = (df - df.min()) * (upper - lower) / (df.max() - df.min()) + lower
        return df.T

    def get_lengths(self, mean=0, std=0.25):
        """
        Calculate length of each band
        :param mean: mean of gaussian dist to use
        :param std: std of gaussian dist to use
        :return:
        """
        # create some random points in range -1,1
        x = np.linspace(-1, 1, 2000)
        # create a gaussian dist
        gaussian = norm(mean, std).pdf(x)
        # find points on x axis where gaussian gets divided into bands
        band_points = np.concatenate((np.linspace(-1, 0, self.r+1), np.linspace(0, 1, self.r+1)[1:]))
        lengths = [0]
        # calculate length of each band
        for i in range(len(band_points)-1):
            nx = x[(x >= band_points[i]) & (x < band_points[i+1])]
            ny = norm(mean, std).pdf(nx)
            lengths.append(2*ny.sum()/gaussian.sum())
        return lengths

    def quantize(self, data, lengths):
        """
        The function quantizes the data
        :param data: dataframe
        :param lengths: lengths of each band
        :return: quantized dataframe
        """
        lengths = np.array(lengths)
        # find bins
        bins = -1 + np.cumsum(np.array(lengths))
        bins[-1] = 1
        # find midpoint of each bin
        mid_points = np.round(np.array([(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]), decimals=4)
        # discretize the data
        for i in range(len(data)):
            ft = data.iloc[i, :]
            # bin the continuos valued data
            ft_digitized = np.digitize(ft, bins) - 1
            # replace last bin with one value less due to np.digitize internal computation flaw
            ft_digitized[ft_digitized == len(mid_points)] = len(mid_points)-1
            # replace bins values with the respective mid points
            ft_quantized = mid_points[ft_digitized]
            data.iloc[i, :] = ft_quantized
        return data

    def windowing(self, df, f):
        """
        Function that applies winowing technique to each row in the dataframe
        :param df: quantized dataframe
        :param f: filename
        :return:
        """
        all_feats = []
        # go through each row
        for i in range(df.shape[0]):
            # slide windows of size w by stride s
            for j in range(0, df.shape[1], self.s):
                idx = (int(f.split(".")[0]), i, j)
                win = df.values[i, j:j+self.w].tolist()
                all_feats.append((idx, win))
        return all_feats


if __name__ == "__main__":
    dir = input("Enter the directory : ")
    r = int(input("Enter the value for r : "))
    w = int(input("Enter the value for w : "))
    s = int(input("Enter the value for s : "))
    task1 = Task1(dir, r, w, s)
