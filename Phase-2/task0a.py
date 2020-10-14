import argparse
import glob
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler


class wordExtractor:
    def __init__(self, fname, bands=4, window_size=10, stride=3, DIR='Data/Z/'):
        """
        init the wordExtractor class
        :param fname - name of gesture file
        :param bands - r value i.e resolution
        :param window_size - size of word window
        :param stride - shift value s
        :DIR - dir to store and read outputs/inputs

        """
        self.r = bands
        self.w = window_size
        self.s = stride
        self.fname = fname
        self.nfname = "0"+self.fname if len(self.fname) == 1 else self.fname
        self.DIR = DIR


    def read_file(self):
        """
        read df values

        :returns values array
        """
        # print(self.fname)
        df = pd.read_csv(self.DIR + '{}.csv'.format(self.fname), header=None)
        values = df.values
        return values


    def normalize(self):
        """
        Normalize values present in time series data between (-1, 1)

        :returns - normalized_values
        """
        values = self.read_file()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_values = scaler.fit_transform(values.T).T
        normalized_values[normalized_values < -1.0] = -1.0
        normalized_values[normalized_values > 1.0] = 1.0
        return normalized_values


    def get_intervals(self):
        """
        Divide (-1, 1) into r equally spaced intervals

        :returns - intervals - array of intervals
        """
        intervals_left = np.linspace(-1, 0, self.r+1)[:-1]
        intervals_right = np.linspace(0, 1, self.r+1)
        intervals = np.concatenate((intervals_left, intervals_right))
        return intervals


    def compute_lengths(self, intervals):
        """
        Compute length of gaussian bands

        :param intervals - equally spaced intervals in (-1, 1)

        :returns - lengths of each band
        """
        lengths = []
        normal_dist = norm(0.0, 0.25)
        for i in range(intervals.shape[0] - 1):
            band_sum = 2 * (normal_dist.cdf(intervals[i+1]) - normal_dist.cdf(intervals[i]))
            lengths.append(band_sum)
        return np.array(lengths)


    def get_quantized_labels(self, lengths):
        """
        Get representative label as midpoint of each band

        :param lengths - lenghths of gaussian bands

        :returns:
            quantized_labels - repr points
            levels - New interval points as per lengths
        """
        levels = -1 + np.cumsum(lengths)
        levels = np.append(-1, levels)
        levels[-1] = 1

        quantized_labels = []
        for i in range(levels.shape[0] - 1):
            label = (levels[i] + levels[i+1])/2
            quantized_labels.append(label)
        quantized_labels = np.array(quantized_labels)
        return quantized_labels, levels


    def quantize(self, normalized_values, levels, quantized_labels):
        """
        Quantize data into the bands

        :param normalized_values - normalized data
        :param levels - band intervals
        :param quantized_labels - repr labels

        :returns quantized_data
        """
        inds_quantized = np.digitize(normalized_values, levels, right=True) - 1
        inds_quantized[inds_quantized < 0] = 0
        quantized_data = quantized_labels[inds_quantized]
        return quantized_data


    def window(self, quantized_data):
        """
        Break a time series into a window of words using s, w

        :param quantized_data

        :returns words - list of gesture words for a file
        """
        words = []
        for sensorId in range(quantized_data.shape[0]):
            t = 0
            max_t_index = quantized_data[sensorId].shape[0]
            while t < max_t_index:
                idx = (self.nfname, sensorId, t)
                end = t+self.w
                if end >= quantized_data.shape[1]:
                    end = quantized_data.shape[1]
                win = quantized_data[sensorId, t:end]
                words.append((idx, win))
                t += self.s
        return words


    def save_word_file(self, words):
        """
        save words as .wrd file
        """
        out_file = self.DIR + '{}.wrd'
        with open(out_file.format(self.fname), 'wb') as handle:
            pickle.dump(words, handle)


    def main(self):
        """
        Workflow of task-1 calling all functions
        """
        normalized_values = self.normalize()
        intervals = self.get_intervals()
        lengths = self.compute_lengths(intervals)
        quantized_labels, levels = self.get_quantized_labels(lengths)
        quantized_data = self.quantize(normalized_values, levels, quantized_labels)
        words = self.window(quantized_data)
        self.save_word_file(words)
        print("intervals: \n", intervals)
        print("lengths: \n", lengths)
        print("levels: \n", levels)


class wordExtractor2(wordExtractor):
    def __init__(self, fname, component, bands=4, window_size=10, stride=3, DIR='Data/'):
        super().__init__(fname, bands, window_size, stride, DIR)
        self.component = component
        self.DIR = self.DIR + '/' + self.component + '/'
        Path('outputs/task0a').mkdir(parents=True, exist_ok=True)
        self.save_path = 'outputs/task0a/{}'.format(self.nfname) + '.wrd'

    def compute_mean_std(self, normalized_values):
        self.avg = normalized_values.mean(axis=1).tolist()
        self.std = normalized_values.std(axis=1).tolist()

    def save_word_file(self, words):
        """
        save words as .wrd file
        """
        avg_key = 'avg_{}'.format(self.component)
        std_key = 'std_{}'.format(self.component)
        word_key = 'words_{}'.format(self.component)

        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                word_map = json.load(f)
        else:
            word_map = dict()

        word_map[avg_key] = self.avg
        word_map[std_key] = self.std
        words_with_avg = []
        for word in words:
            window_avg = sum(word[1]) / len(word[1])  ## should this be window_size?
            words_with_avg.append((word[0], word[1].tolist(), window_avg))
        #words_with_avg.append(avg)
        #words_with_avg.append(std)
        word_map[word_key] = words_with_avg
        with open(self.save_path, 'w') as handle:
            json.dump(word_map, handle, indent=4)

    def main(self):
        """
        Workflow of task-0a calling all functions
        """
        normalized_values = self.normalize()
        self.compute_mean_std(normalized_values)
        intervals = self.get_intervals()
        lengths = self.compute_lengths(intervals)
        quantized_labels, levels = self.get_quantized_labels(lengths)
        quantized_data = self.quantize(normalized_values, levels, quantized_labels)
        words = self.window(quantized_data)
        self.save_word_file(words)
        #print("intervals: \n", intervals)
        #print("lengths: \n", lengths)
        #print("levels: \n", levels)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='script to run task-0a')
    parser.add_argument('-r', type=int, default=3, help='r value for script')
    parser.add_argument('-w', type=int, default=3, help='window size to extract words')
    parser.add_argument('-s', type=int, default=2, help='shift value')
    parser.add_argument('-dir', type=str, default=4, help='Input Directory')
    args = parser.parse_args()

    file_list = glob.glob(args.dir + '/*/*.csv')
    for csv_file in file_list:
        print("Processing: ", csv_file)
        f = csv_file.split('/')[-1].split('.')[0]
        component = csv_file.split('/')[-2]
        wordExtractor_obj = wordExtractor2(fname=f, component=component, bands=args.r, window_size=args.w, stride=args.s, DIR=args.dir)
        wordExtractor_obj.main()
