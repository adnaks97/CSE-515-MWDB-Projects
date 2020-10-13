import os
import json
import numpy as np
import pickle as pkl
from pathlib import Path


class Task2:
    """
    Class to perform task 2
    The class takes the directory where the wrd files are present and calculates the tfidf values
    for each word in each file and sensor
    """
    def __init__(self, dir):
        self.tf_idf2 = {}
        self.tf_idf = {}
        self.dir = os.path.abspath(dir)
        self.out_dir = os.path.join(str(Path(self.dir).parent), "SampleOutputs", "task2")
        self.file_paths = sorted([os.path.join(self.dir, f) for f in os.listdir(self.dir) if ".wrd" in f])
        self.tf = {}
        self.idf = {}
        self.idf2 = {}
        self.calculate_values()
        self.compute_products()
        pkl.dump(self.tf, open(os.path.join(self.out_dir, "tf_dict.pkl"), "wb"))
        pkl.dump(self.tf_idf, open(os.path.join(self.out_dir, "tf_idf_dict.pkl"), "wb"))
        pkl.dump(self.tf_idf2, open(os.path.join(self.out_dir, "tf_idf2_dict.pkl"), "wb"))

        pkl.dump(self.tf, open(os.path.join(self.dir, "tf_dict.pkl"), "wb"))
        pkl.dump(self.tf_idf, open(os.path.join(self.dir, "tf_idf_dict.pkl"), "wb"))
        pkl.dump(self.tf_idf2, open(os.path.join(self.dir, "tf_idf2_dict.pkl"), "wb"))
        self.convert_to_vectors()

    def calculate_values(self):
        """
        Function calculates the tf idf values of words
        """
        # numerator for idf and idf2 calculations
        total_files = 0.
        total_sensors = 20.

        # Loop through all the files in the directory one by one
        for fp in self.file_paths:
            # keep track of how many files visited to use for idf ca;culation
            total_files += 1
            # open data file stored in task 1
            data = pkl.load(open(fp, "rb"))
            # using this to kee track of new sensor data within the same file
            prev_sensor_id = data[0][0][1]
            # count the number of words in a sensor in one file
            count = 0
            for idx, win in data:
                # extract file_id, sensor_id and window vector from data
                file_id = idx[0]
                sensor_id = idx[1]
                word = tuple(win)

                # compute tf values for the sensor
                # when we start seeing data from new sensor_id, we are ready to compute the tf values for the previous sensor
                if prev_sensor_id != sensor_id:
                    for w in self.tf[file_id][prev_sensor_id]:
                        self.tf[file_id][prev_sensor_id][w] /= float(count)
                    count = 1
                else:
                    count += 1

                # calculate counts of words in each file and sensor
                # initializing the dictionary structure to help maintain counts of word occurrences
                if file_id not in self.tf.keys():
                    self.tf[file_id] = {}
                if sensor_id not in self.tf[file_id].keys():
                    self.tf[file_id][sensor_id] = {}
                if word not in self.tf[file_id][sensor_id].keys():
                    self.tf[file_id][sensor_id][word] = 0
                # after initializations, we increment the frequence counter of that word
                self.tf[file_id][sensor_id][word] += 1

                # collecting file_ids that a word in the given sensor exists in
                # initializing the dictionary structure to help maintain existence of word in a file/sensor
                if word not in self.idf.keys():
                    self.idf[word] = {}
                if sensor_id not in self.idf[word].keys():
                    self.idf[word][sensor_id] = []
                # we add the file_id to the list of file that this word appears in
                if file_id not in self.idf[word][sensor_id]:
                    self.idf[word][sensor_id].append(file_id)

                # collecting sensor_ids that a word in the given file exists in
                if word not in self.idf2.keys():
                    self.idf2[word] = {}
                if file_id not in self.idf2[word].keys():
                    self.idf2[word][file_id] = []
                # we append the sensor_id to the list of sensors that this word appears in
                if sensor_id not in self.idf2[word][file_id]:
                    self.idf2[word][file_id].append(sensor_id)

                # assign previous seen sensor to current sensor id
                prev_sensor_id = sensor_id

            for w in self.tf[file_id][sensor_id]:
                self.tf[file_id][sensor_id][w] /= float(count)

            # compute idf2 values
            # once we reach the end of file, we can compute idf2 values for all words in this file
            for w in self.idf2:
                try:
                    self.idf2[w][file_id] = np.log(total_sensors/len(self.idf2[w][file_id]))
                except:
                    pass

        # compute idf values
        # once we read all files, we can compute idf values
        for w in self.idf:
            for s in self.idf[w]:
                self.idf[w][s] = np.log(total_files/len(self.idf[w][s]))

    def compute_products(self):
        """
        Compute tf.idf and tf.idf2 values for the words
        """
        # Loop through all files, sensor and words seen in the previous function
        for file_id in self.tf:
            self.tf_idf[file_id] = {}
            self.tf_idf2[file_id] = {}
            for sensor_id in self.tf[file_id]:
                self.tf_idf[file_id][sensor_id] = {}
                self.tf_idf2[file_id][sensor_id] = {}
                for word in self.tf[file_id][sensor_id]:
                    # compute the product of tf, idf and idf2 maintaining the same structure as tf
                    self.tf_idf[file_id][sensor_id][word] = self.tf[file_id][sensor_id][word] * self.idf[word][sensor_id]
                    self.tf_idf2[file_id][sensor_id][word] = self.tf[file_id][sensor_id][word] * self.idf2[word][file_id]

    def convert_to_vectors(self):
        """
        Convert the univariate time series to vectors of tf, idf values and store them in vectors.txt
        """
        vectors = {}
        # loop through all files, sensors and words by tf, tf_idf, tf_idf2 structure and replace the word series with these new vectors
        for fp in self.file_paths:
            tf, tf_idf, tf_idf2 = [], [], []
            data = pkl.load(open(fp, "rb"))
            prev_sensor_id = 0
            for idx, win in data:
                file_id = idx[0]
                sensor_id = idx[1]
                word = tuple(win)
                if prev_sensor_id != sensor_id:
                    vectors["{}-{}".format(file_id, prev_sensor_id)] = [tf, tf_idf, tf_idf2]
                    tf, tf_idf, tf_idf2 = [], [], []
                tf.append(self.tf[file_id][sensor_id][word])
                tf_idf.append(self.tf_idf[file_id][sensor_id][word])
                tf_idf2.append(self.tf_idf2[file_id][sensor_id][word])
                prev_sensor_id = sensor_id
            vectors["{}-{}".format(file_id, sensor_id)] = [tf, tf_idf, tf_idf2]
        with open(os.path.join(self.dir, "vectors.txt"), "w") as f:
            json.dump(vectors, f)
        with open(os.path.join(self.out_dir, "vectors.txt"), "w") as f:
            json.dump(vectors, f)

if __name__ == "__main__":
    dir = input("Enter the directory : ")
    task2 = Task2(dir)
