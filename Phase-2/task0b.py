import os
import json
import numpy as np
import pickle as pkl
from pathlib import Path


class Task0b(object):
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        self.out_dir = os.path.join("outputs", "task0b")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.file_paths = sorted([os.path.join(self.dir, f) for f in os.listdir(self.dir) if ".wrd" in f])
        self.compNames = ['words_X', 'words_Y', 'words_Z', 'words_W']
        self.tf = {}
        self.idf = {}
        self.entropy = {}
        self.tf_idf = {}
        self.unique_words = {}
        self.calculate_values()
        self.compute_products()
        self.convert_to_vectors()

    def calculate_values(self):
        for c in self.compNames:
            total_files = 0.
            total_sensors = 20.
            print("processing files from component : ", c)
            # Loop through all the files in the directory one by one
            for fp in self.file_paths:
                print("Processing file : ", fp.split("/")[-1])
                # Keep track of how many files visited to use for idf calculation
                total_files += 1
                with open(fp) as f:
                    file = json.load(f)
                count = 0
                data = file[c]
                prev_sensor_id = data[0][0][1]

                for idx in data:
                    file_id = idx[0][0]
                    sensor_id = idx[0][1]
                    word = tuple(idx[1])
                    if (c,sensor_id,word) not in self.unique_words:
                        self.unique_words[(c, sensor_id, word)] = len(self.unique_words.keys())

                    # compute tf values for the sensor
                    # when we start seeing data from new sensor_id, we are ready to compute the tf values for the previous sensor
                    if prev_sensor_id != sensor_id:
                        for w in self.tf[c][file_id][prev_sensor_id]:
                            self.tf[c][file_id][prev_sensor_id][w] /= float(count)
                        count = 1
                    else:
                        count += 1

                    # calculate counts of words in each file and sensor
                    # initializing the dictionary structure to help maintain counts of word occurrences
                    if c not in self.tf.keys():
                        self.tf[c] = {}
                    if file_id not in self.tf[c].keys():
                        self.tf[c][file_id] = {}
                    if sensor_id not in self.tf[c][file_id].keys():
                        self.tf[c][file_id][sensor_id] = {}
                    if word not in self.tf[c][file_id][sensor_id].keys():
                        self.tf[c][file_id][sensor_id][word] = 0
                    # after initializations, we increment the frequence counter of that word
                    self.tf[c][file_id][sensor_id][word] += 1

                    # entropy dictionary
                    if c not in self.entropy.keys():
                        self.entropy[c] = {}
                    if file_id not in self.entropy[c].keys():
                        self.entropy[c][file_id] = {}
                    if sensor_id not in self.entropy[c][file_id].keys():
                        self.entropy[c][file_id][sensor_id] = {}
                    if word not in self.entropy[c][file_id][sensor_id].keys():
                        self.entropy[c][file_id][sensor_id][word] = 0

                    # collecting file_ids that a word in the given sensor exists in
                    # initializing the dictionary structure to help maintain existence of word in a file/sensor
                    if c not in self.idf.keys():
                        self.idf[c] = {}
                    if word not in self.idf[c].keys():
                        self.idf[c][word] = {}
                    if sensor_id not in self.idf[c][word].keys():
                        self.idf[c][word][sensor_id] = []
                    # we add the file_id to the list of file that this word appears in
                    if file_id not in self.idf[c][word][sensor_id]:
                        self.idf[c][word][sensor_id].append(file_id)
                    prev_sensor_id = sensor_id

                for w in self.tf[c][file_id][sensor_id]:
                    self.tf[c][file_id][sensor_id][w] /= float(count)
                    self.entropy[c][file_id][sensor_id][w] = self.tf[c][file_id][sensor_id][w] * np.log(
                        self.tf[c][file_id][sensor_id][w])

            # compute idf values
            # once we read all files, we can compute idf values
            for w in self.idf[c]:
                for s in self.idf[c][w]:
                    self.idf[c][w][s] = np.log(total_files / len(self.idf[c][w][s]))

    def compute_products(self):
        for c in self.tf:
            self.tf_idf[c] = {}
            for file_id in self.tf[c]:
                self.tf_idf[c][file_id] = {}
                for sensor_id in self.tf[c][file_id]:
                    self.tf_idf[c][file_id][sensor_id] = {}
                    for word in self.tf[c][file_id][sensor_id]:
                        self.tf_idf[c][file_id][sensor_id][word] = self.tf[c][file_id][sensor_id][word] * \
                                                                   self.idf[c][word][sensor_id]

    def convert_to_vectors(self):
        self.unique_words = dict(sorted(self.unique_words.items(), key=lambda x: (x[0][0],x[0][1])))
        for fp in self.file_paths:
            file_id = fp.split(".")[0].split("/")[-1]
            tf_vector, tfidf_vector = [], []
            for comb in self.unique_words:
                c = comb[0]
                sensor_id = comb[1]
                word = comb[2]
                if word in self.tf[c][file_id][sensor_id]:
                    tf_vector.append(self.tf[c][file_id][sensor_id][word])
                    tfidf_vector.append(self.tf_idf[c][file_id][sensor_id][word])
                else:
                    tf_vector.append(0)
                    tfidf_vector.append(0)

            json.dump(json.dumps(tf_vector), open(os.path.join(self.out_dir,"tf_vectors_{}.txt".format(file_id)), "w"))
            # pkl.dump(tf_vector, )
            json.dump(json.dumps(tfidf_vector), open(os.path.join(self.out_dir,"tfidf_vectors_{}.txt".format(file_id)), "w"))

        pkl.dump(self.unique_words, open(os.path.join(self.out_dir,"all_words_idx.txt"), "wb"))

if __name__ == "__main__":
    dir = input("Enter the directory to use: ")
    tob = Task0b(dir)
