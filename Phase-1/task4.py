import os
import json
import numpy as np
import operator
import pickle as pkl
from pathlib import Path
from scipy.spatial import distance as dis

class Task4:
    """
    Class to perform task 4
    This class takes the directory, file_id, feature and measure of distance to use from user and calculates the 10
    closest files to the query file in the directory
    """
    def __init__(self, dir):
        self.file_vectors = {}
        self.sensor_words = {k:[] for k in range(20)}
        self.words_rev_dict = {}
        self.words_dict = {}
        self.dir = os.path.abspath(dir)
        self.tf_dict = pkl.load(open(os.path.join(self.dir,"tf_dict.pkl"),"rb"))
        self.tf_idf_dict = pkl.load(open(os.path.join(self.dir, "tf_idf_dict.pkl"), "rb"))
        self.tf_idf2_dict = pkl.load(open(os.path.join(self.dir, "tf_idf2_dict.pkl"), "rb"))
        self.out_dir = os.path.join(str(Path(self.dir).parent), "Outputs", "task4")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.files = [os.path.join(self.dir,x) for x in os.listdir(self.dir) if ".wrd" in x]
        self.num_files = len(self.files)
        self.vectors = json.load(open(os.path.join(self.dir, "vectors.txt"), "r"))
        self.tf, self.tf_idf, self.tf_idf2 = [], [], []
        self.convert_vectors_to_arrays()
        self.get_unique_words_across_files()
        self.convert_file_to_vectors()

    def get_unique_words_across_files(self):
        """
        Finding all unqiue words across all sensors across all files
        :return:
        """
        idx = 0
        for f in self.tf_dict:
            for s in self.tf_dict[f]:
                self.sensor_words[s].extend([x for x in list(self.tf_dict[f][s].keys()) if x not in self.sensor_words[s]])

        for s in self.sensor_words:
            for w in self.sensor_words[s]:
                self.words_dict["{}_{}".format(s, w)] = idx
                self.words_rev_dict[idx] = "{}_{}".format(s, w)
                idx += 1

    def convert_file_to_vectors(self):
        """
        Convert files to vectors of size total_unique_words that can be used for comparison or similarity finding
        :return:
        """
        # go through every file
        for f in self.files:
            fid = int(f.split("/")[-1].split(".")[0])
            self.file_vectors[fid] = {}
            self.file_vectors[fid]["tf"] = np.zeros(len(self.words_dict.keys()))
            self.file_vectors[fid]["tf_idf"] = np.zeros(len(self.words_dict.keys()))
            self.file_vectors[fid]["tf_idf2"] = np.zeros(len(self.words_dict.keys()))
            self.file_vectors[fid]["counts"] = np.zeros(len(self.words_dict.keys()))
            # load the wrd file
            data = pkl.load(open(f, "rb"))
            # go through every word
            for idx, win in data:
                file_id = idx[0]
                sensor_id = idx[1]
                w = tuple(win)
                # update the vectors with tf, tfidf, tfidf2 values
                self.file_vectors[file_id]["counts"][self.words_dict["{}_{}".format(sensor_id,w)]] += 1
                self.file_vectors[file_id]["tf"][self.words_dict["{}_{}".format(sensor_id, w)]] = \
                    self.tf_dict[file_id][sensor_id][w]
                self.file_vectors[file_id]["tf_idf"][self.words_dict["{}_{}".format(sensor_id, w)]] = \
                    self.tf_idf_dict[file_id][sensor_id][w]
                self.file_vectors[file_id]["tf_idf2"][self.words_dict["{}_{}".format(sensor_id, w)]] = \
                    self.tf_idf2_dict[file_id][sensor_id][w] + 1e-7

    def convert_vectors_to_arrays(self):
        """
        THis function converts the sequential file vectors into arrays of shape (20,n) that can be used for sequential similarity testing
        :return:
        """
        prev_file_id = 1
        tf, tf_idf, tf_idf2 = [], [], []
        for key in self.vectors:
            file_id, sensor_id = int(key.split("-")[0]), int(key.split("-")[1])
            if prev_file_id != file_id:
                self.tf.append(np.array(tf).reshape((20, -1)))
                self.tf_idf.append(np.array(tf_idf).reshape((20, -1)))
                self.tf_idf2.append(np.array(tf_idf2).reshape((20, -1)))
                tf, tf_idf, tf_idf2 = [], [], []

            tf.append(self.vectors[key][0])
            tf_idf.append(self.vectors[key][1])
            tf_idf2.append(self.vectors[key][2])
            prev_file_id = file_id

        self.tf.append(np.array(tf).reshape((20, -1)))
        self.tf_idf.append(np.array(tf_idf).reshape((20, -1)))
        self.tf_idf2.append(np.array(tf_idf2).reshape((20, -1)))

    def similarity_score(self, v1, v2):
        """
        THis function is used to find similarity value for 2 arrays of different sizes
        :param v1: vector 1
        :param v2: vector 2
        :return: cosine and euclidean distance across all 20 sensors
        """
        max_l = max(v1.shape[1], v2.shape[1])
        cosine_total_dist = 0
        euclidean_total_dist = 0
        for i in range(len(v1)):
            a = v1[i, :]
            b = v2[i, :]
            a_pad = np.zeros((max_l - len(a), ))
            b_pad = np.zeros((max_l - len(b), ))
            a = np.concatenate((a, a_pad), axis=0)
            b = np.concatenate((b, b_pad), axis=0)
            cosine_total_dist += dis.cosine(a, b)
            euclidean_total_dist += dis.euclidean(a, b)
        return cosine_total_dist, euclidean_total_dist

    def find_similar_files(self, file_id, feature=0, measure="cosine"):
        """
        This function is used to find the similar files to the query file id using a particular similarity measrue and feature
        :param file_id: query file_id
        :param feature: tf, tfidf or tfidf2 to use
        :param measure: cosine or euclidean
        :return: none
        """
        self.file_id = file_id
        self.feature = feature
        self.measure = measure
        if self.feature == 0:
            type = "counts"
        elif self.feature == 1:
            type = "tf"
        elif self.feature == 2:
            type = "tf_idf"
        elif self.feature == 3:
            type = "tf_idf2"
        cosine_scores, euclidean_scores = {}, {}
        # go through every file in the directory
        for fid in self.file_vectors:
            # Find all files other than the query file
            if fid != self.file_id:
                cs = dis.cosine(self.file_vectors[fid][type], self.file_vectors[self.file_id][type])
                es = dis.euclidean(self.file_vectors[fid][type], self.file_vectors[self.file_id][type])
                cosine_scores[fid] = cs
                euclidean_scores[fid] = es
        cosine_scores = dict(sorted(cosine_scores.items(), key=operator.itemgetter(1), reverse=True))
        euclidean_scores = dict(sorted(euclidean_scores.items(), key=operator.itemgetter(1)))

        if measure == 'cosine':
            self.print_results(list(cosine_scores.items())[:10], "vector")
        else:
            self.print_results(list(euclidean_scores.items())[:10], "vector")

    def similarity_check(self, file_id, feature=0, measure='cosine'):
        """
        This function is used to find the similar files to the query file id using a particular similarity measrue and feature for sequential vectors
        :param file_id: query file_id
        :param feature: tf, tfidf or tfidf2 to use
        :param measure: cosine or euclidean
        :return: none
        """
        self.file_id = file_id - 1
        self.feature = feature
        self.measure = measure
        cosine_scores, euclidean_scores = {}, {}
        file_ids = [fid for fid in range(len(self.tf)) if fid != self.file_id]
        # for each feature find the most similar files by sequential simialrity test function
        if feature == 0:
            for fid in file_ids:
                cs, es = self.similarity_score(self.tf[self.file_id], self.tf[fid])
                cosine_scores[fid] = cs
                euclidean_scores[fid] = es
        elif feature == 1:
            for fid in file_ids:
                cs, es = self.similarity_score(self.tf_idf[self.file_id], self.tf_idf[fid])
                cosine_scores[fid] = cs
                euclidean_scores[fid] = es
        else:
            for fid in file_ids:
                cs, es = self.similarity_score(self.tf_idf2[self.file_id], self.tf_idf2[fid])
                cosine_scores[fid] = cs
                euclidean_scores[fid] = es

        cosine_scores = dict(sorted(cosine_scores.items(), key=operator.itemgetter(1)))
        euclidean_scores = dict(sorted(euclidean_scores.items(), key=operator.itemgetter(1)))

        if measure == 'cosine':
            self.print_results(list(cosine_scores.items())[:10], "sequence")
        else:
            self.print_results(list(euclidean_scores.items())[:10], "sequence")

    def print_results(self, items, vector):
        """
        THis function prints the result and stores the output into files
        :param items: ranked files and scores
        :param vector: is it vector or sequential analysis
        :return:
        """
        save_string = "Most similar files found are,\n"
        print("Most similar files found are,")
        for i in range(len(items)):
            file_id = items[i][0]
            score = items[i][1]
            save_string += "& {} & {} \n".format(file_id, np.round(score, decimals=2))
            print("{} - file id {} with a score of {}".format(i+1, file_id, np.round(score, decimals=2)))
        # with open(os.path.join(self.dir, "task4", "{}_similarity_results.txt".format(self.file_id+1)), "w") as f:
        #     f.write(save_string)
        with open(os.path.join(self.out_dir, "{}_{}_{}_{}_similarity_results.txt".format(self.file_id, self.feature, self.measure, vector)), "w") as f:
            f.write(save_string)


if __name__ == "__main__":
    dir = input("Enter the directory : ")
    task4 = Task4(dir)
    while True:
        choice = int(input("Enter a choice : \n 1)Find similar vectors \n 2)Exit \n"))
        if choice == 2:
            break
        else:
            file_ids = [61, 62,63, 64, 65, 66]
            features = [1,2,3]
            measures = ['cosine','euclidean']
            for f in file_ids:
                for fe in features:
                    for m in measures:
                        task4.find_similar_files(f, fe, m)
            # file_id = int(input("Enter the file id you want to check for (1-60) : "))
            # feature = int(input("Enter the feature you want to use (counts - 0, tf - 1, tf_idf - 2, tf_idf2 - 3): "))
            # measure = int(input("Choose a similarity measure to use : 1)Cosine 2)Euclidean : "))
            # measure = 'cosine' if measure == 1 else 'euclidean'
            # # to call sequential similarity use this
            # # task4.similarity_check(file_id, feature, measure)
            # # to call simple vector similarity use this
            # task4.find_similar_files(file_id, feature, measure)
