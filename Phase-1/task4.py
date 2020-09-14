import os
import json
import numpy as np
import operator
from scipy.spatial import distance as dis

class Task4:
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        self.num_files = len([x for x in os.listdir(self.dir) if ".wrd" in x])
        self.vectors = json.load(open(os.path.join(self.dir, "vectors.txt"), "r"))
        self.tf, self.tf_idf, self.tf_idf2 = [], [], []
        self.convert_vectors_to_arrays()

    def convert_vectors_to_arrays(self):
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

    def similarity_check(self, file_id, feature=0, measure='cosine'):
        self.file_id = file_id - 1
        feature = feature
        cosine_scores, euclidean_scores = {}, {}
        file_ids = [fid for fid in range(len(self.tf)) if fid != self.file_id]
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
            self.print_results(list(cosine_scores.items())[:10])
        else:
            self.print_results(list(euclidean_scores.items())[:10])

    def print_results(self, items):
        save_string = "Most similar files found are,\n"
        print("Most similar files found are,")
        for i in range(len(items)):
            file_id = items[i][0]
            score = items[i][1]
            save_string += "{} - file id {} with a score of {} \n".format(i+1, file_id+1, np.round(score, decimals=2))
            print("{} - file id {} with a score of {}".format(i+1, file_id+1, np.round(score, decimals=2)))
        with open(os.path.join(self.dir, "task4", "{}_similarity_results.txt".format(self.file_id+1)), "w") as f:
            f.write(save_string)


if __name__ == "__main__":
    task4 = Task4("Z")
    while True:
        choice = int(input("Enter a choice : \n 1)Find similar vectors \n 2)Exit \n"))
        if choice == 2:
            break
        else:
            file_id = int(input("Enter the file id you want to check for (1-60) : "))
            feature = int(input("Enter the feature you want to use (tf - 0, tf_idf - 1, tf_idf2 - 2): "))
            measure = int(input("Choose a similarity measure to use : 1)Cosine 2)Euclidean : "))
            measure = 'cosine' if measure == 1 else 'euclidean'
            task4.similarity_check(file_id, feature, measure)
