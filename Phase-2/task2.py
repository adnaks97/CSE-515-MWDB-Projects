import os
import json
import numpy as np
import pickle as pkl
from pathlib import Path
import multiprocessing as mpl
from multiprocessing import Pool
from sklearn.metrics import pairwise_distances


class Task2:
    def __init__(self, model, dir):
        self.model = model
        self.dir = os.path.abspath(dir)
        self.task0a_dir = os.path.join(self.dir, "task0a")
        self.task0b_dir = os.path.join(self.dir, "task0b")
        self.task1_dir = os.path.join(self.dir, "task1")
        self.pca = self.nmf = self.lda = self.svd = None
        self.out_dir = os.path.join(self.dir, "task2")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.compNames = ['words_X', 'words_Y', 'words_Z', 'words_W']
        self.file_paths = sorted([os.path.join(self.task0a_dir, f) for f in os.listdir(self.task0a_dir) if ".wrd" in f])
        self.tf_files = sorted([os.path.join(self.task0b_dir, k) for k in os.listdir(os.path.join(self.task0b_dir)) if "tf_" in k and ".txt" in k])
        self.tfidf_files = sorted([os.path.join(self.task0b_dir, k) for k in os.listdir(os.path.join(self.task0b_dir)) if "tfidf_" in k and ".txt" in k])
        self.sequences = {}
        self.tf, self.tfidf, self.entropy = [], [], []
        self.file_idx, self.idx_file = {}, {}
        self._read_wrd_files_()
        self._load_all_vectors_()

    def _load_all_vectors_(self):
        for fname in self.tf_files:
            name = fname.split("/")[-1].split("_")[-1].split(".")[0]
            idx = len(self.file_idx.keys())
            self.file_idx[name] = idx
            self.idx_file[idx] = name
            self.tf.append(json.loads(json.load(open(fname, "r"))))
            x = np.array(self.tf[-1])
            self.entropy.append(np.multiply(-x, np.log2(x+1e-7)))

        self.tfidf = []
        for fname in self.tfidf_files:
            self.tfidf.append(json.loads(json.load(open(fname, "r"))))

        self.tf = np.array(self.tf).reshape((len(self.tf_files), -1))
        self.tfidf = np.array(self.tfidf).reshape((len(self.tfidf_files), -1))
        self.entropy = np.array(self.entropy).reshape((len(self.tfidf_files), -1))

        if os.path.exists(os.path.join(self.task1_dir, "pca_{}_vectors.txt".format(self.model))):
            self.pca = np.array(json.loads(json.load(open(os.path.join(self.task1_dir, "pca_{}_vectors.txt".format(self.model)), "r"))))
        if os.path.exists(os.path.join(self.task1_dir, "svd_{}_vectors.txt".format(self.model))):
            self.svd = np.array(json.loads(json.load(open(os.path.join(self.task1_dir, "svd_{}_vectors.txt".format(self.model)), "r"))))
        if os.path.exists(os.path.join(self.task1_dir, "nmf_{}_vectors.txt".format(self.model))):
            self.nmf = np.array(json.loads(json.load(open(os.path.join(self.task1_dir, "nmf_{}_vectors.txt".format(self.model)), "r"))))
        if os.path.exists(os.path.join(self.task1_dir, "lda_{}_vectors.txt".format(self.model))):
            self.lda = np.array(json.loads(json.load(open(os.path.join(self.task1_dir, "lda_{}_vectors.txt".format(self.model)), "r"))))

        self.allWords = pkl.load(open(os.path.join(self.task0b_dir, "all_words_idx.txt"), "rb"))

    def _read_wrd_files_(self):
        for fp in self.file_paths:
            data = json.load(open(fp, "r"))
            file_name = fp.split("/")[-1].split(".")[0]
            self.sequences[file_name] = {}
            for c in self.compNames:
                self.sequences[file_name][c] = {}
                items = data[c]
                for sid in range(20):
                    words = [(it[1], it[2]) for it in items if it[0][1] == sid]
                    self.sequences[file_name][c][sid] = words

    def _construct_list_for_mp_(self, fn1, fn2):
        all_lists = []
        for c in self.compNames:
            for sid in range(20):
                all_lists.append((fn1, fn2, c, sid, self.sequences[fn1][c][sid], self.sequences[fn2][c][sid]))
        return all_lists

    def _edit_distance_(self, seqs):
        seq1 = seqs[4]
        seq2 = seqs[5]
        word1 = [item[0] for item in seq1]
        word2 = [item[0] for item in seq2]
        f1 = self.file_idx[seqs[0]]
        f2 = self.file_idx[seqs[1]]
        comp = seqs[2]
        sensor = seqs[3]

        n = len(word1)
        m = len(word2)

        # if one of the strings is empty
        if n * m == 0:
            return n + m

        # array to store the conversion history
        d = [[0] * (m + 1) for _ in range(n + 1)]

        # init boundaries
        for i in range(1, n + 1):
            d[i][0] = d[i-1][0] + self.entropy[f1, self.allWords[(comp, sensor, tuple(word1[i-1]))]]
        for j in range(1, m + 1):
            d[0][j] = d[0][j-1] + self.entropy[f2, self.allWords[(comp, sensor, tuple(word2[j-1]))]]

        # DP compute
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                e1 = self.entropy[f1, self.allWords[(comp, sensor, tuple(word1[i - 1]))]]
                e2 = self.entropy[f2, self.allWords[(comp, sensor, tuple(word2[j - 1]))]]
                cost = abs(e1 - e2)
                d[i][j] = min(d[i-1][j]+cost, d[i][j-1]+cost, d[i-1][j-1] if word1[i-1] == word2[j-1] else d[i-1][j-1]+cost)
        return d[n][m]

    def _dtw_distance_(self, seqs):
        seq1 = seqs[4]
        seq2 = seqs[5]
        word1 = [item[1] for item in seq1]
        word2 = [item[1] for item in seq2]

        n = len(word1)
        m = len(word2)

        dtw_matrix = [[0] * (m + 1) for i in range(n + 1)]

        for i in range(1, n + 1):
            dtw_matrix[i][0] = dtw_matrix[i - 1][0] + abs(word1[i - 1])
        for j in range(1, m + 1):
            dtw_matrix[0][j] = dtw_matrix[0][j - 1] + abs(word2[j - 1])

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(word1[i - 1] - word2[j - 1])
                # take last min from a square box
                last_min = min(dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])
                dtw_matrix[i][j] = cost + last_min
        return dtw_matrix[n][m]

    def _save_results_(self, scores, fn, option):
        names = ["dot_pdt_{}_{}.txt", "pca_cosine_{}_{}.txt", "svd_cosine_{}_{}.txt", "nmf_cosine_{}_{}.txt", "lda_cosine_{}_{}.txt",
                 "edit_dist_{}_{}.txt", "dtw_dist_{}_{}.txt"]
        fn = names[option-1].format(fn, self.model)
        json.dump(scores, open(os.path.join(self.out_dir, fn), "w"))

    def _dot_product_similarity_(self, fn):
        idx = self.file_idx[fn]
        if self.model == 1:
            scores = np.dot(self.tf, self.tf[idx].reshape((-1,1))).tolist()
        elif self.model == 2:
            scores = np.dot(self.tfidf, self.tfidf[idx].reshape((-1, 1))).tolist()
        scores = [(self.idx_file[id], s) for id,s in enumerate(scores)]
        top_10_scores = dict(sorted(scores, key=lambda x: x[1], reverse=True)[:10])
        return top_10_scores

    def _edit_cost_distance_(self, fn):
        scores = []
        for file_id in self.sequences:
            print("Calculating for : ", fn, file_id)
            all_list = self._construct_list_for_mp_(fn, file_id)
            scores.append(sum([self._edit_distance_(seqs) for seqs in all_list]))
        scores = [(self.idx_file[id], s) for id, s in enumerate(scores)]
        top_10_scores = dict(sorted(scores, key=lambda x: x[1])[:10])
        return top_10_scores

    def _dtw_cost_distance_(self, fn):
        scores = []
        for file_id in self.sequences:
            print("Calculating for : ", fn, file_id)
            all_list = self._construct_list_for_mp_(fn, file_id)
            scores.append(sum([self._dtw_distance_(seqs) for seqs in all_list]))
        scores = [(self.idx_file[id], s) for id, s in enumerate(scores)]
        top_10_scores = dict(sorted(scores, key=lambda x: x[1])[:10])
        return top_10_scores

    def _pca_similarity_(self, fn):
        idx = self.file_idx[fn]
        scores = (1-pairwise_distances(self.pca, metric="cosine"))[idx]
        scores = [(self.idx_file[id], s) for id, s in enumerate(scores)]
        top_10_scores = dict(sorted(scores, key=lambda x: x[1], reverse=True)[:10])
        return top_10_scores

    def _svd_similarity_(self, fn):
        idx = self.file_idx[fn]
        scores = (1-pairwise_distances(self.svd, metric="cosine"))[idx]
        scores = [(self.idx_file[id], s) for id, s in enumerate(scores)]
        top_10_scores = dict(sorted(scores, key=lambda x: x[1], reverse=True)[:10])
        return top_10_scores

    def _nmf_similarity_(self, fn):
        idx = self.file_idx[fn]
        scores = (1-pairwise_distances(self.nmf, metric="cosine"))[idx]
        scores = [(self.idx_file[id], s) for id, s in enumerate(scores)]
        top_10_scores = dict(sorted(scores, key=lambda x: x[1], reverse=True)[:10])
        return top_10_scores

    def _lda_similarity_(self, fn):
        idx = self.file_idx[fn]
        scores = (1-pairwise_distances(self.lda, metric="cosine"))[idx]
        scores = [(self.idx_file[id], s) for id, s in enumerate(scores)]
        top_10_scores = dict(sorted(scores, key=lambda x: x[1], reverse=True)[:10])
        return top_10_scores

    def find_10_similar_gestures(self, fn, option):
        if option == 1:
            scores = self._dot_product_similarity_(fn)
        elif option == 2:
            scores = self._pca_similarity_(fn)
        elif option == 3:
            scores = self._svd_similarity_(fn)
        elif option == 4:
            scores = self._nmf_similarity_(fn)
        elif option == 5:
            scores = self._lda_similarity_(fn)
        elif option == 6:
            scores = self._edit_cost_distance_(fn)
        elif option == 7:
            scores = self._dtw_cost_distance_(fn)

        self._save_results_(scores, fn, option)


if __name__ == "__main__":
    print("Performing Task 2")
    # directory = input("Enter directory to use: ")
    directory = "outputs"
    user_choice = 0
    # while user_choice != 8:
        # file_name = input("Enter the file id to use: ")

    file_name = [str(1)] #list(range(1, 10))
    # file_name = [str(x) for x in file_name]
    for i in file_name:
        print(i)
        file_name = i.zfill(3)
        print(file_name)
        for i in [1,2]:
            for j in [1,2,3,4,5,6,7]:
            # for j in [4]:
                task2 = Task2(i, directory)
                task2.find_10_similar_gestures(file_name, j)
    # vec_model = int(input("Enter which vector model to use. (1) TF (2) TFIDF : "))
    # print("User Options for similarity approaches, \n(1)Dot Product \n(2)PCA \}[\n(3)SVD \n(4)NMF \n(5)LDA \n(6)Edit Distance \n(7)DTW \n(8)Exit")
    # user_choice = int(input("Enter a user option: "))
    # if user_choice == 8:
    #     break
    # task2 = Task2(directory, vec_model)
    # task2.find_10_similar_gestures(file_name, user_choice)