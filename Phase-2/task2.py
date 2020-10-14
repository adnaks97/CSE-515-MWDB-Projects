import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import pairwise_distances


class Task2:
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        self.pca = self.nmf = self.lda = self.svd = None
        self.out_dir = os.path.join(self.dir, "task2")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.tf_files = sorted([os.path.join(self.dir, "task0b", k) for k in os.listdir(os.path.join(self.dir, "task0b")) if "tf_" in k and ".txt" in k])
        self.tfidf_files = sorted([os.path.join(self.dir, "task0b", k) for k in os.listdir(os.path.join(self.dir, "task0b")) if "tfidf_" in k and ".txt" in k])
        self.tf, self.tfidf = [], []
        self.file_idx, self.idx_file = {}, {}
        self._load_all_vectors_()


    def _load_all_vectors_(self):
        for fname in self.tf_files:
            name = fname.split("/")[-1].split("_")[-1].split(".")[0]
            idx = len(self.file_idx.keys())
            self.file_idx[name] = idx
            self.idx_file[idx] = name
            self.tf.append(json.loads(json.load(open(fname, "r"))))

        self.tfidf = []
        for fname in self.tfidf_files:
            self.tfidf.append(json.loads(json.load(open(fname, "r"))))

        self.tf = np.array(self.tf).reshape((len(self.tf_files), -1))
        self.tfidf = np.array(self.tfidf).reshape((len(self.tfidf_files), -1))

        if os.path.exists(os.path.join(self.dir, "task1", "pca_vectors.txt")):
            self.pca = np.array(json.loads(json.load(open(os.path.join(self.dir, "task1", "pca_vectors.txt"), "r"))))

        if os.path.exists(os.path.join(self.dir, "task1", "svd_vectors.txt")):
            self.svd = np.array(json.loads(json.load(open(os.path.join(self.dir, "task1", "svd_vectors.txt"), "r"))))

        if os.path.exists(os.path.join(self.dir, "task1", "nmf_vectors.txt")):
            self.nmf = np.array(json.loads(json.load(open(os.path.join(self.dir, "task1", "nmf_vectors.txt"), "r"))))

        if os.path.exists(os.path.join(self.dir, "task1", "lda_vectors.txt")):
            self.lda = np.array(json.loads(json.load(open(os.path.join(self.dir, "task1", "lda_vectors.txt"), "r"))))

    def _save_results_(self, scores, fn, option):
        names = ["dot_pdt_{}.txt", "pca_cosine_{}.txt", "svd_cosine_{}.txt", "nmf_cosine_{}.txt", "lda_cosine_{}.txt"]
        fn = names[option-1].format(fn)
        json.dump(scores, open(os.path.join(self.out_dir, fn), "w"))

    def _dot_product_similarity_(self, fn, model):
        idx = self.file_idx[fn]
        if model == 1:
            scores = np.dot(self.tf, self.tf[idx].reshape((-1,1))).tolist()
        elif model == 2:
            scores = np.dot(self.tfidf, self.tfidf[idx].reshape((-1, 1))).tolist()
        scores = [(self.idx_file[id], s) for id,s in enumerate(scores)]
        top_10_scores = dict(sorted(scores, key=lambda x: x[1], reverse=True)[:10])
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

    def find_10_similar_gestures(self, fn, model, option):
        if option == 1:
            scores = self._dot_product_similarity_(fn, model)
        elif option == 2:
            scores = self._pca_similarity_(fn)
        elif option == 3:
            scores = self._svd_similarity_(fn)
        elif option == 4:
            scores = self._nmf_similarity_(fn)
        elif option == 5:
            scores = self._lda_similarity_(fn)

        self._save_results_(scores, fn, option)


if __name__ == "__main__":
    print("Performing Task 2")
    directory = input("Enter directory to use: ")
    task2 = Task2(directory)
    user_choice = 0
    while user_choice != 8:
        file_name = input("Enter the file id to use: ")
        file_name = "0"+file_name if len(file_name) == 1 else file_name
        vec_model = int(input("Enter which vector model to use. (1) TF (2) TFIDF : "))
        print("User Options for similarity approaches, \n(1)Dot Product \n(2)PCA \n(3)SVD \n(4)NMF \n(5)LDA \n(6)Edit Distance \n(7)DTW \n(8)Exit")
        user_choice = int(input("Enter a user option: "))
        if user_choice == 8:
            break
        task2.find_10_similar_gestures(file_name, vec_model, user_choice)