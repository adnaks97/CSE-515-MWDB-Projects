import os
import json
import math
import copy
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from scipy.stats import mode


class Task5:
    def __init__(self, input_dir, vm=2, uc=2):
        # setting up inout and output directories
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join("outputs", "task5")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # creating map for files and indices
        files = sorted([x.split(".")[0] for x in os.listdir(os.path.join(self.input_dir, "task0a")) if ".wrd" in x])
        indices = list(range(0, len(files)))
        self.idx_file_map = dict(zip(indices, files))
        self.file_idx_map = dict(zip(files, indices))
        self.diff = []
        self.vm = vm
        self.uc = uc
        self._get_file_vectors_()

    def _get_file_vectors_(self):
        self.vectors = []
        names = {2: "pca_{}_vectors.txt",
                 3: "svd_{}_vectors.txt",
                 4: "nmf_{}_vectors.txt",
                 5: "lda_{}_vectors.txt"}
        if self.uc in [2,3,4,5]:
            self.vectors = json.loads(
                json.load(open(os.path.join(self.input_dir, "task1", names[self.uc].format(self.vm)), "r")))
        # load tf vectors
        if self.uc == 6:
            files = sorted(
                [os.path.join(self.input_dir, "task0b", x) for x in os.listdir(os.path.join(self.input_dir, "task0b")) if
                 "tf_vectors" in x])
            for f in files:
                self.vectors.append(np.array(json.loads(json.load(open(f, "r")))).reshape((-1, 1)))
        # load tfidf vectors
        elif self.uc == 7:
            files = sorted(
                [os.path.join(self.input_dir, "task0b", x) for x in os.listdir(os.path.join(self.input_dir, "task0b")) if
                 "tfidf_vectors" in x])
            for f in files:
                self.vectors.append(np.array(json.loads(json.load(open(f, "r")))).reshape((-1, 1)))

    def _get_user_feedback_(self, results):
        rel, non_rel = [], []
        for i, r in enumerate(results, 1):
            print("{} : File ID {}".format(i, self.idx_file_map[r]))
            fd = input("Enter your feedback(1-Correct/0-Wrong) : ")
            if fd == '1':
                rel.append(r)
            elif fd == '0':
                non_rel.append(r)
        return rel, non_rel

    @staticmethod
    def get_knn_nodes(sim_matrix, k=3):
        k = sim_matrix.shape[0] - k
        p = np.argsort(sim_matrix, axis=1)
        p[p <= k] = 0
        p[p > k] = 1
        p = p.astype(bool)
        sim_matrix_truncated = np.where(p, sim_matrix, 0)
        return sim_matrix_truncated

    @staticmethod
    def normalize(mat):
        return mat / (mat.sum(axis=0, keepdims=1) + 1e-7)

    def get_sim_matrix(self):
        names = {2: "pca_cosine_sim_matrix_{}.txt",
                 3: "svd_cosine_sim_matrix_{}.txt",
                 4: "nmf_cosine_sim_matrix_{}.txt",
                 5: "lda_cosine_sim_matrix_{}.txt"}
        if self.uc in [2,3,4,5]:
            mat = np.array(
                json.loads(json.load(open(os.path.join(self.input_dir, "task3", names[self.uc].format(self.vm)), "r"))))
        elif self.uc in [6,7]:
            mat = (1 - pairwise_distances(self.vectors, metric="cosine"))
        return mat

    def process_ppr(self, value, m, k):
        sim_matrix = self.get_sim_matrix()
        adj_matrix = self.get_knn_nodes(sim_matrix, k)
        adj_matrix_norm = self.normalize(adj_matrix)
        size = adj_matrix_norm.shape[0]
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        u_old[value - 1] = 1
        v[value - 1] = 1
        A = adj_matrix_norm
        diff = 1
        c = 0.65
        while diff > 1e-20:
            u_new = ((1 - c) * np.matmul(A, u_old)) + (c * v)
            diff = distance.minkowski(u_new, u_old, 1)
            u_old = u_new
        res = [self.idx_file_map[x] for x in u_new.ravel().argsort()[::-1][:m]]
        # c = {}
        # c['user_files'] = n
        # c['dominant_gestures'] = res
        # json.dump(c, open(self.output_dir + "/{}_{}_dominant.txt".format(k, m), "w"), indent='\t')
        return res

    def query_optimization_prob(self, q, f_c, f_w):
        q = q.reshape((1,-1))
        vectors = copy.deepcopy(self.tf_vectors) if self.vm == 1 else copy.deepcopy(self.tfidf_vectors)
        vectors = np.array(vectors).reshape((-1,len(vectors[0])))
        if f_c is not None and f_w is not None:
            q_new = []
            R = vectors[f_c,:]
            fcr = np.array([len(f_c)]*len(R[0])).reshape((1,-1))
            q_i = (q!=0).astype(int)*0.5
            r_i = np.count_nonzero(R, axis=0).astype(np.float32)
            n_i = np.count_nonzero(vectors, axis=0).astype(np.float32)
            r_i = r_i+q_i
            fcr = fcr+q_i

            p_i = (r_i+(n_i/len(vectors)))/(fcr+1)
            p_i[p_i<0] = 1e-5
            p_i[p_i>=1] = 0.99
            u_i = (n_i-r_i+(n_i/len(vectors)))/(len(vectors)-fcr+1)
            u_i[u_i<=0] = 1e-5
            u_i[u_i>=1] = 0.99
            q_new = np.log((p_i*(1-u_i))/(u_i*(1-p_i)))
            q_new = q_new.reshape((-1,))

        else:
            n_i = np.count_nonzero(vectors, axis=0).astype(np.float32)
            p_i = np.array([0.5]*len(vectors[0])).reshape((1,-1))
            u_i = n_i/len(vectors)
            u_i[u_i<=0] = 1e-5
            u_i[u_i>=1] = 0.99
            q_new = np.log((p_i*(1-u_i))/(u_i*(1-p_i)))
            q_new = q_new.reshape((-1,))

        return q_new

    def query_optimization_vectors(self, Q, relevant, non_relevant):
        beta = 1 / len(relevant) if len(relevant) > 0 else 0
        gamma = 1 / len(non_relevant) if len(non_relevant) > 0 else 0

        dr, dnr = 0, 0
        # rel_files_vectors-contains vectors of all relevant file ids
        rel_files_vectors, non_rel_files_vectors = [], []

        for j in relevant:
            rel_files_vectors.append(self.tf_vectors[j])

        a = []
        for c in range(len(rel_files_vectors)):
            mag = math.sqrt(sum(pow(element, 2) for element in rel_files_vectors[c]))
            l = [e / mag for e in rel_files_vectors[c]]
            a.append(l)
        dr = np.sum(np.array(a), axis=0)

        # non relevant files
        for j in non_relevant:
            non_rel_files_vectors.append(self.tf_vectors[j])
        an = []
        for c in range(len(non_rel_files_vectors)):
            mag = math.sqrt(sum(pow(element, 2) for element in non_rel_files_vectors[c]))
            l = [e / mag for e in non_rel_files_vectors[c]]
            an.append(l)
        dnr = np.sum(np.array(an), axis=0)

        optimized_q = Q + beta * dr - gamma * dnr
        return optimized_q

    def main_feedback_loop(self, query_file, k_value):
        idx = self.file_idx_map[query_file]
        q = self.vectors[idx]
        relevant = non_relevant = None
        while True:
            # call retrieval function
            print("Getting results")
            q_new, results = self.new_prob_retrieval(q, relevant, non_relevant)
            # call user feedback function
            print("Getting feedback")
            relevant, non_relevant = self._get_user_feedback_(results)
            # call query mod function
            q_new_vect = self.query_optimization(q, relevant, non_relevant)
            # call retrieval function
            print("Getting results")
            results = self.process_ppr(q, )
            # call user feedback function
            print("Getting feedback")
            relevant, non_relevant = self._get_user_feedback_(results)
            # call query mod function
            print("Query optimization")
            q_new = self.query_optimization(q, relevant, non_relevant)
            # check for uc value
            uc = input("Are you done? (y/n) : ")
            if uc == 'y' or uc == 'Y':
                break
            q = q_new


if __name__ == "__main__":
    input_dir = "phase2_outputs"
    vm = int(input("Which vector model to use? (tf-1/ tfidf-2) : "))
    file = input("Enter a file number : ").zfill(3)
    k = int(input("Enter a value K for outgoing gestures : "))
    for vm in [1,2]:
        for uc in [2,3,4,5,6,7]:
            task5 = Task5(input_dir, vm=vm, uc=uc)
            task5.main_feedback_loop(file, k)
