import os
import json
import math
import copy
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

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
        
        self.mat = self.get_sim_matrix()

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
        p[p < k] = 0
        p[p >= k] = 1
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
        adj_matrix = self.get_knn_nodes(self.mat, k)
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
        res = [x for x in u_new.ravel().argsort()[::-1][:m]]
        return res

    def query_optimization_prob(self, q, f_c, f_w):
        q = q.reshape((1,-1))
        vectors = copy.deepcopy(self.vectors)
        vectors = np.array(vectors).reshape((-1,len(vectors[0])))
        if f_c is not None and f_w is not None:
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
            q_new = q_new.reshape((1,-1))

        else:
            n_i = np.count_nonzero(vectors, axis=0).astype(np.float32)
            p_i = np.array([0.5]*len(vectors[0])).reshape((1,-1))
            u_i = n_i/len(vectors)
            u_i[u_i<=0] = 1e-5
            u_i[u_i>=1] = 0.99
            q_new = np.log((p_i*(1-u_i))/(u_i*(1-p_i)))
            q_new = q_new.reshape((1,-1))

        return q_new

    def query_optimization_vectors(self, Q, relevant, non_relevant):
        beta = 1 / len(relevant) if len(relevant) > 0 else 0
        gamma = 1 / len(non_relevant) if len(non_relevant) > 0 else 0

        dr, dnr = 0, 0
        # rel_files_vectors-contains vectors of all relevant file ids
        rel_files_vectors, non_rel_files_vectors = [], []

        for j in relevant:
            rel_files_vectors.append(self.vectors[j])

        a = []
        for c in range(len(rel_files_vectors)):
            mag = math.sqrt(sum(pow(element, 2) for element in rel_files_vectors[c]))
            l = [e / mag for e in rel_files_vectors[c]]
            a.append(l)
        dr = np.sum(np.array(a), axis=0)

        # non relevant files
        for j in non_relevant:
            non_rel_files_vectors.append(self.vectors[j])
        an = []
        for c in range(len(non_rel_files_vectors)):
            mag = math.sqrt(sum(pow(element, 2) for element in non_rel_files_vectors[c]))
            l = [e / mag for e in non_rel_files_vectors[c]]
            an.append(l)
        dnr = np.sum(np.array(an), axis=0)

        optimized_q = Q + beta * dr - gamma * dnr
        return optimized_q

    def modify_sim_matrix(self, query, idx):
        sims = cosine_similarity(self.vectors, query).reshape((-1,))
        self.mat[idx,:] = sims
        self.mat[:,idx] = sims

    def main_feedback_loop(self, query_file, k_value):
        idx = self.file_idx_map[query_file]
        q = np.array(self.vectors[idx]).reshape((1,-1))
        relevant = non_relevant = None
        while True:
            # call retrieval function
            print("Getting results")
            results = self.process_ppr(idx+1, 10, 30)
            # call user feedback function
            print("Getting feedback")
            relevant, non_relevant = self._get_user_feedback_(results)
            # call query mod function
            q_new_vect = self.query_optimization_vectors(q, relevant, non_relevant)
            q_new_prob = self.query_optimization_prob(q, relevant, non_relevant)
            q_change_prob = np.count_nonzero(q-q_new_prob)
            q_change_vect = np.count_nonzero(q-q_new_vect)
            print("Total query dimensions - {} \nVector optim changed dims - {} \nProbabilistic optim changed dims - {}".format(q.shape[1], q_change_vect, q_change_prob))
            q_opt = int(input("Choose which method to use? (q_vect-1, q_prob-2): "))
            if q_opt == 1:
                q_new = q_new_vect
            else:
                q_new = q_new_prob
            # update sim_matrix
            self.modify_sim_matrix(q_new, idx)
            # check for uc value
            uc = input("Continue? (y/n) : ")
            if uc == 'n' or uc == 'N':
                break
            q = q_new


if __name__ == "__main__":
    input_dir = "phase2_outputs"
    vm = int(input("Which vector model to use? (tf-1/ tfidf-2) : "))
    file = input("Enter a file number : ").zfill(3)
    k = int(input("Enter a value K for outgoing gestures : "))
    task5 = Task5(input_dir, vm=2, uc=2)
    task5.main_feedback_loop(file, k)
