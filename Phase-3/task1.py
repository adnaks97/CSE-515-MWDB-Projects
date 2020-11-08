import os
import math
import json
import numpy as np
from pathlib import Path
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from multiprocessing.dummy import Pool as ThreadPool


class Task1:
    def __init__(self, input_dir, vm=2, uc=2):
        self.vm = vm
        self.uc = uc
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join("outputs", "task1")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        files = sorted([x.split(".")[0] for x in os.listdir(os.path.join(self.input_dir, "task0a")) if ".wrd" in x])
        indices = list(range(0, len(files)))
        self.idx_file_map = dict(zip(indices, files))
        self.file_idx_map = dict(zip(files, indices))

    @staticmethod
    def get_knn_nodes(sim_matrix, k=3):
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
        names = {2:"pca_cosine_sim_matrix_{}.txt", 
                 3:"svd_cosine_sim_matrix_{}.txt",
                 4:"nmf_cosine_sim_matrix_{}.txt",
                 5:"lda_cosine_sim_matrix_{}.txt"}
        mat = np.array(json.loads(json.load(open(os.path.join(self.input_dir,"task3",names[self.uc].format(self.vm)), "r"))))
        return mat

    def preprocess_ppr(self, n, m, k):
        sim_matrix = self.get_sim_matrix()
        adj_matrix = self.get_knn_nodes(sim_matrix, k)
        adj_matrix_norm = self.normalize(adj_matrix)
        res = []
        pool = ThreadPool(4000)
        for idx in n:
            res.append(pool.apply_async(self.process_ppr, args=(adj_matrix_norm, idx, self.idx_file_map[idx],)).get())
        pool.close()
        pool.join()
        self.cal_m_dominant(res, m, k)

    def cal_m_dominant(self, n_sim, m_value, k_value):
        res = {}
        for n_value in n_sim:
            files = [self.idx_file_map[x] for x in n_value[2].ravel().argsort()[::-1][:m_value]]
            res[n_value[0]] = files
        json.dump(res, open(self.output_dir + "/{}_{}_dominant.txt".format(k_value, m_value), "w"))

    @staticmethod
    def process_ppr(adj_matrix_norm, idx, file_name):
        size = adj_matrix_norm.shape[0]
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        u_old[idx - 1, 0] = 1
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        v[idx - 1, 0] = 1
        A = adj_matrix_norm
        diff = 1
        c = 0.15
        while diff > 1e-20:
            u_new = ((1 - c) * np.matmul(A, u_old)) + (c * v)
            diff = distance.minkowski(u_new, u_old, 1)
            u_old = u_new
        result = (file_name, idx, u_new)
        return result


if __name__ == "__main__":
    print("Performing Task 1")
    input_directory = input("Enter directory to use: ")
    k = int(input("Enter a value K for outgoing gestures : "))
    m = int(input("Enter a value M for most dominant gestures : "))
    n = list(map(int, input("Enter N indices separated by space : ").split()))
    task1 = Task1(input_directory)
    task1.preprocess_ppr(n, m, k)
