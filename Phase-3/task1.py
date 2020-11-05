import os
import math
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import normalize
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance


class Task1:
    def __init__(self, input_dir):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join("outputs1", "task1")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_knn_nodes(sim_matrix, k=3):
        p = np.argsort(sim_matrix, axis=1)
        p[p<k] = 0
        p[p>=k] = 1
        p = p.astype(bool)
        sim_matrix_truncated = np.where(p, sim_matrix, 0)
        return sim_matrix_truncated
    
    @staticmethod
    def normalize(mat):
        return mat/(mat.sum(axis=0, keepdims=1)+1e-7)

    @staticmethod
    def get_sim_matrix():
        mat = np.array(json.loads(json.load(open("outputs/task3/nmf_cosine_sim_matrix_2.txt", "r"))))
        return mat

    def preprocess_ppr(self, n, m, k):
        sim_matrix = self.get_sim_matrix()
        adj_matrix = self.get_knn_nodes(sim_matrix, k)
        adj_matrix_norm = self.normalize(adj_matrix)
        res = []
        pool = ThreadPool(4000)
        for idx in n:
            res.append(pool.apply_async(self.process_ppr, args=(adj_matrix_norm, idx,)).get())
        pool.close()
        pool.join()
        self.cal_m_dominant(res, m, k)

    def cal_m_dominant(self, n_sim, m_value, k_value):
        res = {}
        for n_value in n_sim:
            res[n_value[0]] = n_value[1].ravel().argsort()[::-1][:m_value].tolist()
        json.dump(res, open(self.output_dir + "/{}_{}_dominant.txt".format(k_value, m_value), "w"))

    @staticmethod
    def process_ppr(adj_matrix_norm, idx):
        size = adj_matrix_norm.shape[0]
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        u_old[idx - 1, 0] = 1
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        v[idx - 1, 0] = 1
        A = adj_matrix_norm
        diff = 1
        c = 0.15
        while diff > 1e-20:
            u_new = ((1-c) * np.matmul(A, u_old)) + (c * v)
            diff = distance.minkowski(u_new, u_old, 1)
            u_old = u_new
        result = (idx, u_new)
        return result


if __name__ == "__main__":
    print("Performing Task 1")
    input_directory = input("Enter directory to use: ")
    k = int(input("Enter a value K for outgoing gestures : "))
    m = int(input("Enter a value M for most dominant gestures : "))
    n = list(map(int, input("Enter N indices separated by space : ").split()))
    n = list(range(32,94,1))
    task1 = Task1(input_directory)
    task1.preprocess_ppr(n, m, k)
