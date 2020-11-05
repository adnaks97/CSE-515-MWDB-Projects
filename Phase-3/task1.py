import os
import math
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance

class Task1:
    def __init__(self, input_dir, output_dir):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join(os.path.abspath(output_dir),"task1")

    @staticmethod
    def get_knn_nodes(sim_matrix, k=3):
        p = np.argsort(sim_matrix, axis=1)
        p[p>=k] = 1
        p[p<k] = 0
        p = p.astype(bool)
        sim_matrix_truncated = np.where(p, sim_matrix, 0)
        return sim_matrix_truncated
    
    @staticmethod
    def normalize(mat):
        return mat/mat.sum(axis=0,keepdims=1)

    @staticmethod
    def get_sim_matrix():
        return 0

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

    @staticmethod
    def process_ppr(adj_matrix_norm, idx):
        size = adj_matrix_norm.shape[0]
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        u_old[idx - 1, 0] = 1
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        v[idx - 1, 0] = 1
        A = adj_matrix_norm
        diff = 1
        while diff > 0.2:
            u_new = (0.2 * np.matmul(A, u_old)) + (0.8 * v)
            diff = distance.euclidean(u_new, u_old)
            u_old = u_new
        result = (idx, u_new)
        return result


