import os
import math
import numpy as np
from pathlib import Path
from sklearn.preprocessing import normalize

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
    
    def 