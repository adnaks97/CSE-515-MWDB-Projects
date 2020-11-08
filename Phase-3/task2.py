import os
import math
import numpy as np
import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance
from task1 import Task1


class Task2:
    def __init__(self, input_dir):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join("outputs1", "task1")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # self.labels = pd.read_excel("all_labels.xlsx")
        # self.train_labels = pd.read_excel("sample_training_labels.xlsx")
        # self.train_file_num = self.train_labels.iloc[:, 0].tolist()
        # self.all_file_num = self.labels.iloc[:, 0].tolist()
        # self.labels = self.labels.iloc[:, [0, 1]].apply(tuple, axis=1).to_numpy().tolist()
        # self.train_labels = self.train_labels.iloc[:, [0, 1]].apply(tuple, axis=1).to_numpy().tolist()
        # self.n = list(set(self.all_file_num) - set(self.train_file_num))
        files = sorted([x.split(".")[0] for x in os.listdir(os.path.join(self.input_dir, "task0a")) if ".wrd" in x])
        indices = list(range(0, len(files)))
        train_labels = pd.read_excel("sample_training_labels.xlsx", header=None)
        self.train_file_num = train_labels.iloc[:, 0].tolist()
        self.train_file_num = [str(i).zfill(3) for i in self.train_file_num]
        print(self.train_file_num)
        self.remove_indices_mat = list(set(files) - set(self.train_file_num))
        print(sorted(self.remove_indices_mat))
        # self.remove_indices_mat = [int(i.lstrip('0')) for i in self.remove_indices_mat]
        # print(sorted(self.remove_indices_mat))
        self.idx_file_map = dict(zip(indices, files))
        self.file_idx_map = dict(zip(files, indices))
        self.train_file_num_indices = [self.file_idx_map[x] for x in self.train_file_num]
        self.remove_indices_mat = [self.file_idx_map[x] for x in self.remove_indices_mat]
        print(sorted(self.remove_indices_mat))
        print(sorted(self.train_file_num_indices))

    def preprocess_ppr_task2(self, m, k):
        sim_matrix = task1.get_sim_matrix()
        ppr_matrix = sim_matrix
        ppr_matrix = np.delete(ppr_matrix, self.remove_indices_mat, 0)
        ppr_matrix = np.delete(ppr_matrix, self.remove_indices_mat, 1)
        print(ppr_matrix.shape)
        for index in self.remove_indices_mat:
            new_row = sim_matrix[index]
            new_row = np.delete(new_row, self.remove_indices_mat, 0)
            new_column = sim_matrix[:, index]
            new_column = np.delete(new_column, self.remove_indices_mat, 0)
            new_column = np.append(new_column, 1)
            ppr_new_matrix = np.vstack([ppr_matrix, new_row])
            ppr_new_matrix = np.column_stack((ppr_new_matrix, new_column))
            print(ppr_new_matrix.shape)
            adj_matrix = task1.get_knn_nodes(ppr_new_matrix, k)
            adj_matrix_norm = task1.normalize(adj_matrix)
            res = []
            pool = ThreadPool(4000)
            for i in self.train_file_num:
                idx = adj_matrix_norm.shape[0]
                res.append(pool.apply_async(task1.process_ppr, args=(adj_matrix_norm, idx, i)).get())
            pool.close()
            pool.join()
            task1.cal_m_dominant(res, m, k)


if __name__ == "__main__":
    print("Performing Task 2")
    input_directory = input("Enter directory to use: ")
    knn_k = int(input("Enter a value K for KNN : "))
    ppr_k = int(input("Enter a value K for outgoing gestures (PPR) : "))
    m_value = int(input("Enter a value M for most dominant gestures : "))
    task2 = Task2(input_directory)
    task1 = Task1(input_directory)
    task2.preprocess_ppr_task2(m_value, ppr_k)
