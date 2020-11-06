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
        self.labels = pd.read_excel("all_labels.xlsx")
        self.train_labels = pd.read_excel("sample_training_labels.xlsx")
        self.train_file_num = self.train_labels.iloc[:, 0].tolist()
        self.all_file_num = self.labels.iloc[:, 0].tolist()
        self.labels = self.labels.iloc[:, [0, 1]].apply(tuple, axis=1).to_numpy().tolist()
        self.train_labels = self.train_labels.iloc[:, [0, 1]].apply(tuple, axis=1).to_numpy().tolist()
        self.n = list(set(self.all_file_num) - set(self.train_file_num))


    def preprocess_ppr_task2(self, m, k):
        sim_matrix = self.get_sim_matrix()
        adj_matrix = self.get_knn_nodes(sim_matrix, k)
        adj_matrix_norm = self.normalize(adj_matrix)
        res = []
        pool = ThreadPool(4000)
        for idx in self.n:
            res.append(pool.apply_async(self.process_ppr, args=(adj_matrix_norm, idx,)).get())
        pool.close()
        pool.join()
        self.cal_m_dominant(res, m, k)


if __name__ == "__main__":
    print("Performing Task 2")
    input_directory = input("Enter directory to use: ")
    knn_k = int(input("Enter a value K for KNN : "))
    ppr_k = int(input("Enter a value K for outgoing gestures (PPR) : "))
    m_value = int(input("Enter a value M for most dominant gestures : "))
    task2 = Task2(input_directory)
    task1 = Task1(input_directory)
    task2.preprocess_ppr_task2(m_value, ppr_k)
