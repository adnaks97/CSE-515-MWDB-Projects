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
from scipy.stats import mode


class Task2:
    def __init__(self, input_dir):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join("outputs", "task2")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        files = sorted([x.split(".")[0] for x in os.listdir(os.path.join(self.input_dir, "task0a")) if ".wrd" in x])
        indices = list(range(0, len(files)))
        train_labels = pd.read_excel("sample_training_labels.xlsx", header=None)
        all_labels = pd.read_excel("all_labels.xlsx", header=None)
        self.all_files_classes = dict(zip(all_labels.iloc[:,0].apply(lambda x: str(x).zfill(3)).tolist(),all_labels.iloc[:,1].tolist()))
        self.train_file_num = train_labels.iloc[:, 0].tolist()
        self.train_file_num = [str(i).zfill(3) for i in self.train_file_num]
        self.class_labels_map = dict(zip(self.train_file_num,train_labels.iloc[:,1].tolist()))
        self.remove_indices_mat = list(set(files) - set(self.train_file_num))
        self.idx_file_map = dict(zip(indices, files))
        self.file_idx_map = dict(zip(files, indices))
        self.train_file_num_indices = sorted([self.file_idx_map[x] for x in self.train_file_num])
        self.remove_indices_mat = sorted([self.file_idx_map[x] for x in self.remove_indices_mat])
        self.reduced_index = dict(zip(list(range(len(self.train_file_num_indices))),self.train_file_num_indices))

    def preprocess_ppr_task2(self, m, k):
        sim_matrix = task1.get_sim_matrix()
        ppr_matrix = sim_matrix
        ppr_matrix = sim_matrix[np.array(self.train_file_num_indices)[:, None], np.array(self.train_file_num_indices)]
        res = []
        pool = ThreadPool(4000)
        for index in self.remove_indices_mat:
            file_name = self.idx_file_map[index]
            new_row = sim_matrix[index, self.train_file_num_indices]
            new_column = sim_matrix[self.train_file_num_indices, index]
            new_column = np.append(new_column, 1)
            ppr_new_matrix = np.column_stack((np.vstack([ppr_matrix, new_row]),new_column))
            adj_matrix = task1.get_knn_nodes(ppr_new_matrix, k)
            adj_matrix_norm = task1.normalize(adj_matrix)
            idx = adj_matrix_norm.shape[0]
            res.append(pool.apply_async(task1.process_ppr, args=(adj_matrix_norm, idx, file_name)).get())
        pool.close()
        pool.join()
        
        result = {}
        for n_value in res:
            files = np.array([self.idx_file_map[self.reduced_index[x]] for x in n_value[2].ravel()[:-1].argsort()[::-1][:m]])
            classes = np.array([self.class_labels_map[x] for x in files])
            scores = np.array(sorted(n_value[2].ravel())[:-1][::-1][:m])
            scores = scores / (scores.sum(axis=0, keepdims=1) + 1e-7)
            result[n_value[0]] = {} 
            result[n_value[0]]['files'] = files
            result[n_value[0]]['classes'] = classes
            result[n_value[0]]['scores'] = scores
        self.post_process(result,k,m)

    def post_process(self, result, k, m):
        final_result = {}
        for f in result:
            cl_voting = mode(result[f]['classes'])[0][0]
            un_clss = np.unique(result[f]['classes'])
            scores = []
            for c in un_clss:
                scores.append(np.sum(result[f]['scores'][result[f]['classes']==c]))
            cl_wt_sc = un_clss[np.argmax(scores)]
            final_result[f] = {}
            final_result[f]['voting'] = cl_voting
            final_result[f]['weighted_scores'] = cl_wt_sc
            final_result[f]['actual_label'] = self.all_files_classes[f]
        json.dump(final_result, open(self.output_dir + "/{}_{}_dominant.txt".format(k, m), "w"), indent="\t")


if __name__ == "__main__":
    print("Performing Task 2")
    input_directory = "phase2_outputs" #input("Enter directory to use: ")
    knn_k = 5 #int(input("Enter a value K for KNN : "))
    ppr_k = 30 #int(input("Enter a value K for outgoing gestures (PPR) : "))
    m_value = 11 #int(input("Enter a value M for most dominant gestures : "))
    task2 = Task2(input_directory)
    task1 = Task1(input_directory)
    task2.preprocess_ppr_task2(m_value, ppr_k)
