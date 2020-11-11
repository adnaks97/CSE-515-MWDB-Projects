import os
import numpy as np
import json
import pandas as pd
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance
from scipy.stats import mode


class Task2:
    def __init__(self, input_dir, vm=2, uc=2):
        self.vm = vm
        self.uc = uc
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
        mat = np.array(
            json.loads(json.load(open(os.path.join(self.input_dir, "task3", names[self.uc].format(self.vm)), "r"))))
        return mat

    @staticmethod
    def process_ppr(adj_matrix_norm, idx, file_name):
        size = adj_matrix_norm.shape[0]
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        u_old[idx - 1, 0] = 1
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        v[idx - 1, 0] = 1
        A = adj_matrix_norm
        diff = 1
        c = 0.8
        while diff > 1e-20:
            u_new = ((1 - c) * np.matmul(A, u_old)) + (c * v)
            diff = distance.minkowski(u_new, u_old, 1)
            u_old = u_new
        result = (file_name, idx, u_new)
        return result

    def preprocess_ppr_task2(self, m, k):
        sim_matrix = self.get_sim_matrix()
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
            adj_matrix = self.get_knn_nodes(ppr_new_matrix, k)
            adj_matrix_norm = self.normalize(adj_matrix)
            idx = adj_matrix_norm.shape[0]
            res.append(pool.apply_async(self.process_ppr, args=(adj_matrix_norm, idx, file_name)).get())
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
        voting_acc = wt_scor_accuracy = 0.
        for f in result:
            cl_voting = mode(result[f]['classes'])[0][0]
            un_clss = np.unique(result[f]['classes'])
            scores = []
            for c in un_clss:
                scores.append(np.sum(result[f]['scores'][result[f]['classes']==c]))
            cl_wt_sc = un_clss[np.argmax(scores)]
            final_result[f] = {}
            cl_actual = self.all_files_classes[f]
            final_result[f]['voting'] = cl_voting
            final_result[f]['weighted_scores'] = cl_wt_sc
            final_result[f]['actual_label'] = cl_actual
            if cl_actual == cl_voting:
                voting_acc += 1
            if cl_actual == cl_wt_sc:
                wt_scor_accuracy += 1
        json.dump(final_result, open(self.output_dir + "/{}_{}_dominant.txt".format(k, m), "w"), indent="\t")
        print("Accuracy Voting {} Weighted Scoring {}".format(voting_acc/len(result),wt_scor_accuracy/len(result)))

if __name__ == "__main__":
    print("Performing Task 2")
    input_directory = "phase2_outputs" #input("Enter directory to use: ")
    knn_k = 5 #int(input("Enter a value K for KNN : "))
    ppr_k = 30 #int(input("Enter a value K for outgoing gestures (PPR) : "))
    m_value = 9 #int(input("Enter a value M for most dominant gestures : "))
    task2 = Task2(input_directory)
    task2.preprocess_ppr_task2(m_value, ppr_k)
