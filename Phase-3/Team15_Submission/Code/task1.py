import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import distance


class Task1:
    def __init__(self, input_dir, vm=2, uc=2):
        self.vm = vm
        self.uc = uc
        self.input_dir = os.path.abspath(input_dir)
        self.input_dir_for_plots=os.path.abspath(input_dir_for_plots)
        self.output_dir = os.path.join("outputs", "task1")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        files = sorted([x.split(".")[0] for x in os.listdir(os.path.join(self.input_dir, "task0a")) if ".wrd" in x])
        indices = list(range(0, len(files)))
        self.idx_file_map = dict(zip(indices, files))
        self.file_idx_map = dict(zip(files, indices))

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
        names = {2:"pca_cosine_sim_matrix_{}.txt", 
                 3:"svd_cosine_sim_matrix_{}.txt",
                 4:"nmf_cosine_sim_matrix_{}.txt",
                 5:"lda_cosine_sim_matrix_{}.txt"}
        mat = np.array(json.loads(json.load(open(os.path.join(self.input_dir,"task3",names[self.uc].format(self.vm)), "r"))))
        return mat

    def plot_dominant_gestures(self,res,k):
        for j in range(len(res)):
            for i in ['W','X','Y','Z']:
                d = os.path.join(self.input_dir_for_plots, i)
                df=pd.read_csv(os.path.join(d, res[j].lstrip('0').replace('-','_')+".csv"), header=None)
                df=df.T
                fig=df.plot()
                outFile = os.path.join(self.output_dir, res[j] + '_' + i + '_' + str(k) + '.png')
                fig.figure.savefig(outFile)

    def process_ppr(self, n, m, k):
        self.output_dir = self.output_dir+"/{}_{}_{}".format(k,m,",".join([str(x) for x in n]))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        sim_matrix = self.get_sim_matrix()
        adj_matrix = self.get_knn_nodes(sim_matrix, k)
        adj_matrix_norm = self.normalize(adj_matrix)
        size = adj_matrix_norm.shape[0]
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        for value in n:
            u_old[value - 1] = 1/len(n)
            v[value - 1] = 1/len(n)
        A = adj_matrix_norm
        diff = 1
        c = 0.65
        while diff > 1e-20:
            u_new = ((1 - c) * np.matmul(A, u_old)) + (c * v)
            diff = distance.minkowski(u_new, u_old, 1)
            u_old = u_new
        res = [self.idx_file_map[x] for x in u_new.ravel().argsort()[::-1][:m]]
        self.plot_dominant_gestures(res,k)
        c = {}
        c['user_files'] = n
        c['dominant_gestures'] = res 
        json.dump(c, open(self.output_dir + "/{}_{}_dominant.txt".format(k, m), "w"), indent='\t')
        

if __name__ == "__main__":
    print("Performing Task 1")
    input_directory = "phase2_outputs" # input("Enter the input directory to use: ")
    input_dir_for_plots=input("Enter the directory that contains all files of all components ")
    k = int(input("Enter a value K for outgoing gestures : "))
    m = int(input("Enter a value M for most dominant gestures : "))
    n = list(map(int, input("Enter N indices separated by space : ").split()))
    task1 = Task1(input_directory)
    task1.process_ppr(n, m, k)
