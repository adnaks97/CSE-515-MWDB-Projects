import os
import json
import math
import copy
import numpy as np
import pandas as pd
import pickle as pkl
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

class Display(object):
    def submit_query(self):
        self.query_id = int(self.query_entry.get())
        self.results_cnt = int(self.param_entry.get())
        self.window.destroy()
        
    def create_first(self):
        self.window = tk.Tk()
        frame1 = tk.Frame(master=self.window, width=100)
        query_label = tk.Label(master=frame1, width=50, text="Enter query file id")
        self.query_entry = tk.Entry(master=frame1, width=20)
        param_label = tk.Label(master=frame1, width=50, text="Enter number of files to retrieve")
        self.param_entry = tk.Entry(master=frame1, width=20)
        submit = tk.Button(master=frame1, width=10, text="submit", command=self.submit_query)
        query_label.pack()
        self.query_entry.pack()
        param_label.pack()
        self.param_entry.pack()
        submit.pack()
        frame1.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
        self.window.mainloop()
    
    def submit_feedback(self):
        self.relevant, self.irrelevant = [], []
        for fr in self.frames:
            if fr['var'].get() == 1:
                self.relevant.append(fr['name']['text'])
            elif fr['var'].get() == 0:
                self.irrelevant.append(fr['name']['text'])
        self.window.destroy()
    
    def result_and_feedback(self, query_id, results):
        self.window = tk.Tk()
        query_id = tk.Label(master=self.window, width=60, text="Results for query id " + query_id)
        query_id.pack()
        self.frames = []
        for i in range(len(results)):
            items = {}
            items['frame'] = tk.Frame(master=self.window, width=100, height=100)
            items['name'] = tk.Label(master=items['frame'], width=60, text=results[i])
            items['var'] = tk.IntVar()
            c = ttk.Radiobutton(items['frame'], text='Correct', variable=items['var'], value=1)
            ic = ttk.Radiobutton(items['frame'], text='Incorrect', variable=items['var'], value=0)
            dk = ttk.Radiobutton(items['frame'], text="Don't know", variable=items['var'], value=-1)
            items['name'].pack(fill=tk.BOTH, side=tk.LEFT)
            c.pack(fill=tk.BOTH, side=tk.LEFT)
            ic.pack(fill=tk.BOTH, side=tk.LEFT)
            dk.pack(fill=tk.BOTH, side=tk.LEFT)
            items['frame'].pack(pady=7, padx=10)
            self.frames.append(items)
        submit = tk.Button(master=self.window, width=10, text="submit", command=self.submit_feedback)
        submit.pack()
        self.window.mainloop()

class Task6:
    def __init__(self, input_dir, vm=2, uc=2):
        # setting up inout and output directories
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join("outputs", "task6")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.display = Display()
        self.display.create_first()
        self.query_id = str(self.display.query_id).zfill(3)
        self.top_retrieve = self.display.results_cnt
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
                self.vectors.append(np.array(json.loads(json.load(open(f, "r")))))
        # load tfidf vectors
        elif self.uc == 7:
            files = sorted(
                [os.path.join(self.input_dir, "task0b", x) for x in os.listdir(os.path.join(self.input_dir, "task0b")) if
                 "tfidf_vectors" in x])
            for f in files:
                self.vectors.append(np.array(json.loads(json.load(open(f, "r")))))
    
    ################################################################# PPR Retrieval functions ##############################################################
    
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

    def process_ppr(self, mat, value, m, k, c=0.65):
        adj_matrix = self.get_knn_nodes(mat, k)
        adj_matrix_norm = self.normalize(adj_matrix)
        size = adj_matrix_norm.shape[0]
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        u_old[value - 1] = 1
        v[value - 1] = 1
        A = adj_matrix_norm
        diff = 1
        while diff > 1e-10:
            u_new = ((1 - c) * np.matmul(A, u_old)) + (c * v)
            diff = distance.minkowski(u_new, u_old, 1)
            print(diff)
            u_old = u_new
        res = [x for x in u_new.ravel().argsort()[::-1][:m]]
        return res
    
    def modify_sim_matrix(self, mat, query, idx):
        sims = cosine_similarity(self.vectors, query).reshape((-1,))
        mat[idx,:] = sims
        mat[:,idx] = sims
        return mat 

    ###### Probabilistic Retrieval functions #######################################################################################################################
    
    def prob_retrieval(self, q_new):
        sim = {}
        for j in range(len(self.vectors)):
            sim[j] = np.sum(np.multiply(self.vectors[j],q_new))

        res = list(dict(sorted(sim.items(), key=lambda x: x[1], reverse=True)[:10]).keys())
        return res
    
    ######################################################## Query Optimizations #############################################################

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

        q_new = Q + beta * dr - gamma * dnr
        return q_new

    ############################# ---------------------------------------------------------------------- #####################################

    def choose_result_set(self, results):
        return list(results.keys())[0]
    
    def main_feedback_loop(self, ppr_k_value):
        idx = self.file_idx_map[self.query_id]
        q_vect, q_prob = np.array(self.vectors[idx]).reshape((1,-1)), np.array(self.vectors[idx]).reshape((1,-1))
        ppr_sim_matrix_vect = self.get_sim_matrix()
        ppr_sim_matrix_prob = self.get_sim_matrix()
        relevant = non_relevant = None

        iter = 1
        while True:
            # call retrieval function
            print("Getting results")
            results = {}
            # ppr results collection
            results['ppr_vect'] = self.process_ppr(ppr_sim_matrix_vect, idx+1, self.top_retrieve, ppr_k_value)
            results['ppr_prob'] = self.process_ppr(ppr_sim_matrix_prob, idx+1, self.top_retrieve, ppr_k_value)
            # prob results collection
            results['prob_vect'] = self.prob_retrieval(q_vect)
            results['prob_prob'] = self.prob_retrieval(q_prob)
            
            # Choose a result
            k = self.choose_result_set(results)
            chosen_result = [self.idx_file_map[x] for x in results[k]]

            # call user feedback function
            print("Getting feedback")
            self.display.result_and_feedback(self.query_id, chosen_result)
            relevant = [self.file_idx_map[x] for x in self.display.relevant]
            non_relevant = [self.file_idx_map[x] for x in self.display.irrelevant]
            
            # call query mod function
            if "_vect" in k:
                q_vect = self.query_optimization_vectors(q_vect, relevant, non_relevant)
                q_prob = self.query_optimization_prob(q_vect, relevant, non_relevant)
            else:
                q_vect = self.query_optimization_vectors(q_prob, relevant, non_relevant)
                q_prob = self.query_optimization_prob(q_prob, relevant, non_relevant)
            
            # PPR update matrix with new similarty values
            self.modify_sim_matrix(ppr_sim_matrix_prob, q_prob, idx)
            self.modify_sim_matrix(ppr_sim_matrix_vect, q_vect, idx)
            # check for uc value
            uc = input("Continue? (y/n) : ")
            if uc == 'n' or uc == 'N':
                break
            
            iter += 1

if __name__ == "__main__":
    input_dir = "phase2_outputs"
    vm = 1
    ppr_k = int(input("Enter a value K for outgoing gestures in PPR : "))
    task6 = Task6(input_dir, vm=vm, uc=2)
    task6.main_feedback_loop(ppr_k)