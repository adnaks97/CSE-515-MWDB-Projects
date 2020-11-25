import os
import json
import math
import copy
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from heapq import nlargest
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns


class Task4:
    def __init__(self, input_dir, vm=2):
        self.vm = vm
        # setting up inout and output directories
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join("outputs", "task4")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # creating map for files and indices
        files = sorted([x.split(".")[0] for x in os.listdir(os.path.join(self.input_dir, "task0a")) if ".wrd" in x])
        indices = list(range(0, len(files)))
        self.idx_file_map = dict(zip(indices, files))
        self.file_idx_map = dict(zip(files, indices))
        self.all_words_idx = pkl.load(open(os.path.join(self.input_dir,"task0b","all_words_idx.txt"), "rb"))
        wrds = list(self.all_words_idx.keys())
        c = []
        for w in wrds:
            c.append((w[0],w[1],*tuple(w[2])))
        self.all_idx_words = dict(zip(list(self.all_words_idx.values()),c))
        self.diff = []
        self.change_in_sensor = []
        self.change_in_comp = []
        self.change_in_words = []
        self._get_file_vectors_()
    
    def _get_file_vectors_(self):
        self.tf_vectors, self.tfidf_vectors = [], []
        # load tf vectors
        files = sorted([os.path.join(self.input_dir, "task0b",x) for x in os.listdir(os.path.join(self.input_dir, "task0b")) if "tf_vectors" in x])
        for f in files:
            self.tf_vectors.append(np.array(json.loads(json.load(open(f,"r")))).reshape((-1,1)))
        # load tfidf vectors
        files = sorted([os.path.join(self.input_dir, "task0b",x) for x in os.listdir(os.path.join(self.input_dir, "task0b")) if "tfidf_vectors" in x])
        for f in files:
            self.tfidf_vectors.append(np.array(json.loads(json.load(open(f,"r")))).reshape((-1,1)))
    
    def _get_user_feedback_(self, results):
        rel, non_rel = [], []
        for i,r in enumerate(results,1):
            # fn = int(self.idx_file_map[r].lstrip("0").split("-")[0])
            # if 200 <= fn <= 300:
            #     rel.append(r)
            # else:
            #     non_rel.append(r)
            print("{} : File ID {}".format(i,self.idx_file_map[r]))
            fd = input("Enter your feedback(1-Correct/0-Wrong) : ")
            if fd == '1':
                rel.append(r)
            elif fd == '0':
                non_rel.append(r)
        return rel, non_rel
    
    def query_optimization(self, Q, relevant, non_relevant):
        beta = 1/len(relevant) if len(relevant)>0 else 0
        gamma = 1/len(non_relevant) if len(non_relevant)>0 else 0
        
        dr, dnr = 0, 0
        #rel_files_vectors-contains vectors of all relevant file ids
        rel_files_vectors, non_rel_files_vectors = [], []

        for j in relevant:
            rel_files_vectors.append(self.tf_vectors[j])
        
        a=[]
        for c in range(len(rel_files_vectors)):
            mag = math.sqrt(sum(pow(element, 2) for element in rel_files_vectors[c]))
            l = [e/mag for e in rel_files_vectors[c]]
            a.append(l)
        dr = np.sum(np.array(a), axis=0)
        
        #non relevant files
        for j in non_relevant:
            non_rel_files_vectors.append(self.tf_vectors[j]) 
        an=[]
        for c in range(len(non_rel_files_vectors)):
            mag = math.sqrt(sum(pow(element, 2) for element in non_rel_files_vectors[c]))
            l = [e/mag for e in non_rel_files_vectors[c]]
            an.append(l)
        dnr = np.sum(np.array(an), axis=0)
        
        optimized_q = Q + beta*dr - gamma*dnr
        return optimized_q

    def relative_importance(self, D1, D2):
        
        def get_list_items(indices, D1, D2):
            retD1 = {}
            retD2 = {}
            for k in indices:
                retD1[k], retD2[k] = [], []
                for x in indices[k]:
                    retD1[k].append(D1[x])
                    retD2[k].append(D2[x])
            return retD1, retD2

        def change_in_vector(d1, d2):
            diff = {}
            change = []
            for k in d1:
                c = np.sum(np.array(d1[k]) - np.array(d2[k])).ravel()[0]
                diff[k]=c
                change.append(c)
            newDiff = dict(sorted(diff.items(), key=lambda x: x[1], reverse=True))
            minmax = MinMaxScaler()
            change = minmax.fit_transform(np.array(change).reshape((-1,1)))
            return newDiff, change
        
        sensors = {}
        for k in range(20):
            i = [self.all_words_idx[key] for key in self.all_words_idx if key[1]==k]
            sensors[k] = i
        
        comp_names = ['words_X', 'words_Y', 'words_W', 'words_Z']
        comp_vector = {}
        for c in comp_names:
            i = [self.all_words_idx[key] for key in self.all_words_idx if key[0]==c]
            comp_vector[c] = i
        
        sd1, sd2 = get_list_items(sensors, D1, D2)
        cd1, cd2 = get_list_items(comp_vector, D1, D2)
        diff_s, sensor_change = change_in_vector(sd1, sd2)
        diff_c, comp_change = change_in_vector(cd1, cd2)
        D1 = D1.reshape((-1,))
        word_change = np.array(D1) - np.array(D2)
        minmax = MinMaxScaler()
        word_change = minmax.fit_transform(np.array(word_change).reshape((-1, 1)))
        diff_w_indices = np.argsort((np.array(D1) - np.array(D2)))[::-1]
        difference = {}
        difference['sensors'] = diff_s
        difference['components'] = diff_c
        # difference['words'] = diff_w_indices
        # difference['words'] = [self.all_idx_words[w] for w in difference['words'].ravel().tolist()]
        self.change_in_sensor.append(sensor_change.tolist())
        self.change_in_comp.append(comp_change.tolist())
        self.change_in_words.append(word_change.tolist())
        self.diff.append(difference)
    
    def new_prob_retrieval(self, q, f_c, f_w):
        q = q.reshape((1,-1))
        sim = {}
        vectors = copy.deepcopy(self.tf_vectors) if self.vm == 1 else copy.deepcopy(self.tfidf_vectors)
        vectors = np.array(vectors).reshape((-1,len(vectors[0])))
        if f_c is not None and f_w is not None:
            q_new = []
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
            q_new = q_new.reshape((-1,))

        else:
            n_i = np.count_nonzero(vectors, axis=0).astype(np.float32)
            p_i = np.array([0.5]*len(vectors[0])).reshape((1,-1))
            u_i = n_i/len(vectors)
            u_i[u_i<=0] = 1e-5
            u_i[u_i>=1] = 0.99
            x = u_i*(1-p_i)
            y = p_i*(1-u_i)
            q_new = np.log((p_i*(1-u_i))/(u_i*(1-p_i)))
            q_new = q_new.reshape((-1,))

        for j in range(len(vectors)):
            sim[j] = np.sum(np.multiply(vectors[j],q_new))

        top_10_scores = dict(sorted(sim.items(), key=lambda x: x[1], reverse=True)[:10])
        return q_new, top_10_scores

    def plot_heatmap(self, q_file):

        fig1 = plt.figure(figsize=(14, 8))
        sns.heatmap(np.array(self.change_in_sensor).reshape((-1,len(self.change_in_sensor[0]))), cmap="gray", xticklabels=False)
        fig1.savefig(os.path.join(self.output_dir, "sensor_change_{}_{}.png".format(q_file, self.vm)))

        fig2 = plt.figure(figsize=(14, 8))
        sns.heatmap(np.array(self.change_in_comp).reshape((-1,len(self.change_in_comp[0]))), cmap="gray", xticklabels=False)
        fig2.savefig(os.path.join(self.output_dir, "comp_change_{}_{}.png".format(q_file, self.vm)))        

        fig3 = plt.figure(figsize=(14, 8))
        sns.heatmap(np.array(self.change_in_words).reshape((-1,len(self.change_in_words[0]))), cmap="gray", xticklabels=False)
        fig3.savefig(os.path.join(self.output_dir, "words_change_{}_{}.png".format(q_file, self.vm)))        

    def main_feedback_loop(self, query_file):
        self.acc_measure = []
        idx = self.file_idx_map[query_file.zfill(7)]
        if self.vm == 1:
            q = self.tf_vectors[idx]
        else:
            q = self.tfidf_vectors[idx]
        relevant = non_relevant = None
        q = np.array([item for sublist in q for item in sublist])
        while True:
            # call retrieval function
            print("Getting results")
            q_new, results = self.new_prob_retrieval(q, relevant, non_relevant)
            q_change = np.count_nonzero(q_new - q)
            print("Changed dimensions {}".format(q_change))
            # call user feedback function
            print("Getting feedback")
            relevant, non_relevant = self._get_user_feedback_(results)
            self.acc_measure.append(len(relevant)/10.)
            # call query mod function
            print("Relative importance")
            self.relative_importance(q, q_new)
            # check for uc value
            uc = input("Are you done? (y/n) : ")
            if uc == 'y' or uc == 'Y':
                self.plot_heatmap(query_file)              
                break
            q = q_new
        fig = plt.figure(figsize=(8,8))
        plt.plot(self.acc_measure)
        plt.xlabel("Iterations")
        plt.ylabel("Relevance")
        plt.grid()
        fig.savefig(self.output_dir+"/{}_{}_relevance_plot.png".format(query_file, self.vm))
        json.dump(self.diff, open(os.path.join(self.output_dir,"{}_{}_query_difference.txt".format(self.vm,query_file)),"w"), indent="\t")

if __name__ == "__main__":
    input_dir = "phase2_outputs"
    vm = 2 #int(input("Which vector model to use? (tf-1/ tfidf-2) : "))
    file = input("Enter a file number : ").zfill(7)
    task4 = Task4(input_dir, vm=vm)
    task4.main_feedback_loop(file)