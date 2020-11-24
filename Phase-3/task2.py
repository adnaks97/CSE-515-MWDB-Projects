import os
import numpy as np
import json
from numpy import random
import pandas as pd
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance
from scipy.stats import mode
# from numpy.linalg import multi_dot
from scipy.spatial import distance

class Task2:
    def __init__(self, input_dir, vm=2, uc=2):
        self.vm = vm
        self.uc = uc
        # setting up inout and output directories
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.join("outputs", "task2")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        # creating map for files and indices
        files = sorted([x.split(".")[0] for x in os.listdir(os.path.join(self.input_dir, "task0a")) if ".wrd" in x])
        indices = list(range(0, len(files)))
        self.idx_file_map = dict(zip(indices, files))
        self.file_idx_map = dict(zip(files, indices))
        # creating map for train and test gesture files v/s indices
        train_labels = pd.read_excel("sample_training_labels_new.xlsx", header=None)
        all_labels = pd.read_excel("all_labels_new.xlsx", header=None)
        self.all_files_classes = dict(zip(all_labels.iloc[:,0].apply(lambda x: str(x).zfill(7)).tolist(),all_labels.iloc[:,1].tolist()))
        self.train_file_num = train_labels.iloc[:, 0].apply(lambda x: str(x).zfill(7)).tolist()
        self.class_labels_map = dict(zip(self.train_file_num,train_labels.iloc[:,1].tolist()))
        self.remove_indices_mat = list(set(files) - set(self.train_file_num))
        self.train_file_num_indices = sorted([self.file_idx_map[x] for x in self.train_file_num])
        self.remove_indices_mat = sorted([self.file_idx_map[x] for x in self.remove_indices_mat])
        self.reduced_index = dict(zip(list(range(len(self.train_file_num_indices))),self.train_file_num_indices))

    @staticmethod
    def get_knn_nodes(sim_matrix, k=3):
        k = sim_matrix.shape[1] - k
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
        mat = np.array(json.loads(json.load(open(os.path.join(self.input_dir, "task3", names[self.uc].format(self.vm)), "r"))))
        return mat

    def get_vectors(self):
        names =  {1: "pca_{}_vectors.txt",
                  2: "svd_{}_vectors.txt",
                  3: "nmf_{}_vectors.txt", 
                  4: "lda_{}_vectors.txt"}
        vec = np.array(json.loads(json.load(open(os.path.join(self.input_dir, "task1", names[self.uc].format(self.vm)), "r"))))
        return vec

    @staticmethod
    def ppr_process(adj_matrix_norm, seed_nodes, c=0.8):
        size = adj_matrix_norm.shape[0]
        idx = random.choice(seed_nodes)
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        u_old[idx - 1, 0] = 1
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        for x in seed_nodes:
            v[x, 0] = 1/len(seed_nodes)
        A = adj_matrix_norm
        diff = 1
        icnt = 0
        while diff > 1e-20 and icnt < 100:
            u_new = ((1 - c) * np.matmul(A, u_old)) + (c * v)
            diff = distance.minkowski(u_new, u_old, 1)
            u_old = u_new
            icnt += 1
        # u_new[seed_nodes] = 0.
        # u_new = (u_new - np.min(u_new)) / (np.max(u_new) - np.min(u_new) + 1e-7) #u_new.sum(axis=0, keepdims=1) + 1e-7)
        return u_new
    
    def run_new_ppr(self, m, k):
        sim_matrix = self.get_sim_matrix()
        adj_matrix = self.get_knn_nodes(sim_matrix, k)
        adj_matrix_norm = self.normalize(adj_matrix)
        cl_scores = []
        un_classes = np.unique(list(self.class_labels_map.values()))
        for cls in un_classes:
            files = [x[0] for x in self.class_labels_map.items() if x[1] == cls]
            ind = [self.file_idx_map[x] for x in files]
            cl_scores.append(self.ppr_process(adj_matrix_norm, ind)[self.remove_indices_mat].tolist())
        cl_scores = np.array(cl_scores).reshape((len(un_classes),-1))
        labels = np.argmax(cl_scores, axis=0).tolist()
        pred_labels = un_classes[labels]
        result = {}
        acc = 0.
        for i,f in enumerate(self.remove_indices_mat):
            file = self.idx_file_map[f]
            label = pred_labels[i]
            act_label = self.all_files_classes[file]
            result[file] = {}
            result[file]['pred'] = label
            result[file]['true'] = act_label
            if label == act_label:
                acc += 1
        acc = acc/len(self.remove_indices_mat)
        result['$Accuracy'] = acc
        json.dump(dict(sorted(result.items())), open(self.output_dir + "/{}_{}_dominant_{}_{}.txt".format(k, m, self.vm, self.uc), "w"), indent="\t")


    @staticmethod
    def process_ppr(adj_matrix_norm, idx, rem_indices, file_name, c=0.8):
        size = adj_matrix_norm.shape[0]
        u_old = np.zeros(size, dtype=float).reshape((-1, 1))
        u_old[idx - 1, 0] = 1
        v = np.zeros(size, dtype=float).reshape((-1, 1))
        v[idx - 1, 0] = 1
        A = adj_matrix_norm
        diff = 1
        icnt = 0
        while diff > 1e-20 and icnt < 100:
            u_new = ((1 - c) * np.matmul(A, u_old)) + (c * v)
            diff = distance.minkowski(u_new, u_old, 1)
            u_old = u_new
            icnt += 1
        u_new[rem_indices] = -0.001
        result = (file_name, idx, u_new)
        return result        

    def preprocess_ppr_task2(self, m, k):
        sim_matrix = self.get_sim_matrix()
        adj_matrix = self.get_knn_nodes(sim_matrix, k)
        adj_matrix_norm = self.normalize(adj_matrix)
        res = []
        pool = ThreadPool(12000)
        for index in self.remove_indices_mat:
            file_name = self.idx_file_map[index]
            res.append(pool.apply_async(self.process_ppr, args=(adj_matrix_norm, index, self.remove_indices_mat, file_name, 0.65)).get())
        pool.close()
        pool.join()
        
        result = {}
        for n_value in res:
            u_new = n_value[2]
            files = np.array([self.idx_file_map[x] for x in u_new.ravel()[:-1].argsort()[::-1][:m]])
            scores = np.array(sorted(u_new.ravel())[:-1][::-1][:m])
            classes = np.array([self.class_labels_map[x] for x in files])
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
            scores = [np.sum(result[f]['scores'][result[f]['classes']==c]) for c in un_clss]
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
            final_result['acc'] = {}
            final_result['acc']['voting'] = voting_acc/len(result)
            final_result['acc']['wt_scores'] = wt_scor_accuracy/len(result)
        json.dump(final_result, open(self.output_dir + "/{}_{}_dominant_{}_{}.txt".format(k, m, self.vm, self.uc), "w"), indent="\t")
    
    def decision_tree(self):
        file = self.get_vectors()
        targets = {'vattene':0, 'combinato':1, 'daccordo':2}
        key_list = list(targets.keys()) 
        val_list = list(targets.values()) 
        X = np.round(np.array([list(file[i]) for i in self.train_file_num_indices]), decimals=4)
        labels = np.array([self.class_labels_map[k] for k in self.train_file_num])
        y = np.array([targets[k] for k in labels])
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(X, y)      
        result = {}
        for r in self.remove_indices_mat:
            feat = list(file[r])
            result[self.idx_file_map[r]] = {}
            ypred = clf.predict([feat])
            result[self.idx_file_map[r]]['predicted'] = key_list[val_list.index(ypred[0])]
            result[self.idx_file_map[r]]['original'] = self.all_files_classes[self.idx_file_map[r]]
        count=0
        for k in result:
            if result[k]['predicted'] == result[k]['original']:
                count+=1
        result["Accuracy"] = (count/len(result))*100
        json.dump(result, open(self.output_dir + "/decision_tree_{}_{}.txt".format(self.vm, self.uc), "w"), indent="\t")
    
    def knn(self,k):
         
        def most_found(array):
            list_of_words = []
            for i in range(len(array)):
                if array[i] not in list_of_words:
                    list_of_words.append(array[i])
            most_counted = ''
            n_of_most_counted = None
            for i in range(len(list_of_words)):
                counted = array.count(list_of_words[i])
                if n_of_most_counted == None:
                    most_counted = list_of_words[i]
                    n_of_most_counted = counted
                elif n_of_most_counted < counted:
                    most_counted = list_of_words[i]
                    n_of_most_counted = counted
                elif n_of_most_counted == counted:
                    most_counted = None
            return most_counted
        
        def find_neighbors(point, data, labels, k):
            n_of_dimensions = len(point)
            neighbors = []
            neighbor_labels = []
            try:
                # mahalobonis
                dist = distance.squareform(distance.pdist(data, metric='mahalanobis', VI=None))
            except:
                # do euclidean
                dist = distance.squareform(distance.pdist(data, metric='euclidean'))
            dist = dist[-1,:-1]
            neigh_indices = np.argsort(dist)[:k]
            neighbour_labels = labels[neigh_indices].ravel()
            return neighbour_labels.tolist()

        def k_nearest_neighbor(point, data, labels, k):
                # If two different labels are most found, continue to search for 1 more k
            while True:
                neighbor_labels = find_neighbors(point, data, labels, k=k)
                label = most_found(neighbor_labels)
                if label != None:
                    break
                k += 1
                if k >= len(data):
                    break
            return label
        file = self.get_vectors()
        targets = {'vattene':0, 'combinato':1, 'daccordo':2}
        key_list = list(targets.keys()) 
        val_list = list(targets.values())
        X = np.round(np.array([list(file[i]) for i in self.train_file_num_indices]),decimals=4)
        labels = np.array([self.class_labels_map[k] for k in self.train_file_num])
        y = np.array([targets[k] for k in labels]).reshape((-1,1))     
        result = {}
        for r in self.remove_indices_mat:
            feat = list(file[r])
            result[self.idx_file_map[r]] = {}
            feat_list=np.array(feat)
            X_new=np.vstack((X,feat_list))
            ypred=k_nearest_neighbor(feat, X_new, y, k)
            result[self.idx_file_map[r]]['predicted'] = key_list[val_list.index(ypred)]
            result[self.idx_file_map[r]]['original'] = self.all_files_classes[self.idx_file_map[r]]
        count=0
        for k in result:
            if result[k]['predicted'] == result[k]['original']:
                count+=1
        result["Accuracy"] = (count/len(result))*100
        json.dump(result, open(self.output_dir + "/knn_{}_{}.txt".format(self.vm, self.uc), "w"), indent="\t")
    
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

if __name__ == "__main__":
    print("Performing Task 2")
    input_directory = "phase2_outputs" #input("Enter directory to use: ")
    knn_k =  int(input("Enter a value K for KNN : "))
    ppr_k = int(input("Enter a value K for outgoing gestures (PPR) : "))
    m_value = int(input("Enter a value M for most dominant gestures : "))
    algo = int(input("Enter the algorithm to use (1-PPR \t2.Decision Tree \t3.KNN ) : "))
    task2 = Task2(input_directory, 2, 2)
    if algo == 1:
        # old ppr
        task2.preprocess_ppr_task2(m_value, ppr_k)
        # new ppr
        # task2.run_new_ppr(m_value, ppr_k)
    elif algo == 2:
        task2.decision_tree()
    else:
        task2.knn(knn_k)
    # for vm in [1,2]:
    #     for uc in [2,3,4,5]:
    #         for ppr_k in [20,25,30]:
    #             print("PPR with vm {} and uc {}".format(vm, uc))
    #             task2 = Task2(input_directory, vm, uc)
    #             # task2.preprocess_ppr_task2(m_value, ppr_k)
    #             task2.run_new_ppr(m_value, ppr_k)
    #         # print("PPR Done")
    #     for uc in [1, 2, 3, 4]:
    #         print("DT and KNN with vm {} and uc {}".format(vm, uc))
    #         task2 = Task2(input_directory, vm, uc)
    #         task2.decision_tree()
    #         task2.knn(knn_k)