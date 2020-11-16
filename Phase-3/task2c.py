import numpy as np

import pandas as pd
import json
import os
from pathlib import Path
from sklearn.preprocessing import normalize
from multiprocessing.dummy import Pool as ThreadPool
from scipy.spatial import distance
from task1 import Task1
from scipy.stats import mode
from pprint import pprint
from sklearn import preprocessing

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

class Trial:
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
        print("training: ", self.train_file_num_indices)
        print("labels: ", self.train_file_num)
        print(self.class_labels_map)
        
    
    def decision_tree(self):
        file = np.array(json.loads(json.load(open("phase2_outputs/task1/nmf_2_vectors.txt", "r"))))
        print(file)
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
        print("total", len(result))
        print("correct", count)  
        print("accuracy:", count/len(result))
    
    
if __name__ == "__main__":
    print("Performing Task 2")
    input_directory = "phase2_outputs" #input("Enter directory to use: ")
    knn_k = 5 #int(input("Enter a value K for KNN : "))
    ppr_k = 30 #int(input("Enter a value K for outgoing gestures (PPR) : "))
    m_value = 11 #int(input("Enter a value M for most dominant gestures : "))
    task2 = Trial(input_directory)
    task2.decision_tree()
    
#     task1 = Task1(input_directory)
#     task2.preprocess_ppr_task2(m_value, ppr_k)  