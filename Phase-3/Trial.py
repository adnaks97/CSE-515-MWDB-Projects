import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
frame = pd.read_excel("sample_training_labels.xlsx", header=None)
j = np.array(json.loads(json.load(open("phase2_outputs/task1/pca_2_vectors.txt", "r"))))
files = list(frame[0]) 

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
        print("Removed indices", self.remove_indices_mat)
        self.reduced_index = dict(zip(list(range(len(self.train_file_num_indices))),self.train_file_num_indices))
    
    def knn(self):
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
        
        def find_neighbors(point, data, labels, k=3):
            n_of_dimensions = len(point)
            neighbors = []
            neighbor_labels = []
            for i in range(0, k):
                nearest_neighbor_id = None
                smallest_distance = None
                for i in range(0, len(data)):
                    eucledian_dist = 0
                    for d in range(0, n_of_dimensions):
                        dist = abs(point[d] - data[i][d])
                        eucledian_dist += dist
                    eucledian_dist = np.sqrt(eucledian_dist)
                    if smallest_distance == None:
                        smallest_distance = eucledian_dist
                        nearest_neighbor_id = i
                    elif smallest_distance > eucledian_dist:
                        smallest_distance = eucledian_dist
                        nearest_neighbor_id = i
                neighbors.append(data[nearest_neighbor_id])
                neighbor_labels.append(labels[nearest_neighbor_id])

                data=np.delete(data,data[nearest_neighbor_id])
                labels=np.delete(labels,labels[nearest_neighbor_id])
            return neighbor_labels

        def k_nearest_neighbor(point, data, labels, k=3):
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
        
        file = np.array(json.loads(json.load(open("phase2_outputs/task1/pca_2_vectors.txt", "r"))))
        targets = {'vattene':0, 'combinato':1, 'daccordo':2}
        key_list = list(targets.keys()) 
        val_list = list(targets.values()) 
        X = np.array([list(file[i]) for i in self.train_file_num_indices])
        labels = np.array([self.class_labels_map[k] for k in self.train_file_num])
        y = np.array([targets[k] for k in labels])       
        result = {}
        for r in self.remove_indices_mat:
            feat = list(file[r])
            result[self.idx_file_map[r]] = {}
            ypred=k_nearest_neighbor(feat, X, y, k=5)
            result[self.idx_file_map[r]]['predicted'] = key_list[val_list.index(ypred[0])]
            result[self.idx_file_map[r]]['original'] = self.all_files_classes[self.idx_file_map[r]]
               
if __name__ == "__main__":
    print("Performing Task 2")
    input_directory = "phase2_outputs" #input("Enter directory to use: ")
    knn_k = 5 #int(input("Enter a value K for KNN : "))
    ppr_k = 30 #int(input("Enter a value K for outgoing gestures (PPR) : "))
    m_value = 11 #int(input("Enter a value M for most dominant gestures : "))
    task2 = Trial(input_directory)
    task2.knn()