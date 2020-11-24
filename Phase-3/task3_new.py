import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from scipy.spatial import distance
import json
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity


class LSH:
    def __init__(self, L, k, input_dir, vm=1, uc=6):
        self.L = L
        self.k = k
        self.vm = vm
        self.uc = uc
        self.input_dir = input_dir
        self.mode = 'tf' if vm == 1 else 'tfidf'
        self.data = None
        self.hash_tables = []
        self.n_words = None
        self.output_dir = os.path.join("outputs", "task3")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.file_list = sorted(glob.glob('{}/task0b/{}_*.txt'.format(input_dir, self.mode)))
        self.idx_2_file = {i: fname.split("_")[-1].split(".")[0] for i, fname in enumerate(self.file_list)}
        self.file_2_idx = {fname.split("_")[-1].split(".")[0]: i for i, fname in enumerate(self.file_list)}

    def get_random_vectors(self):
        """Generates a random array of num_words*k dimension"""
        return np.random.randn(self.n_words, self.k)

    def load_data(self):
        names = {2: "pca_{}_vectors.txt",
                 3: "svd_{}_vectors.txt",
                 4: "nmf_{}_vectors.txt",
                 5: "lda_{}_vectors.txt"}
        if self.uc in [2, 3, 4, 5]:
            self.data = np.array(
                json.loads(json.load(open(os.path.join(self.input_dir, "task1", names[self.uc].format(self.vm)), "r"))))
        else:
            self.data = np.array([json.loads(json.load(open(fname, 'r'))) for fname in self.file_list])
        self.n_words = len(self.data[0])

    def binary_2_integer(self, binary_vectors):
        exponents = np.array([2 ** i for i in range(self.k - 1, -1, -1)])
        return binary_vectors.dot(exponents)

    def hash_data(self, data, random_vectors):
        if (len(data.shape) == 1):
            data = data.reshape(1, data.shape[0])
        binary_repr = data.dot(random_vectors) >= 0
        # binary_inds = self.binary_2_integer(binary_repr)
        binary_string_arr = []
        for idx, binaryArr in enumerate(binary_repr.astype(int).astype(str)):
            binary_string_arr.append(str.encode(''.join(binaryArr)))
        return binary_string_arr

    def train(self):
        self.load_data()
        for i in range(self.L):
            random_vectors = self.get_random_vectors()
            binary_inds = self.hash_data(self.data, random_vectors)
            table = defaultdict(list)
            for idx, bin_ind in enumerate(binary_inds):
                table[bin_ind].append(idx)
            hash_table = {'random_vectors': random_vectors, 'table': table}
            self.hash_tables.append(hash_table)

    def query(self, data_point, max_results):
        retrieval = set()
        n_buckets = 0
        n_candidates = 0
        permLevel = 0

        while (len(retrieval) < max_results):
            for hash_table in self.hash_tables:
                table = hash_table['table']
                random_vectors = hash_table['random_vectors']
                binary_idx = self.hash_data(data_point, random_vectors)[0]

                # Neighbors of permutations of the original hash
                if (permLevel > 0):
                    for hash_table_idx in table.keys():
                        # counting changed bits
                        xorNum = bin(int(binary_idx, 2) ^ int(hash_table_idx, 2))[2:]
                        bitChanges = xorNum.encode().count(b'1')

                        # if this is the current permutation level desired
                        if (bitChanges == permLevel):
                            n_buckets += 1
                            retrieval.update(table[hash_table_idx])

                        if (len(retrieval) >= max_results):
                            break

                    if (len(retrieval) >= max_results):
                        break
                # Original hash neighbors
                else:
                    if (table[binary_idx]):
                        n_buckets += 1
                        retrieval.update(table[binary_idx])

            permLevel += 1

            # No More Permutations left
            if (permLevel == self.k):
                break

        retrieval = list(retrieval)
        sim_scores = cosine_similarity(np.expand_dims(data_point, 0), self.data[retrieval]).ravel()
        data_idx = sim_scores.argsort()[::-1][:max_results]
        assert len(retrieval) == len(sim_scores)
        result = {'n_buckets': n_buckets, 'n_candidates': len(retrieval), 'scores': sim_scores[data_idx],
                  'retrieved_files': [self.idx_2_file[retrieval[d]] for d in data_idx],
                  'retrieved_idx': [retrieval[d] for d in data_idx]}
        return result
        # json.dump(result, open(self.output_dir + "/{}_{}_dominant_{}_{}.txt".format(k, m, self.vm, self.uc), "w"),
                #   indent="\t")


    def loadDataMatrix(self,input_dir, vm, uc):
        names = {2: "pca_{}_vectors.txt",
                 3: "svd_{}_vectors.txt",
                 4: "nmf_{}_vectors.txt",
                 5: "lda_{}_vectors.txt"}
        mode = 'tf' if vm == 1 else 'tfidf'
        file_list = glob.glob(input_dir + '/task0b/{}_*.txt'.format(mode))
        data = dict()
        if uc in [2, 3, 4, 5]:
            vectors = np.array(json.loads(json.load(open(os.path.join(input_dir, "task1", names[uc].format(vm)), "r"))))
            for i, fname in enumerate(sorted(file_list)):
                fileNum = self.file_2_idx[fname.split("_")[-1].split('.')[0]]
                data[fileNum] = vectors[i]
        else:
            for fname in sorted(file_list):
                fileNum = self.file_2_idx[fname.split("_")[-1].split('.')[0]]
                data[fileNum] = json.loads(json.load(open(fname, 'r')))
        return data


if __name__ == "__main__":
    L = 4 #int(input("Enter Number of Layers : "))
    k = 8#int(input("Enter K : "))
    # vm = int(input("Enter vm (1 - tf, 2 - tfidf) : "))
    # uc = int(input("Enter uc (2 - pca, 3 - svd, 4 - nmf, 5 - lda, 6 - tf, 7 - tfidf ) : "))
    # inputDir = input("Enter input directory : ")
    inputDir = 'phase2_outputs'
    query_file = []
    for vm in [1,2]:
        for uc in [2,3,4,5,6,7]:
            lsh = LSH(L=L, k=k, input_dir=inputDir, vm=vm, uc=uc)
            lsh.train()
            data = lsh.loadDataMatrix(inputDir, vm, uc)
            print("Hashing Complete!\n")
            for queryFile in [6,257,565]:
                queryFile = lsh.file_2_idx[str(queryFile).zfill(3)]
                maxResults = 10 #int(input("Enter maximum results : "))
                lsh.query(np.array(data[queryFile]), max_results=maxResults)
