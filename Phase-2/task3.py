from pathlib import Path
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD, NMF
import json
import numpy as np
import os
import pickle as pkl

class Task3:
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        self.task0a_dir = os.path.join(self.dir, "task0a")
        self.task0b_dir = os.path.join(self.dir, "task0b")
        self.task1_dir = os.path.join(self.dir, "task1")
        self.pca = self.nmf = self.lda = self.svd = self.model = None
        self.out_dir = os.path.join(self.dir, "task3")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.compNames = ['words_X', 'words_Y', 'words_Z', 'words_W']
        self.file_paths = sorted([os.path.join(self.task0a_dir, f) for f in os.listdir(self.task0a_dir) if ".wrd" in f])
        self.tf_files = sorted([os.path.join(self.task0b_dir, k) for k in os.listdir(os.path.join(self.task0b_dir)) if "tf_" in k and ".txt" in k])
        self.tfidf_files = sorted([os.path.join(self.task0b_dir, k) for k in os.listdir(os.path.join(self.task0b_dir)) if "tfidf_" in k and ".txt" in k])
        self.sequences = {}
        self.tf, self.tfidf, self.entropy, self.scores_final = [], [], [], []
        self.file_idx, self.idx_file = {}, {}
        self._read_wrd_files_()
        self._load_all_vectors_()

    def _load_all_vectors_(self):
        for fname in self.tf_files:
            name = fname.split("/")[-1].split("_")[-1].split(".")[0]
            idx = len(self.file_idx.keys())
            self.file_idx[name] = idx
            self.idx_file[idx] = name
            self.tf.append(json.loads(json.load(open(fname, "r"))))
            x = np.array(self.tf[-1])
            self.entropy.append(np.multiply(-x, np.log2(x+1e-7)))

        self.tfidf = []
        for fname in self.tfidf_files:
            self.tfidf.append(json.loads(json.load(open(fname, "r"))))

        self.tf = np.array(self.tf).reshape((len(self.tf_files), -1))
        self.tfidf = np.array(self.tfidf).reshape((len(self.tfidf_files), -1))
        self.entropy = np.array(self.entropy).reshape((len(self.tfidf_files), -1))

        if os.path.exists(os.path.join(self.task1_dir, "pca_vectors.txt")):
            self.pca = np.array(json.loads(json.load(open(os.path.join(self.task1_dir, "pca_vectors.txt"), "r"))))
        if os.path.exists(os.path.join(self.task1_dir, "svd_vectors.txt")):
            self.svd = np.array(json.loads(json.load(open(os.path.join(self.task1_dir, "svd_vectors.txt"), "r"))))
        if os.path.exists(os.path.join(self.task1_dir, "nmf_vectors.txt")):
            self.nmf = np.array(json.loads(json.load(open(os.path.join(self.task1_dir, "nmf_vectors.txt"), "r"))))
        if os.path.exists(os.path.join(self.task1_dir, "lda_vectors.txt")):
            self.lda = np.array(json.loads(json.load(open(os.path.join(self.task1_dir, "lda_vectors.txt"), "r"))))

        self.allWords = pkl.load(open(os.path.join(self.task0b_dir, "all_words_idx.txt"), "rb"))

    def _read_wrd_files_(self):
        for fp in self.file_paths:
            data = json.load(open(fp, "r"))
            file_name = fp.split("/")[-1].split(".")[0]
            self.sequences[file_name] = {}
            for c in self.compNames:
                self.sequences[file_name][c] = {}
                items = data[c]
                for sid in range(20):
                    words = [(it[1], it[2]) for it in items if it[0][1] == sid]
                    self.sequences[file_name][c][sid] = words

    def _construct_list_for_mp_(self, fn1, fn2):
        all_lists = []
        for c in self.compNames:
            for sid in range(20):
                all_lists.append((fn1, fn2, c, sid, self.sequences[fn1][c][sid], self.sequences[fn2][c][sid]))
        return all_lists

    def _edit_distance_(self, seqs):
        seq1 = seqs[4]
        seq2 = seqs[5]
        word1 = [item[0] for item in seq1]
        word2 = [item[0] for item in seq2]
        f1 = self.file_idx[seqs[0]]
        f2 = self.file_idx[seqs[1]]
        comp = seqs[2]
        sensor = seqs[3]

        n = len(word1)
        m = len(word2)

        # if one of the strings is empty
        if n * m == 0:
            return n + m

        # array to store the conversion history
        d = [[0] * (m + 1) for _ in range(n + 1)]

        # init boundaries
        for i in range(1, n + 1):
            d[i][0] = d[i-1][0] + self.entropy[f1, self.allWords[(comp, sensor, tuple(word1[i-1]))]]
        for j in range(1, m + 1):
            d[0][j] = d[0][j-1] + self.entropy[f2, self.allWords[(comp, sensor, tuple(word2[j-1]))]]

        # DP compute
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                e1 = self.entropy[f1, self.allWords[(comp, sensor, tuple(word1[i - 1]))]]
                e2 = self.entropy[f2, self.allWords[(comp, sensor, tuple(word2[j - 1]))]]
                cost = abs(e1 - e2)
                d[i][j] = min(d[i-1][j]+cost, d[i][j-1]+cost, d[i-1][j-1] if word1[i-1] == word2[j-1] else d[i-1][j-1]+cost)
        return d[n][m]

    def _dtw_distance_(self, seqs):
        seq1 = seqs[4]
        seq2 = seqs[5]
        word1 = [item[1] for item in seq1]
        word2 = [item[1] for item in seq2]

        n = len(word1)
        m = len(word2)

        dtw_matrix = [[0] * (m + 1) for i in range(n + 1)]

        for i in range(1, n + 1):
            dtw_matrix[i][0] = dtw_matrix[i - 1][0] + abs(word1[i - 1])
        for j in range(1, m + 1):
            dtw_matrix[0][j] = dtw_matrix[0][j - 1] + abs(word2[j - 1])

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(word1[i - 1] - word2[j - 1])
                # take last min from a square box
                last_min = min(dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1])
                dtw_matrix[i][j] = cost + last_min
        return dtw_matrix[n][m]

    def _save_results_(self, scores, option, p_comp, sem_id):
        names = ["dot_pdt_{}.txt", "pca_cosine_{}.txt", "svd_cosine_{}.txt", "nmf_cosine_{}.txt", "lda_cosine_{}.txt",
                 "edit_dist_{}.txt", "dtw_dist_{}.txt"]
        matrix = "sim_matrix"
        score_name = "score"
        sim_matrix = names[option-1].format(matrix)
        score_data = names[option-1].format(score_name)
        json.dump(json.dumps(scores), open(os.path.join(self.out_dir, sim_matrix), "w"))
        # json.dump(self.scores_final, open(os.path.join(self.out_dir, score_data), "w"))
        self.semantic_identifier(option, scores, p_comp, sem_id)

    def semantic_identifier(self, option, scores, p_comp, sem_id):
        names = ["dot_pdt_{}.txt", "pca_cosine_{}.txt", "svd_cosine_{}.txt", "nmf_cosine_{}.txt", "lda_cosine_{}.txt",
                 "edit_dist_{}.txt", "dtw_dist_{}.txt"]
        if sem_id == 1:
            name = "svd_matrix_{}".format(p_comp)
            sim_name = "svd_sim_matrix_{}".format(p_comp)
            sem_file_name = names[option - 1].format(name)
            sim_file_name = names[option - 1].format(sim_name)
            self.model = TruncatedSVD(n_components=p_comp)
        elif sem_id == 2:
            name = "nmf_matrix_{}".format(p_comp)
            sim_name = "nmf_sim_matrix_{}".format(p_comp)
            sem_file_name = names[option - 1].format(name)
            sim_file_name = names[option - 1].format(sim_name)
            self.model = NMF(n_components=p_comp)

        top_p = self.model.fit_transform(scores)
        json.dump(json.dumps(top_p.tolist()), open(os.path.join(self.out_dir, sem_file_name), "w"))
        file_name = {v: k for k, v in self.file_idx.items()}
        with open(os.path.join(self.out_dir, sim_file_name), "w+") as f:
            f.write("[")
            for topic in self.model.components_:
                f.write("{")
                i = 0
                for idx in np.argsort(topic)[::-1]:
                    file = file_name[i]
                    i += 1
                    score = topic[idx]
                    f.write("{}:{},".format(file, score))
                f.write('},\n')
            f.write("]")

    def _dot_product_similarity_(self, model):
        scores_flat = []
        f = 0
        for i in range(len(self.file_paths)):
            if model == 1:
                scores = np.dot(self.tf, self.tf[i].reshape((-1, 1))).tolist()
            elif model == 2:
                scores = np.dot(self.tfidf, self.tfidf[i].reshape((-1, 1))).tolist()
            flat = [item for sublist in scores for item in sublist]
            scores_flat.append(flat)
            f += 1
            if f == 3:
                break
        return scores_flat

    def _edit_cost_distance_(self):
        scores = []
        f = 0
        for file_id in self.sequences:
            scores.append(
                [sum([self._edit_distance_(seqs) for seqs in self._construct_list_for_mp_(fn, file_id)]) for fn in
                 self.sequences])
            f += 1
            if f == 3:
                break
        for i in range(len(scores)):
            scores_max = max(scores[i])
            for j in range(len(scores[i])):
                scores[i][j] = (scores_max - scores[i][j]) / scores_max
        return scores

    def _dtw_cost_distance_(self):
        scores = []
        f = 0
        for file_id in self.sequences:
            scores.append(
                [sum([self._dtw_distance_(seqs) for seqs in self._construct_list_for_mp_(fn, file_id)]) for fn in
                 self.sequences])
            f += 1
            if f == 3:
                break
        for i in range(len(scores)):
            scores_max = max(scores[i])
            for j in range(len(scores[i])):
                scores[i][j] = (scores_max - scores[i][j]) / scores_max
        return scores

    def _pca_similarity_(self):
        scores = (1-pairwise_distances(self.pca, metric="cosine"))
        scores = scores.tolist()
        return scores

    def _svd_similarity_(self):
        scores = (1-pairwise_distances(self.svd, metric="cosine"))
        scores = scores.tolist()
        return scores

    def _nmf_similarity_(self):
        scores = (1-pairwise_distances(self.nmf, metric="cosine"))
        scores = scores.tolist()
        return scores

    def _lda_similarity_(self):
        scores = (1-pairwise_distances(self.lda, metric="cosine"))
        scores = scores.tolist()
        return scores

    def process(self, model, option, p_comp, sem_id):
        if option == 1:
            scores = self._dot_product_similarity_(model)
        elif option == 2:
            scores = self._pca_similarity_()
        elif option == 3:
            scores = self._svd_similarity_()
        elif option == 4:
            scores = self._nmf_similarity_()
        elif option == 5:
            scores = self._lda_similarity_()
        elif option == 6:
            scores = self._edit_cost_distance_()
        elif option == 7:
            scores = self._dtw_cost_distance_()

        self._save_results_(scores, option, p_comp, sem_id)


if __name__ == "__main__":
    print("Performing Task 3")
    directory = input("Enter directory to use: ")
    task3 = Task3(directory)
    user_choice = 0
    while user_choice != 8:
        vec_model = int(input("Enter which vector model to use. (1) TF (2) TFIDF : "))
        sem_model = int(input("Enter which semantic identifier to use. (1) SVD (2) NMF : "))
        p_components = int(input("Enter number of components (p): "))
        print("User Options for similarity approaches, \n(1)Dot Product \n(2)PCA \n(3)SVD \n(4)NMF \n(5)LDA \n(6)Edit Distance \n(7)DTW \n(8)Exit")
        user_choice = int(input("Enter a user option: "))
        if user_choice == 8:
            break
        task3.process(vec_model, user_choice, p_components, sem_model)
