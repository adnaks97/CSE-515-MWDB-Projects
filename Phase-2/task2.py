import os
import numpy as np
import pickle as pkl


class Task2:
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        self.tf_files = [os.path.join(self.dir,k) for k in os.listdir(self.dir) if "tf_" in k and ".txt" in k]
        self.tfidf_files = [os.path.join(self.dir, k) for k in os.listdir(self.dir) if "tfidf_" in k and ".txt" in k]
        self._load_all_vectors_()

    def _load_all_vectors_(self):
        self.tf = {}
        for fname in self.tf_files:
            self.tf[fname.split("/")[-1]] = np.array(pkl.load(open(fname, "r"))).reshape((1,-1))

        self.tfidf = {}
        for fname in self.tfidf_files:
            self.tfidf[fname.split("/")[-1]] = np.array(pkl.load(open(fname, "r"))).reshape((1,-1))

    def _dot_product_similarity_(self, fn, model):
        scores = {}
        for k in self.tf:
            if k != fn:
                if model == 1:
                    scores[k] = np.dot(self.tf[fn],self.tf[k]).ravel()[0]
                elif model == 2:
                    scores[k] = np.dot(self.tfidf[fn],self.tfidf[k]).ravel()[0]
        top_10_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10])
        return top_10_scores

    def find_10_similar_gestures(self, fn, model, option):
        if option == 0:
            print(self._dot_product_similarity_(fn, model))


if __name__ == "__main__":
    print("Performing Task 2")
    directory = "data"
    task2 = Task2(directory)
    user_choice = 0
    while user_choice != 8:
        file_name = input("Enter the file id to use: ")
        vec_model = int(input("Enter which vector model to use. (1) TF (2) TFIDF : "))
        print("User Options for similarity approaches, \n(1)Dot Product \n(2)PCA \n(3)SVD \n(4)NMF \n(5)LDA \n(6)Edit Distance \n(7)DTW \n(8)Exit")
        user_choice = int(input("Enter a user option: "))
        if user_choice == 8:
            break
        task2.find_10_similar_gestures(file_name, vec_model, user_choice)