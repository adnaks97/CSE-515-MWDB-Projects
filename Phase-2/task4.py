import numpy as np
import glob
import random
from pathlib import Path
import json
import sys
import os
from sklearn.preprocessing import MinMaxScaler

class Task4(object):
    def __init__(self, inputDir, task, k=None, maxIter=None, userChoice=None, vectorModel=None):
        self.input_dir = os.path.abspath(inputDir)
        self.vector_model = vectorModel
        self.out_dir = os.path.join("outputs", "task4", "task4"+task)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.k = k
        self.userChoice = userChoice
        self.max_iter = maxIter
        self.task = task
        self.filenames = ["dot_pdt_{}_{}", "pca_cosine_{}_{}", "svd_cosine_{}_{}", "nmf_cosine_{}_{}", "lda_cosine_{}_{}", "edit_dist_{}_{}", "dtw_dist_{}_{}"]
        self.tf_files = sorted([k for k in os.listdir(os.path.join("outputs", "task0b")) if "tf_" in k and ".txt" in k])
        self.file_idx, self.idx_file = {}, {}
        for i,f in enumerate(self.tf_files):
            fn = f.split(".")[0].split("_")[-1]
            self.file_idx[fn] = i
            self.idx_file[i] = fn
        self.perform_corresponding_task()

    def perform_corresponding_task(self):
        if self.task == "a":
            fn = self.filenames[self.userChoice - 1].format("svd_new", "*_*")
            if len(glob.glob(os.path.join(self.input_dir, "task3", fn))) == 0:
                sys.exit("Please first run this option for Task 3 to make the file available")
            if self.vector_model == 1:
                fn = [x for x in glob.glob(os.path.join(self.input_dir, "task3", fn)) if "_1.txt" in x][0]
            elif self.vector_model == 2:
                fn = [x for x in glob.glob(os.path.join(self.input_dir, "task3", fn)) if "_2.txt" in x][0]
            svd_mat = np.array(json.loads(json.load(open(os.path.join(self.input_dir, "task3", fn), "r"))))
            svd_mat = svd_mat.reshape((svd_mat.shape[0], -1))
            cluster_assignments = self.top_p_assign_svd(svd_mat)
            self.outputFileName = "cluster_assignment_4a_{}.txt".format(fn.split("/")[-1].split(".")[0])
            self.write_data(cluster_assignments)


        if self.task == "b":
            fn = self.filenames[self.userChoice - 1].format("nmf_new", "*_*")
            if len(glob.glob(os.path.join(self.input_dir, "task3", fn))) == 0:
                sys.exit("Please first run this option for Task 3 to make the file available")
            if self.vector_model == 1:
                fn = [x for x in glob.glob(os.path.join(self.input_dir, "task3", fn)) if "_1.txt" in x][0]
            elif self.vector_model == 2:
                fn = [x for x in glob.glob(os.path.join(self.input_dir, "task3", fn)) if "_2.txt" in x][0]
            nmf_mat = np.array(json.loads(json.load(open(os.path.join(self.input_dir, "task3", fn), "r"))))
            nmf_mat = nmf_mat.reshape((nmf_mat.shape[0], -1))
            cluster_assignments = self.top_p_assign_nmf(nmf_mat)
            self.outputFileName = "cluster_assignment_4b_{}.txt".format(fn.split("/")[-1].split(".")[0])
            self.write_data(cluster_assignments)

        if self.task == "c":
            fn = self.filenames[self.userChoice - 1].format("sim_matrix", "*")
            if self.vector_model == 1:
                self.inputFileName = [x for x in glob.glob(os.path.join(self.input_dir, "task3", fn)) if "_1.txt" in x][0]
            elif self.vector_model == 2:
                self.inputFileName = [x for x in glob.glob(os.path.join(self.input_dir, "task3", fn)) if "_2.txt" in x][0]
            self.outputFileName = "kmeans_{}_{}.txt".format(self.inputFileName.split("/")[-1].split('_sim_matrix')[0], self.vector_model)
            self.k_means_clustering()

        if self.task == "d":
            fn = self.filenames[self.userChoice - 1].format("sim_matrix", "*")
            if self.vector_model == 1:
                self.inputFileName = [x for x in glob.glob(os.path.join(self.input_dir, "task3", fn)) if "_1.txt" in x][0]
            elif self.vector_model == 2:
                self.inputFileName = [x for x in glob.glob(os.path.join(self.input_dir, "task3", fn)) if "_2.txt" in x][0]
            self.outputFileName = "spectral_{}_{}.txt".format(self.inputFileName.split("/")[-1].split('_sim_matrix')[0], self.vector_model)
            self.spectral_clustering()

    def read_data(self):
        if (not os.path.exists(os.path.join(self.input_dir, "task3", self.inputFileName))):
            sys.exit("Please first run this option for Task 3 to make the file available")
        with open(os.path.join(self.input_dir, "task3", self.inputFileName), 'r') as inputData:
            self.inputMatrix = np.array(eval(inputData.read()[1:-1]))

    def write_data(self, dataAssignments):
        json.dump(dataAssignments, open(os.path.join(self.out_dir, self.outputFileName), 'w'))

    # def top_p_assign(self, fi):
    #     p_component = fi.shape[1]
    #     min_max_scaler = MinMaxScaler()
    #     normalized_array = min_max_scaler.fit_transform(fi)
    #     top_p = {i:[] for i in range(p_component)}
    #     for i, j in enumerate(normalized_array):
    #         idx = np.argmax(j)
    #         top_p[idx].append(self.idx_file[i])
    #     return top_p

    def top_p_assign_svd(self, fi):
        p_component_svd = fi.shape[1]
        min_max_scaler = MinMaxScaler()
        normalized_array = min_max_scaler.fit_transform(fi)
        top_p_svd = {i: [] for i in range(p_component_svd)}
        for i, j in enumerate(normalized_array):
            idx = np.argmax(j)
            top_p_svd[idx].append(self.idx_file[i])
        return top_p_svd

    def top_p_assign_nmf(self, fi):
        p_component_nmf = fi.shape[1]
        top_p_nmf = {i: [] for i in range(p_component_nmf)}
        for i, j in enumerate(fi):
            idx = np.argmax(j)
            top_p_nmf[idx].append(self.idx_file[i])
        return top_p_nmf

    def k_means_clustering(self):
        self.read_data()
        clusters, dataAssignments = self.k_means()
        self.write_data(clusters)
        self.outputFileName = "spectral_dataAssignments_{}.txt".format(self.inputFileName.split("/")[-1].split('_sim_matrix')[0])
        self.write_data(dataAssignments)

    def spectral_clustering(self):
        self.read_data()
        self.make_laplacian()
        clusters, dataAssignments = self.spectral()
        self.write_data(clusters)
        self.outputFileName = "kmeans_dataAssignments_{}.txt".format(
            self.inputFileName.split("/")[-1].split('_sim_matrix')[0])
        self.write_data(dataAssignments)

    def isEqual(self, a, b):
        try:
            np.testing.assert_equal(a, b)
            return True
        except:
            return False

    def k_means(self):
        curIter = 0
        if(self.k>len(self.inputMatrix)):
            kRandomIndexes = random.sample(range(0,len(self.inputMatrix)), len(self.inputMatrix)) + [random.randint(0, len(self.inputMatrix)) for i in range(self.k-len(self.inputMatrix))]
        else:
            kRandomIndexes = random.sample(range(0,len(self.inputMatrix)), self.k)
        # Initializing centroids
        centroids = np.array([self.inputMatrix[i] for i in kRandomIndexes])
        prevCentroidAssignments = {}
        while(curIter < self.max_iter):
            centroidAssignments = {}
            dataAssignments = []
            # Assignments to centroid
            for row in self.inputMatrix:
                distances = [np.linalg.norm(row-centroid) for centroid in centroids]
                centroidToPlaceIn = distances.index(min(distances))
                dataAssignments.append(centroidToPlaceIn)
                if(centroidToPlaceIn in centroidAssignments):
                    centroidAssignments[centroidToPlaceIn].append(row)
                else:
                    centroidAssignments[centroidToPlaceIn] = [row]
            # Keep old centroids
            prevCentroids = centroids.copy()
            # Find new centroids
            for i in centroidAssignments.keys():
                centroids[i] = np.mean(centroidAssignments[i], axis=0)
            # Calculate error - NOT NEEDED
            error = np.linalg.norm(prevCentroids-centroids)
            print(curIter,error)
            # If 2 consecutive runs w/ same results, then converged
            if self.isEqual(centroidAssignments,prevCentroidAssignments):
                print("Assignments Equal. Converged!")
                break
            # Save previous assignments
            prevCentroidAssignments = centroidAssignments.copy()
            curIter += 1
        clusters = {}
        for i, val in enumerate(dataAssignments):
            clusters[val] = clusters.get(val, []) + [self.idx_file[i]]
        return clusters, dataAssignments

    def make_laplacian(self):
        """
        Function to compute Laplacian given similarity matrix of gestures.
        """
        # Adjacency matrix based on similarity values
        self.A = np.copy(self.inputMatrix)
        # Making all similarity values <0 as 0
        self.A[self.A < 0] = 0

        # Creating Diagonal matrix as degree of each node
        self.D = np.diag(self.A.sum(axis=1))

        # Compute unnormalized laplacian
        self.L = self.D - self.A

    def compute_eigen_values(self):
        """
        Function to compute eigen values, eigenvectors of a matrix and return k-dimensional embedding of matrix based on decomposition
        returns:
            embedding - np array of dims len(data) x k
        """
        # Find eigen values and eigen vectors
        eigvals, eigvecs = np.linalg.eig(self.L)

        # First k eigen values
        inds = np.argsort(eigvals)
        # K dimensional embedding
        embedding = eigvecs[inds[:self.k]].T
        return embedding

    def spectral(self):
        """
        Function to do spectral clustering. Calls function to find Laplacian and then computes k-clusters.
        """
        embedding = self.compute_eigen_values()
        print("Data dims in latent space: ", embedding.shape)
        # Doing K-Means on first k eigen vectors (k-dimensional embedding)
        self.inputMatrix = embedding
        clusters, dataAssignments = self.k_means()
        return clusters, dataAssignments

if __name__=="__main__":
    directory = input("Enter directory to use: ")
    user_choice = 0
    while user_choice != 8:
        task = input("Enter the subtask to perform (a/b/c/d) : ").lower()
        vec_model = int(input("Enter which vector model to use. (1) TF (2) TFIDF : "))
        print("User Options for K-Means clustering, \n(1)Dot Product \n(2)PCA \n(3)SVD \n(4)NMF \n(5)LDA \n(6)Edit Distance \n(7)DTW \n(8)Exit")
        user_choice = int(input("Enter a user option: "))
        if user_choice == 8:
            break
        if task in ["c", "d"]:
            k = int(input("Enter number of clusters (p): "))
            Task4(directory, task, k, 100, user_choice, vec_model)
        else:
            Task4(directory, task, 5, 100, user_choice, vec_model)