import numpy as np
import os
import random
from pathlib import Path
import json
import sys

class Task4c(object):
    def __init__(self, inputDir, k, maxIter, userChoice):
        self.input_dir = os.path.abspath(inputDir)
        self.out_dir = os.path.join("outputs", "task4", "task4c")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.k = k
        self.max_iter = maxIter
        self.filenames = ["dot_pdt_{}.txt", "pca_{}.txt", "svd_{}.txt", "nmf_{}.txt", "lda_{}.txt", "edit_dist_{}.txt", "dtw_dist_{}.txt"]
        self.inputFileName = self.filenames[userChoice-1].format("sim_matrix")
        self.outputFileName = "kmeans_{}.txt".format(self.inputFileName.split('_sim_matrix')[0])
        
    def process(self):
        self.read_data()
        dataAssignments = self.kmeans()
        self.write_data(dataAssignments)
        
    def read_data(self):
        if(not os.path.exists(os.path.join(self.input_dir, self.inputFileName))):
            sys.exit("Please first run this option for Task 3 to make the file available")
            
        with open(os.path.join(self.input_dir, self.inputFileName), 'r') as inputData:
            self.inputMatrix = np.array(eval(inputData.read()[1:-1]))
            
    def write_data(self, dataAssignments):
        textData = json.dumps({i:dataAssignments[i] for i in range(len(dataAssignments))})
        with open(os.path.join(self.out_dir, self.outputFileName), 'w+') as f:
            f.write(textData)
            
    def isEqual(self, a,b):
        try:
            res = np.testing.assert_equal(a, b)
            return True
        except:
            return False
        
    def kmeans(self):
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
            if(self.isEqual(centroidAssignments, prevCentroidAssignments)):
                print("Assignments Equal. Converged!")
                break

            # Save previous assignments
            prevCentroidAssignments = centroidAssignments.copy()

            curIter += 1

        return dataAssignments
        
		
if __name__=="__main__":
    print("Performing Task 4c")
    maxIterations = 1000
    directory = input("Enter directory to use: ")
    user_choice = 0 
    while user_choice != 8:
        k = int(input("Enter number of clusters (p): "))
        print("User Options for K-Means clustering, \n(1)Dot Product \n(2)PCA \n(3)SVD \n(4)NMF \n(5)LDA \n(6)Edit Distance \n(7)DTW \n(8)Exit")
        user_choice = int(input("Enter a user option: "))
        if user_choice == 8:
            break
        Task4c(directory, k, maxIterations, user_choice).process()