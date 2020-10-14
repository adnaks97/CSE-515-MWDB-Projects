from sklearn.decomposition import PCA, TruncatedSVD, NMF, LatentDirichletAllocation
import numpy as np
import pickle as pkl
from pathlib import Path
import os

class Task1(object):
    def __init__(self, inputDir, k, vector_model, technique):
        self.input_dir = os.path.abspath(inputDir)
        self.out_dir = os.path.join("outputs", "task1")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.num_components = k
        self.vector_model = vector_model
        self.technique = technique
        self.file_vectors = []
        self.reduced_file_vectors = []
        self.output_filename = ""
        self.model = None
        self.word_indexes = self.get_word_indexes("all_words_idx.txt")
        
        self.load_vectors()
        self.run_model()
        self.write_outputs()
        
    def get_word_indexes(self, indexFileName):
        tempIndexData = pkl.load(open(os.path.join(self.input_dir, indexFileName), 'rb'))
        return {tempIndexData[i]:i for i in tempIndexData} # reversing key/values
    
    def load_vectors(self):
        vectorFilePrefix = "tf_vectors_" if self.vector_model==1 else "tfidf_vectors_"
        
        vectors = {}
        for fileName in os.listdir(self.input_dir):
            if(fileName.startswith(vectorFilePrefix)):
                fileNumber = int(fileName.split('.')[0].split('_')[-1])
                with open(os.path.join(self.input_dir, fileName), 'r') as f:
                    vectors[fileNumber] = eval(f.read()[1:-1]) #[1:-1] to remove extra "

        self.file_vectors = np.array([vectors[key] for key in sorted(vectors)])
    
    def run_model(self):
        if(self.technique==1):
            self.output_filename = "pca_{}.txt".format(self.num_components)
            self.model = PCA(n_components=self.num_components)
        elif(self.technique==2):
            self.output_filename = "svd_{}.txt".format(self.num_components)
            self.model = TruncatedSVD(n_components=self.num_components)
        elif(self.technique==3):
            self.output_filename = "nmf_{}.txt".format(self.num_components)
            self.model = NMF(n_components=self.num_components)
        else:
            self.output_filename = "lda_{}.txt".format(self.num_components)
            self.model = LatentDirichletAllocation(n_components=self.num_components)

        self.reduced_file_vectors = self.model.fit_transform(self.file_vectors)
        
    def write_outputs(self):
        with open(os.path.join(self.out_dir, self.output_filename), "w+") as f:
            f.write("[")
            for topic in self.model.components_:
                f.write("{")
                for idx in np.argsort(topic)[::-1]:
                    originalWord = self.word_indexes[idx]
                    score = topic[idx]
                    f.write("{}:{},".format(originalWord, score))
                f.write('},\n')
            f.write("]")
            
if __name__=="__main__":
    inputDir = input("Enter the directory to use: ")
    numComponents = int(input("Enter number of components (k): "))
    vectorModel = int(input("Enter vector model (1-TF, 2-TFIDF): "))
    technique = int(input("Enter model to use (1-PCA, 2-SVD, 3-NMF, 4-LDA): "))
    t1 = Task1(inputDir, numComponents, vectorModel, technique)