import os
import json
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD, NMF, LatentDirichletAllocation

class Task1(object):
    def __init__(self, inputDir, k, vector_model, technique, out_dir="outputs"):
        self.input_dir = os.path.abspath(inputDir)
        self.out_dir = os.path.join(out_dir, "task1")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.num_components = k
        self.vector_model = vector_model
        self.technique = technique
        self.file_vectors = []
        self.reduced_file_vectors = []
        self.output_filename = ""
        self.model = None
        self.word_indexes = self.get_word_indexes("all_words_idx.txt")
        files = sorted([x.split(".")[0] for x in os.listdir(os.path.join("phase2_outputs", "task0a")) if ".wrd" in x])
        indices = list(range(0, len(files)))
        self.idx_file_map = dict(zip(indices, files))
        self.file_idx_map = dict(zip(files, indices))
        self.load_vectors()
        self.run_model()
        # self.write_outputs()
        self.write_task2_inputs()

    #Load vector index word mapping    
    def get_word_indexes(self, indexFileName):
        tempIndexData = pkl.load(open(os.path.join(self.input_dir, indexFileName), 'rb'))
        return {tempIndexData[i]:i for i in tempIndexData} # reversing key/values
    
    #Load TF and TF-IDF vectors for each file
    def load_vectors(self):
        self.vector_file_prefix = "tf_vectors_" if self.vector_model==1 else "tfidf_vectors_"
        
        vectors = {}
        for fileName in os.listdir(self.input_dir):
            if(fileName.startswith(self.vector_file_prefix)):
                fileNumber = fileName.split('.')[0].split('_')[-1]
                with open(os.path.join(self.input_dir, fileName), 'r') as f:
                    vectors[fileNumber] = json.loads(json.load(f))

        self.file_vectors = np.array([vectors[key] for key in sorted(vectors)])
    
    #Running feature reduction
    def run_model(self):
        #Choose model based on user option
        if self.technique==1:
            self.output_filename = "pca_{}_{}.txt".format(self.vector_model, self.num_components)
            self.model = PCA(n_components=self.num_components)
        elif self.technique==2:
            self.output_filename = "svd_{}_{}.txt".format(self.vector_model, self.num_components)
            self.model = TruncatedSVD(n_components=self.num_components)
        elif self.technique==3:
            self.output_filename = "nmf_{}_{}.txt".format(self.vector_model, self.num_components)
            self.model = NMF(n_components=self.num_components, max_iter=500)
        else:
            self.output_filename = "lda_{}_{}.txt".format(self.vector_model, self.num_components)
            self.model = LatentDirichletAllocation(n_components=self.num_components)

        # scaler = MinMaxScaler()
        # self.file_vectors = scaler.fit_transform(self.file_vectors)
        self.reduced_file_vectors = self.model.fit_transform(self.file_vectors)
    
    #Writing results as <word, scores> pairs     
    def write_outputs(self):
        name = self.output_filename.split("_")[0] + "_{}_vectors.txt".format(self.vector_model)
        json.dump(json.dumps(self.reduced_file_vectors.tolist()), open(os.path.join(self.out_dir, name), "w"))
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


    #Write reduced dimensional data
    def write_task2_inputs(self):
        name = self.output_filename.split("_")[0] + "_{}_vectors.txt".format(self.vector_model)
        json.dump(json.dumps(self.reduced_file_vectors.tolist()), open(os.path.join(self.out_dir, name), "w"))


if __name__=="__main__":
    inputDir = input("Enter the directory to use: ")
    numComponents = int(input("Enter number of components (k): "))
    vectorModel = int(input("Enter vector model (1-TF, 2-TFIDF): "))
    technique = int(input("Enter model to use (1-PCA, 2-SVD, 3-NMF, 4-LDA): "))
    t1 = Task1(inputDir, numComponents, vectorModel, technique)