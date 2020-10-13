import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

class Task3:
    """
    Class to perform task 3
    Takes a dir and user preferences about file no and feature and plots a heatmap
    """
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        self.out_dir = os.path.join(str(Path(self.dir).parent), "Outputs", "task3")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        # load all vectors created in  the previous task
        self.vectors = json.load(open(os.path.join(self.dir, "vectors.txt"), "r"))

    def plot_heatmap(self, file_id, feature=0):
        """
        This function takes a file_id and feature identifier to extract the specified data from the vector data and plots a heatmap
        :param file_id: file number ranging from 1-60 in sample data
        :param feature: feature identifier (tf tfidf, tfidf2) ranging [0-2]
        :return: None
        """
        num_sensors = 20
        features = []
        for s in range(num_sensors):
            features.append(self.vectors["{}-{}".format(file_id, s)][feature])
        ft_arr = np.array(features).reshape((num_sensors, -1))
        fig = plt.figure(figsize=(14, 8))
        sns.heatmap(ft_arr, cmap="gray", xticklabels=False)
        fig.savefig(os.path.join(self.out_dir, "{}_{}.png".format(file_id, feature)))
        plt.show()

    def get_user_preference(self):
        """
        Get user options to plot heatmap
        :return: None
        """
        file_id = int(input("Enter the file no you wish to see (1-66): "))
        feature = int(input("Enter the feature you wish to plot (tf - 0, tfidf - 1, tfidf2 - 2) : "))
        self.plot_heatmap(file_id, feature)


if __name__ == "__main__":
    dir = input("Enter the directory : ")
    task3 = Task3(dir)
    while True:
        option = int(input("Choose an option : \n 1) Check some plots \n 2) Exit \n"))
        if option == 2:
            break
        else:
            task3.get_user_preference()
