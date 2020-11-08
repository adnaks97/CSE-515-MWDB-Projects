import os
import glob
from pathlib import Path
from phase2.task0a import wordExtractor2
from phase2.task0b import Task0b
from phase2.task1 import Task1
from phase2.task3 import Task3

class Preprocess:
    def __init__(self, data_dir, output_dir):
        self.data_dir = os.path.abspath(data_dir)
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def task0a(self, r=3, s=3, w=3):
        if not os.path.exists(os.path.join(self.output_dir,"task0a")):
            file_list = glob.glob(self.data_dir + '/*/*.csv')
            for csv_file in file_list:
                print("Processing: ", csv_file)
                f = csv_file.split('/')[-1].split('.')[0]
                component = csv_file.split('/')[-2]
                wordExtractor_obj = wordExtractor2(fname=f, component=component, bands=r, window_size=w, stride=s, DIR=self.data_dir, out_dir=self.output_dir)
                wordExtractor_obj.main()
        else:
            print("already have results of task0a in outputs")

    def task0b(self):
        if not os.path.exists(os.path.join(self.output_dir,"task0b")) or len(os.listdir(os.path.join(self.output_dir,"task0b"))) == 0:
            Task0b(os.path.join(self.output_dir,"task0a"), self.output_dir)
        else:
            print("already have results of task0b in outputs")

    def task1(self, k=30, vm=2, tc=1):
        names = {1:"pca_{}_vectors.txt", 
                 2:"svd_{}_vectors.txt",
                 3:"nmf_{}_vectors.txt",
                 4:"lda_{}_vectors.txt"}
        
        if not os.path.exists(os.path.join(self.output_dir,"task1",names[tc].format(vm))):
            Task1(os.path.join(self.output_dir,"task0b"), k, vm, tc, self.output_dir)
        else:
            print("already have results of task1 in outputs")
    
    def task3(self, vm=2, uc=2):
        names = {2:"pca_cosine_sim_matrix_{}.txt", 
                 3:"svd_cosine_sim_matrix_{}.txt",
                 4:"nmf_cosine_sim_matrix_{}.txt",
                 5:"lda_cosine_sim_matrix_{}.txt"}

        if not os.path.exists(os.path.join(self.output_dir,"task3",names[uc].format(vm))):
            task3 = Task3(vm, self.output_dir)
            task3.process(uc)
        else:
            print("already have results of task1 in outputs")
        
if __name__ == "__main__":
    ob = Preprocess("Data_new_test", "phase2_outputs")
    ob.task0a()
    ob.task0b()
    ob.task1()
    ob.task3()