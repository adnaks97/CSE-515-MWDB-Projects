import os
import json

dir = os.path.abspath("outputs/task2")
fn = "03"
files = [os.path.join(dir,f) for f in os.listdir(dir) if fn+".txt" in f]

scores = []
for f in files:
    scores.append((f.split("/")[-1].split("_")[0], list(json.load(open(f,"r")).keys())))

for it in scores:
    print(it)