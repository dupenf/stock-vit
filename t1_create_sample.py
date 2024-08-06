import os
import pandas as pd

def create_1():
    features_dir="./datasets/features"
    files = [a for a in sorted(os.listdir(features_dir), key=lambda x: str(x[5:]))]
    for file in files:
        print(file)
        d = pd.read_csv(os.path.join(features_dir, file))
        
        a = pd.DataFrame()
        a = d.iloc[0:51][:]

        p = os.path.join(features_dir, "test"+file)
        a.to_csv(p)
        
        
        
# create_1()


def create_2():
    code = "sz.300001.csv"
    file="./datasets/features/" + code 
    d = pd.read_csv(file)
    
    a = pd.DataFrame()
    a = d.iloc[0:52][:]

    p = os.path.join("./datasets/feature1", code)
    a.to_csv(p)
    
create_2()