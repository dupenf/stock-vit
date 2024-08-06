import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
from m3_loaddatasets import DatasetsDay
import numpy as np
from m6_vit import Vit3D


np.set_printoptions(threshold=10000)
################################################################################
seq_length = 50
device = "cuda"
model = torch.load("./model1.pt").to(device)
################################################################################
features_files = "./datasets/feature1"
# features_files = "./02stocks/vitday/datasets/feature1"
files = [ os.path.join(features_files, a) for a in sorted(os.listdir(features_files), 
                                                          key=lambda x: (x[4:])) ]
################################################################################
model.eval()
################################################################################
for epoch in range(1000):    
################################################################################    
    for file in files:
        print(file)
        d = pd.read_csv(file)
        a = DatasetsDay(d,day_length=seq_length)
        loader = DataLoader(a, batch_size=1, shuffle=True)
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            print(f">>> Y: ", y)
            outputs = model(x)            
            print(f">>> outputs:", outputs)
            print(torch.argmax(outputs,dim=1))
            break
        break
    break

        
