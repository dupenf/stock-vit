import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
from einops.layers.torch import Rearrange
from torch import Tensor
import torch.optim as optim
import numpy as np
import torch
import os
import pandas as pd
from m3_loaddatasets import DatasetsDay
from m6_vit import Vit3D

#############################################################################
seq_length = 50
features_len=8
device = "cuda"
m = torch.load("./model1.pt").to(device)
#############################################################################
features_files = "./datasets/features1"
files = [ os.path.join(features_files, a) for a in sorted(os.listdir(features_files), 
                                                          key=lambda x: (x[4:])) ]

zdfs = []
for file in files:
    print(file)
    epoch_losses = []
    
    d = pd.read_csv(file)
    a = DatasetsDay(d,day_length=seq_length)
    x = DatasetsDay.get_last_item()
    x = x.to(device)
    outputs = m(x)
    cls = outputs.argmax()
    zdf = cls / 1000 - 0.3 + 1
    print(zdf)
    zdfs.append(zdf)

