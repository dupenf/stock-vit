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
from m6_vit import ViT3D
from m3_loaddatasets import DatasetsDay
import matplotlib.pyplot as plt
import numpy as np


torch.set_printoptions(threshold=float("inf"))

################################################################################
seq_length = 30
features_len=8
device = "cuda"

model = ViT3D(    
    seq_len=seq_length,
    features_len=features_len,
    num_classes=2
    ).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=3, verbose=True
    )  # .to(device)
################################################################################
features_files = "./datasets/features"
# features_files = "./02stocks/vitday/datasets/feature1"

files = [ os.path.join(features_files, a) for a in sorted(os.listdir(features_files), 
                                                          key=lambda x: (x[4:])) ]
################################################################################

################################################################################
for epoch in range(1000):    
    model.train()
################################################################################    
    for file in files:
        print(file)
        epoch_losses = []
        d = pd.read_csv(file)
        a = DatasetsDay(d,day_length=seq_length)
        loader = DataLoader(a, batch_size=128, shuffle=False)
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            # print(x)
            # print(f">>> Y: ", y)
            
            optimizer.zero_grad()
            outputs = model(x)
            # print(f">>> outputs:", outputs.shape)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            print(loss.item())
            #################################################################
            #################################################################
            # print(outputs)
            # o = torch.argmax(outputs,dim=1)
            # if torch.equal(y,o):
            #     print(f"succeedsssssssssssssssssssssssss->",o)
            # else:
            #     print(f"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee:",y,o)
            # p1 = outputs[0:1][:]
            # p2 = outputs[1:1][:]
            # p1 = p1.squeeze()
            # p2 = p2.squeeze()
            # print(p1)
            # plt.plot(p1.detach().cpu().numpy())
            # plt.plot(p2.detach().cpu().numpy())
            # plt.show()            
            #################################################################
            #################################################################
            # break
            
        mean_loss = np.mean(epoch_losses)
        print(f">>> Epoch {epoch} train loss: ", mean_loss)
        # epoch_losses = []
        # # Something was strange when using this?
        # # model.eval()
        # for step, (inputs, labels) in enumerate(test_dataloader):
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #     epoch_losses.append(loss.item())
            
        #     print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))
        
        # if epoch % 100 == 0:
        #         #         models_save = os.path.join(
        #         #     models_path, "./{}-{}-{}.pt".format(m_name, epoch, mean_loss)
        #         # )
        scheduler.step(mean_loss)
        torch.save(model,"./model.pt")
        
    #     break
    # break