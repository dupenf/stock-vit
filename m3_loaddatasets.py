import pandas as pd
import torch
import os
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

max_y = 0
min_y = 99999

const_y_days = 5


def y_feature_extract(y1):
    y = y1["zdf"].values[0]
    y = y - 1 + 0.3
    y = int(y * 1000)  # [0, 4500] num_classes = 4500
    
    return y

def y_feature_extract_1_0(y1):
    y = y1["zdf"].max()
    if y > 1.1:
        return 1
    else:
        return 0

def feature_extract_x_7(x1):

    close = x1[0:1]["close"].values[0]
    volume = x1[0:1]["volume"].values[0]

    # print(x)
    # print(x.shape)
    factor = 1
    x = pd.DataFrame()
    x["high"] = x1["high"] / close * factor
    x["low"] = x1["low"] / close * factor
    x["open"] = x1["open"] / close * factor
    x["close"] = x1["close"] / close * factor

    x["ma5"] = x1["ma5"] / close * factor
    x["ma20"] = x1["ma20"] / close * factor
    x["ma60"] = x1["ma60"] / close * factor

    # x["volume"] = x1["volume"] / volume * factor

    return x


def feature_extract_x_2(x1):
    close = x1[0:1]["close"].values[0]
    volume = x1[0:1]["volume"].values[0]

    # print(x)
    # print(x.shape)
    factor = 10
    x = pd.DataFrame()
    x["high"] = (x1["high"] - x1["close"]) / close * factor
    x["low"] = (x1["low"] - x1["close"]) / close * factor
    x["open"] = (x1["open"] - x1["close"]) / close * factor
    # x["close"] = (x1["close"] - x["close"]) / close * factor
    x["zdf"] = x1["zdf"] - 1

    x["ma5"] = (x1["ma5"] - x1["close"]) / close * factor
    x["ma20"] = (x1["ma20"] - x1["close"]) / close * factor
    x["ma60"] = (x1["ma60"] - x1["close"]) / close * factor

    x["volume"] = x1["volume"] / volume * factor

    return x

def feature_extract_x(x1):

    close = x1[0:1]["close"].values[0]
    volume = x1[0:1]["volume"].values[0]

    # print(x)
    # print(x.shape)
    factor = 10
    x = pd.DataFrame()
    x["high"] = x1["high"] / close * factor
    x["low"] = x1["low"] / close * factor
    x["open"] = x1["open"] / close * factor
    x["close"] = x1["close"] / close * factor

    x["ma5"] = x1["ma5"] / close * factor
    x["ma20"] = x1["ma20"] / close * factor
    x["ma60"] = x1["ma60"] / close * factor

    x["volume"] = x1["volume"] / volume * factor

    return x


def features(x1, y1):
    y = y_feature_extract_1_0(y1)
    # if y >= 600 or y <= 0:
    #     print("error--------------------y > 600")
    #     print(y)
    #     print(y1)
        
    x = feature_extract_x_2(x1)

    return x, [y]


class DatasetsDay(Dataset):
    def __init__(self, df, day_length=40):
        self.df = df
        self.day_length = day_length

    def __len__(self):
        d = self.df.shape[0]
        d = d - self.day_length - const_y_days
        return 0 if d < 0 else d

    def __getitem__(self, i):
        a = i
        e = a + self.day_length

        if e >= self.df.shape[0]:
            print(
                "error================================================datasets_loader?"
            )
            return torch.ones(0, 0), torch.ones(0, 0)

        x1 = self.df.iloc[a:e, :]
        y1 = self.df.iloc[e : e + const_y_days, :]
        # print(x)
        # print(y)
        ############################################
        x, y = features(x1, y1)
        ############################################
        #
        x = torch.Tensor(x.values)
        # x = torch.unsqueeze(x, dim=0)
        # print(x.shape)
        # print("-----------------------x.shape")

        y = torch.LongTensor(y)
        y = torch.squeeze(y)

        return x, y

    def get_last_item(self):
        s = self.df.shape[0] - self.day_length
        e = self.df.shape[0]
        x1 = self.df.iloc[s:e, :]
        x = feature_extract_x(x1)
        x = torch.Tensor(x.values)

        return x


def test(features_dir="./datasets/features1/"):
    seq_length = 50
    files = [a for a in sorted(os.listdir(features_dir), key=lambda x: str(x[5:]))]
    for file in files:
        print(file)
        d = pd.read_csv(os.path.join(features_dir, file))
        a = DatasetsDay(d, day_length=seq_length)
        # print(len(a))

        global max_y
        global min_y
        max_y = 0
        min_y = 99999

        loader = DataLoader(a, batch_size=1,shuffle=True)
        for step, (x, y) in enumerate(loader):
            
            
            
            # print(step)
            print(x.shape)
            print(y.shape)
            # print("-----------------------x.shape")
            # print(x)
            # print(y)
            # print(x.shape)
            # print(y.shape)
            break
            pass
        print(max_y)
        print(min_y)


# files = [print(a) for a in sorted(os.listdir("./datasets/features"), key=lambda x: x[5:])]

# test(features_dir="./02stocks/vit15minutes/datasets/features")
# test(features_dir="./datasets/features")
