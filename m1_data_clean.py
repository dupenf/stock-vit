import pandas as pd
import torch
import os
import numpy as np


def clean_volume_0(df):

    # null replace to 0
    df = df.replace(np.nan, 0)
    df = df.replace(np.inf, 0)
    df = df.fillna(0)

    df["volume"] = df["volume"].astype(int)

    index = df.loc[df["volume"] == 0].index
    df = df.drop(df.index[index])

    index = df.loc[df["close"] == 0].index
    df = df.drop(df.index[index])
    df.reset_index(drop=True)

    return df


def process_features(in_dir, out_dir):
    # codes = pd.read_csv("./datasets/all_codes.csv")
    # codes = codes["code"].tolist()

    # codes = ["sz.300001"]
    file_dir = in_dir  # "./datasets/origins"

    a_s = [a for a in sorted(os.listdir(file_dir), key=lambda x: str(x[5:]))]
    for a in a_s:
        file = os.path.join(file_dir, a)
        df = pd.read_csv(file)

        df = clean_volume_0(df)

        df.to_csv(os.path.join(out_dir, a))
        print(a)


process_features(in_dir="./datasets/cyday", out_dir="./datasets/cleaned/")
