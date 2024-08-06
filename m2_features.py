import akshare as ak
from io import StringIO
from bson.json_util import dumps
import json
import os
import numpy as np
import pandas as pd

X_Length = 30

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 1000)


def calc_pre_high_list(df):
    #  往前追溯3个高点，如果只有一个， 则另外两个和第一个相等。
    # df["high"] = df["high"].astype(float)
    ph = []
    pl = []
    for i in range(len(df)):
        h = df.iloc[i]["high"]
        l = df.iloc[i]["low"]
        if len(ph) < 1:
            ph.append(h)
            pl.append(l)
            df.iloc[i]["ph"] = h
            df.iloc[i]["pl"] = l
            continue

        ma20 = df.iloc[i]["ma20"]
        ma60 = df.iloc[i]["ma60"]
        if ma20 > ma60:
            r = all(v > h for v in ph)
            if not r:
                pre_v = -1
                for v in reversed(ph):
                    if v > h:
                        pre_v = v
                if pre_v > 0:
                    ph.append(pre_v)
                else:
                    ph.append(h)
        else:
            r = all(v < h for v in pl)
            if not r:
                pre_v = -1
                for v in reversed(pl):
                    if v < h:
                        pre_v = v
                if pre_v > 0:
                    ph.append(pre_v)
                else:
                    ph.append(l)

        df.iloc[i]["ph"] = ph[:-1]
        df.iloc[i]["pl"] = pl[:-1]

    return df


def calc_pre_high_list2(df):
    #  往前追溯3个高点，如果只有一个， 则另外两个和第一个相等。
    # df["high"] = df["high"].astype(float)
    ph = []
    pl = []
    h_i = 0
    l_i = 0
    for i in range(len(df)):
        h = df.iloc[i]["high"]
        l = df.iloc[i]["low"]
        if len(ph) < 1:
            ph.append(h)
            pl.append(l)
            df.iloc[i]["ph"] = h
            df.iloc[i]["pl"] = l
            continue
        c = df.iloc[i]["close"]
        ma20 = df.iloc[i]["ma20"]
        ma60 = df.iloc[i]["ma60"]
        if c > ma20:
            r = all(v > h for v in ph)
            if not r:
                pre_v = -1
                for v in reversed(ph):
                    if v > h:
                        pre_v = v
                if pre_v > 0:
                    ph.append(pre_v)
                else:
                    ph.append(h)
        else:
            r = all(v < h for v in pl)
            if not r:
                pre_v = -1
                for v in reversed(pl):
                    if v < h:
                        pre_v = v
                if pre_v > 0:
                    ph.append(pre_v)
                else:
                    ph.append(l)

        df.iloc[i]["ph"] = ph[:-1]
        df.iloc[i]["pl"] = pl[:-1]

    return df

def kdj_window(df, window=160, m1=60, m2=60, low="low", high="high", close="close"):
    low_list = df[low].rolling(window).min()
    low_list.fillna(value=df[low].expanding().min(), inplace=True)
    high_list = df[high].rolling(window).max()
    high_list.fillna(value=df[high].expanding().max(), inplace=True)

    rsv = (df[close] - low_list) / (high_list - low_list) * 100
    df['k' + str(window)] = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    df['d' + str(window)] = df['k9'].ewm(alpha=1 / m2, adjust=False).mean()
    df['j' + str(window)] = 3 * df['k9'] - 2 * df['d9']


def kdj4(df, low="low", high="high", close="close"):
    low_list = df[low].rolling(window=4).min()
    low_list.fillna(value=df[low].expanding().min(), inplace=True)
    high_list = df[high].rolling(window=4).max()
    high_list.fillna(value=df[high].expanding().max(), inplace=True)

    rsv = (df[close] - low_list) / (high_list - low_list) * 100
    df['k4'] = rsv.ewm(com=3).mean()
    df['d4'] = df['k4'].ewm(com=3).mean()
    df['j4'] = 3 * df['k4'] - 2 * df['d4']


def kdj160(df, low="low", high="high", close="close"):
    low_list = df[low].rolling(window=160).min()
    low_list.fillna(value=df[low].expanding().min(), inplace=True)
    high_list = df[high].rolling(window=160).max()
    high_list.fillna(value=df[high].expanding().max(), inplace=True)

    rsv = (df[close] - low_list) / (high_list - low_list) * 100
    df['k160'] = rsv.ewm(com=60).mean()
    df['d160'] = df['k160'].ewm(com=60).mean()
    df['j160'] = 3 * df['k160'] - 2 * df['d160']


def kdj9(df, low="low", high="high", close="close"):
    low_list = df[low].rolling(window=9).min()
    low_list.fillna(value=df[low].expanding().min(), inplace=True)
    high_list = df[high].rolling(window=9).max()
    high_list.fillna(value=df[high].expanding().max(), inplace=True)

    rsv = (df[close] - low_list) / (high_list - low_list) * 100
    df['k9'] = rsv.ewm(com=3).mean()
    df['d9'] = df['k9'].ewm(com=3).mean()
    df['j9'] = 3 * df['k9'] - 2 * df['d9']


def kdj45(df, low="low", high="high", close="close"):
    low_list = df[low].rolling(window=45).min()
    low_list.fillna(value=df[low].expanding().min(), inplace=True)
    high_list = df[high].rolling(window=45).max()
    high_list.fillna(value=df[high].expanding().max(), inplace=True)

    rsv = (df[close] - low_list) / (high_list - low_list) * 100
    df['k45'] = rsv.ewm(com=15).mean()
    df['d45'] = df['k45'].ewm(com=15).mean()
    df['j45'] = 3 * df['k45'] - 2 * df['d45']


def ma(df, close="close"):
    df['ma5'] = df[close].rolling(window=5).mean().dropna()
    df['ma20'] = df[close].rolling(window=20).mean().dropna()
    df["ma60"] = df[close].rolling(window=60).mean().dropna()
    # df['ma1000'] = df[close].rolling(window=1000).mean().dropna()



# def zdf(data):
#     if data.shape[0] % 16 != 0:
#         print("error================================> not 16 ")
        
#     zdf = pd.DataFrame()
#     zdf["zdf"] = -1    
#     for i in range(int(data.shape[0] / 16)-1):
#         s = int(i * 16 + 16)
#         e = int(s + 16)
#         print(e)
#         # print(e)
#         p1 = data[s-1:s]["close"].values[0]
#         p2 = data[e-1:e]["close"].values[0]
#         print(p1)
#         print(p2)
#         zdf[e-1:e]["zdf"] = p2 / p1
#         # break
#     print(zdf["zdf"])
#     data["zdf"] = zdf["zdf"]
    
#     return data

def get_features(data):    
    lines = data.shape[0]
    if lines < 120:
        return None
        
    ma(data)
    # data = zdf(data)
    data = data.loc[60:, :]
    # data = data.copy()    
    # data.reset_index(drop=True)
    
    # data = calc_pre_high_list(data)
    dataset = pd.DataFrame()
    
    close = data.iloc[0:1]["close"].values[0]
    volume = data.iloc[0:1]["volume"].values[0]
    # print(volume)
    dataset["date"] = data["date"]        
    
    # dataset["high"] = data["high"].astype(float) / close
    # dataset["low"] = data["low"].astype(float) / close
    # dataset["open"] = data["open"].astype(float) / close
    # dataset["close"] = data["close"].astype(float) / close
    
    # dataset["ma5"] = data["ma5"].astype(float) / close
    # dataset["ma20"] = data["ma20"].astype(float) / close
    # dataset["ma60"] = data["ma60"].astype(float) / close
    
    # dataset["volume"] = data["volume"].astype(float) / volume
    
    
    dataset["zdf"] = data["close"].astype(float) / data["preclose"]
    
    
    dataset["high"] = data["high"].astype(float) 
    dataset["low"] = data["low"].astype(float) 
    dataset["open"] = data["open"].astype(float) 
    dataset["close"] = data["close"].astype(float)
    
    dataset["ma5"] = data["ma5"].astype(float) 
    dataset["ma20"] = data["ma20"].astype(float)
    dataset["ma60"] = data["ma60"].astype(float)
    dataset["volume"] = data["volume"].astype(float) 
    
    
    
    
    if all(dataset['zdf'] >= 0.7) and all(dataset['zdf'] < 1.3):
        return dataset
    
    return None
    
    # null replace to 0
    # dataset = dataset.replace(np.nan, 0)
    # dataset = dataset.replace(np.inf, 0)
    # dataset = dataset.fillna(0)
    
    # return dataset



def process_features(in_dir,out_dir):
    # codes = pd.read_csv("./datasets/all_codes.csv")
    # codes = codes["code"].tolist()
    
    # codes = ["sz.300001"]
    file_dir = in_dir # "./datasets/origins"
    
    a_s = [a for a in sorted(os.listdir(file_dir), key=lambda x:str(x[5:]))]
    for a in a_s:    
        # if a.startswith("sz.30"):
        #     continue
        file  = os.path.join(file_dir,a)        
        df = pd.read_csv(file)
        df = get_features(df)
        if df is not None:
            # file = "./datasets/features/"+ a
            file = out_dir + a
            df.to_csv(file, index=False)
            print(a)
        # break
    
process_features(in_dir="./datasets/cleaned",out_dir="./datasets/features/")

# process_features(in_dir="./02stocks/vit15minutes/datasets/origins",
#                  out_dir="./02stocks/vit15minutes/datasets/features/")