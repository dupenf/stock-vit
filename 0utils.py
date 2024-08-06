import pandas as pd
from bson.json_util import dumps
from io import StringIO
import json
from datetime import datetime
import time


def download_to_column(df):
    ##################################################################
    # df = stock_hfq_df
    # aa = df.loc[:, "收盘"]
    # df["前收盘"] = aa.shift()
    # df.at[0, "前收盘"] = df.at[0, "收盘"]
    #
    # tt = df.loc[:, "换手率"]
    # df["前换手率"] = tt.shift()
    # df.at[0, "前换手率"] = df.at[0, "换手率"]

    # df["换手率"] = df["换手率"].astype(float)
    # df["前换手率"] = df["换手率"].pct_change()
    # df["收盘"] = df["收盘"].astype(float)
    # df["前收盘"] = df["收盘"].pct_change()
    ##################################################################
    stock = pd.DataFrame()
    # stock["d"] = time.strftime("%Y-%m-%d", time.localtime(df.loc[:, "日期"]))
    stock["dateT"] = df.loc[:, "日期"].astype(str)
    stock["open"] = df.loc[:, "开盘"].astype(float)
    stock["close"] = df.loc[:, "收盘"].astype(float)
    # stock["pre_close"] = df.loc[:, "前收盘"]
    stock["high"] = df.loc[:, "最高"].astype(float)
    stock["low"] = df.loc[:, "最低"].astype(float)
    stock["turnover"] = df.loc[:, "换手率"].astype(float)
    # stock['pre_turnover'] = df.loc[:, "前换手率"]
    stock["zf"] = df.loc[:, "振幅"].astype(float)
    stock["zdf"] = df.loc[:, "涨跌幅"].astype(float)
    # # kdj9(stock)
    # # kdj45(stock)
    # # ma(stock)
    # data = stock.replace(np.nan, 0)
    # data = data.replace(np.inf, 0)
    # data = data.fillna(0)

    return stock


def jsonl_2_dp(jsonl):
    json_data = dumps(jsonl, indent=2)
    pd_stocks = pd.read_json(StringIO(json_data))
    return pd_stocks


def dp_2_jsonl(dataset):
    docs = dataset.to_json(orient="records")
    docs = json.loads(docs)
    # collection.insert_many(docs)
    return docs


def timestamp2time(timestamp=1707769336):
    dateArray = datetime.datetime.fromtimestamp(timestamp)
    # otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
    otherStyleTime = dateArray.strftime("%Y-%m-%d")
    print(otherStyleTime)
    return otherStyleTime


def time2timestamp(t="2024-02-13 04:22:16"):
    # 字符类型的时间
    # 转为时间数组
    timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
    print(timeArray)
    # timeArray可以调用tm_year等
    # 转为时间戳
    timeStamp = int(time.mktime(timeArray))
    print(timeStamp)
    return timeStamp


######################################################
######################################################
######################################################
def data_clean(df):
    return

######################################################
######################################################