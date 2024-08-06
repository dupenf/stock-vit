import baostock as bs
import pandas as pd

#### 登陆系统 ####
# lg = bs.login()
# 显示登陆返回信息
# print("login respond error_code:" + lg.error_code)
# print("login respond  error_msg:" + lg.error_msg)




def get_all_codes(save_path="./datasets"):
    # s = bs.query_all_stock()
    # s_df = s.get_data()
    # s_df.to_csv("./datasets/all_codes.csv", encoding="utf-8", index=False)
    #### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print("login respond error_code:" + lg.error_code)
    print("login respond  error_msg:" + lg.error_msg)

    #### 获取证券信息 ####
    rs = bs.query_all_stock(day="2024-06-20")
    print("query_all_stock respond error_code:" + rs.error_code)
    print("query_all_stock respond  error_msg:" + rs.error_msg)

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == "0") & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    #### 结果集输出到csv文件 ####
    result.to_csv(save_path+"/all_codes.csv", encoding="utf-8", index=False)
    print(result)

    #### 登出系统 ####
    bs.logout()


def download_code_hist(
    save_path="./datasets/origins",
    code="sh.600000",
    start_date="1999-09-01",
    end_date="2024-06-30",
    freq="d",
    adjustflag="1"
):

    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    fields = "date,time,code,open,high,low,close,volume,adjustflag",
    if freq == "d":
        fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    rs = bs.query_history_k_data_plus(
        code,
        # "date,time,code,open,high,low,close,volume,adjustflag",
        # "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        fields=fields,
        start_date=start_date,
        end_date=end_date,
        frequency=freq,
        adjustflag=adjustflag, # hfq
    )

    #### 打印结果集 ####
    data_list = []
    while (rs.error_code == "0") & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    
    # print(result)

    #### 结果集输出到csv文件 ####
    filename = save_path + "/" + code + ".csv"
    result.to_csv(filename, index=True)
    # print(result)


def download_codes(save_path="./datasets/origins", filters=[],freq="d",adjustflag="1"):
    codes = pd.read_csv("./datasets/all_codes.csv")
    codes = codes["code"].tolist()

    lg = bs.login()
    # 显示登陆返回信息
    print("login respond error_code:" + lg.error_code)
    print("login respond  error_msg:" + lg.error_msg)

    for code in codes:
        ignore = True
        for f in filters:
            if code.startswith(f):
                ignore = False
        if ignore:
            continue

        print(code)
        # if code < "sz.300138":
        #     continue
        
        download_code_hist(save_path=save_path ,code=code,freq=freq,adjustflag=adjustflag)
    ### 登出系统 ####
    bs.logout()


# get_all_codes(save_path="./datasets")
download_codes(
    save_path="./datasets/cyday",
    filters=["sh.68","sz.3"],
    # filters=["sz.301"]
    freq="d",
    adjustflag="1" # hfq
    )


