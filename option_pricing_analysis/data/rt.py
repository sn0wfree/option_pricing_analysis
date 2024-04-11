# coding=utf-8
import datetime
import sqlite3
from gevent import monkey

monkey.patch_all()
import json
import time

import pandas as pd
import requests
import grequests
from CodersWheel.QuickTool.file_cache import file_cache


@file_cache(granularity='d', enable_cache=True)
def option_sse_minute_sina(symbol: str = "10003720") -> pd.DataFrame:
    """
    指定期权品种在当前交易日的分钟数据, 只能获取当前交易日的数据, 不能获取历史分钟数据
    https://stock.finance.sina.com.cn/option/quotes.html
    :param symbol: 期权代码
    :type symbol: str
    :return: 指定期权的当前交易日的分钟数据
    :rtype: pandas.DataFrame
    """
    url = "https://stock.finance.sina.com.cn/futures/api/openapi.php/StockOptionDaylineService.getOptionMinline"
    params = {"symbol": f"CON_OP_{symbol}"}
    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "referer": "https://stock.finance.sina.com.cn/option/quotes.html",
        "sec-ch-ua": '" Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "script",
        "sec-fetch-mode": "no-cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36",
    }
    r = requests.get(url, params=params, headers=headers)
    data_json = r.json()
    temp_df = data_json["result"]["data"]
    data_df = pd.DataFrame(temp_df)
    data_df.columns = ["时间", "价格", "成交", "持仓", "均价", "日期"]
    data_df = data_df[["日期", "时间", "价格", "成交", "持仓", "均价"]]
    data_df["日期"] = pd.to_datetime(data_df["日期"]).dt.date
    data_df["日期"].ffill(inplace=True)
    data_df["价格"] = pd.to_numeric(data_df["价格"])
    data_df["成交"] = pd.to_numeric(data_df["成交"])
    data_df["持仓"] = pd.to_numeric(data_df["持仓"])
    data_df["均价"] = pd.to_numeric(data_df["均价"])
    return data_df


@file_cache(granularity='d', enable_cache=True)
def option_current_cffex_em() -> pd.DataFrame:
    url = "https://futsseapi.eastmoney.com/list/option/221"
    params = {
        "orderBy": "zdf",
        "sort": "desc",
        "pageSize": "20000",
        "pageIndex": "0",
        "token": "58b2fa8f54638b60b87d69b31969089c",
        "field": "dm,sc,name,p,zsjd,zde,zdf,f152,vol,cje,ccl,xqj,syr,rz,zjsj,o",
        "blockName": "callback",
        "_:": "1706689899924",
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    temp_df = pd.DataFrame(data_json["list"])
    temp_df.reset_index(inplace=True)
    temp_df["index"] = temp_df["index"] + 1
    temp_df.rename(
        columns={
            "index": "序号",
            "rz": "日增",
            "dm": "代码",
            "zsjd": "-",
            "ccl": "持仓量",
            "syr": "剩余日",
            "o": "今开",
            "p": "最新价",
            "sc": "市场标识",
            "xqj": "行权价",
            "vol": "成交量",
            "name": "名称",
            "zde": "涨跌额",
            "zdf": "涨跌幅",
            "zjsj": "昨结",
            "cje": "成交额",
        },
        inplace=True,
    )
    temp_df = temp_df[
        [
            "序号",
            "代码",
            "名称",
            "最新价",
            "涨跌额",
            "涨跌幅",
            "成交量",
            "成交额",
            "持仓量",
            "行权价",
            "剩余日",
            "日增",
            "昨结",
            "今开",
            "市场标识",
        ]
    ]
    temp_df["最新价"] = pd.to_numeric(temp_df["最新价"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["持仓量"] = pd.to_numeric(temp_df["持仓量"], errors="coerce")
    temp_df["行权价"] = pd.to_numeric(temp_df["行权价"], errors="coerce")
    temp_df["剩余日"] = pd.to_numeric(temp_df["剩余日"], errors="coerce")
    temp_df["日增"] = pd.to_numeric(temp_df["日增"], errors="coerce")
    temp_df["昨结"] = pd.to_numeric(temp_df["昨结"], errors="coerce")
    temp_df["今开"] = pd.to_numeric(temp_df["今开"], errors="coerce")
    return temp_df


@file_cache(granularity='d', enable_cache=True)
def option_current_em() -> pd.DataFrame:
    """
    东方财富网-行情中心-期权市场
    https://quote.eastmoney.com/center/qqsc.html
    :return: 期权价格
    :rtype: pandas.DataFrame
    """
    url = "https://23.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "200000",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": "m:10,m:12,m:140,m:141,m:151,m:163,m:226",
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,"
                  "f23,f24,f25,f22,f28,f11,f62,f128,f136,f115,f152,f133,f108,f163,f161,f162",
        "_": "1606225274063",
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    temp_df = pd.DataFrame(data_json["data"]["diff"])
    temp_df.reset_index(inplace=True)
    temp_df["index"] = temp_df["index"] + 1
    temp_df.columns = [
        "序号",
        "_",
        "最新价",
        "涨跌幅",
        "涨跌额",
        "成交量",
        "成交额",
        "_",
        "_",
        "_",
        "_",
        "_",
        "代码",
        "市场标识",
        "名称",
        "_",
        "_",
        "今开",
        "_",
        "_",
        "_",
        "_",
        "_",
        "_",
        "_",
        "昨结",
        "_",
        "持仓量",
        "_",
        "_",
        "_",
        "_",
        "_",
        "_",
        "_",
        "行权价",
        "剩余日",
        "日增",
    ]
    temp_df = temp_df[
        [
            "序号",
            "代码",
            "名称",
            "最新价",
            "涨跌额",
            "涨跌幅",
            "成交量",
            "成交额",
            "持仓量",
            "行权价",
            "剩余日",
            "日增",
            "昨结",
            "今开",
            "市场标识",
        ]
    ]
    temp_df["最新价"] = pd.to_numeric(temp_df["最新价"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["持仓量"] = pd.to_numeric(temp_df["持仓量"], errors="coerce")
    temp_df["行权价"] = pd.to_numeric(temp_df["行权价"], errors="coerce")
    temp_df["剩余日"] = pd.to_numeric(temp_df["剩余日"], errors="coerce")
    temp_df["日增"] = pd.to_numeric(temp_df["日增"], errors="coerce")
    temp_df["昨结"] = pd.to_numeric(temp_df["昨结"], errors="coerce")
    temp_df["今开"] = pd.to_numeric(temp_df["今开"], errors="coerce")
    option_current_cffex_em_df = option_current_cffex_em()
    big_df = pd.concat(objs=[temp_df, option_current_cffex_em_df], ignore_index=True)
    big_df["序号"] = range(1, len(big_df) + 1)
    return big_df


@file_cache(granularity='d', enable_cache=True)
def __option_current_em() -> pd.DataFrame:
    option_current_em_df = option_current_em()
    return option_current_em_df


def _em_parse_text(data_text):
    data_json = json.loads(data_text[data_text.find("(") + 1: data_text.rfind(")")])
    temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["trends"]])
    temp_df.columns = ["time", "close", "high", "low", "volume", "amount", "-"]
    temp_df = temp_df[["time", "close", "high", "low", "volume", "amount"]]
    temp_df["close"] = pd.to_numeric(temp_df["close"], errors="coerce")
    temp_df["high"] = pd.to_numeric(temp_df["high"], errors="coerce")
    temp_df["low"] = pd.to_numeric(temp_df["low"], errors="coerce")
    temp_df["volume"] = pd.to_numeric(temp_df["volume"], errors="coerce")
    temp_df["amount"] = pd.to_numeric(temp_df["amount"], errors="coerce")
    return temp_df


def option_minute_em_async():
    pass


@file_cache(granularity='d', enable_cache=True)
def option_minute_em(symbol: str = "MO2404-P-4450") -> pd.DataFrame:
    """
    东方财富网-行情中心-期权市场-分时行情
    https://wap.eastmoney.com/quote/stock/151.cu2404P61000.html
    :param symbol: 期权代码; 通过调用 ak.option_current_em() 获取
    :type symbol: str
    :return: 指定期权的分钟频率数据
    :rtype: pandas.DataFrame
    """
    option_current_em_df = __option_current_em()
    option_current_em_df["标识"] = (
            option_current_em_df["市场标识"].astype(str)
            + "."
            + option_current_em_df["代码"]
    )
    id_ = option_current_em_df[option_current_em_df["代码"] == symbol]["标识"].values[0]
    url = "https://push2.eastmoney.com/api/qt/stock/trends2/get"
    params = {
        "secid": id_,
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f17",
        "fields2": "f51,f53,f54,f55,f56,f57,f58",
        "iscr": "0",
        "iscca": "0",
        "ut": "f057cbcbce2a86e2866ab8877db1d059",
        "ndays": "1",
        "cb": "quotepushdata1",
    }
    r = requests.get(url, params=params)
    data_text = r.text
    # data_json = json.loads(data_text[data_text.find("(") + 1: data_text.rfind(")")])
    # temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["trends"]])
    # temp_df.columns = ["time", "close", "high", "low", "volume", "amount", "-"]
    # temp_df = temp_df[["time", "close", "high", "low", "volume", "amount"]]
    # temp_df["close"] = pd.to_numeric(temp_df["close"], errors="coerce")
    # temp_df["high"] = pd.to_numeric(temp_df["high"], errors="coerce")
    # temp_df["low"] = pd.to_numeric(temp_df["low"], errors="coerce")
    # temp_df["volume"] = pd.to_numeric(temp_df["volume"], errors="coerce")
    # temp_df["amount"] = pd.to_numeric(temp_df["amount"], errors="coerce")

    temp_df = _em_parse_text(data_text)
    return temp_df


class OptionMinutes(object):
    @staticmethod
    def get_codes():
        option_current_em_df = option_current_em()
        return option_current_em_df

    @staticmethod
    def get_option_minute(symbol="MO2404-C-5400"):
        option_minute_em_df = option_minute_em(symbol=symbol)
        return option_minute_em_df

    @classmethod
    def option_minute_all(cls, prefixes=('MO', 'HO', 'IO')):
        option_current_em_df = cls.get_codes()
        option_current_em_df['prefix'] = list(map(lambda x: x[:2].upper(), option_current_em_df['代码'].unique()))
        code_list = option_current_em_df[option_current_em_df['prefix'].isin(prefixes)]['代码'].unique().tolist()
        h = []
        for code in code_list:
            res = cls.get_option_minute(symbol=code)
            res['contract_code'] = code
            h.append(res)
            print(f'downloaded {code}!')
            time.sleep(1)
        df = pd.concat(h)

        return df


if __name__ == '__main__':
    # codes = ak.option_sse_codes_sina(symbol="看涨期权",
    #                                  trade_date="202404",
    #                                  underlying="000852", )

    # option_finance_minute_sina_df = ak.option_finance_minute_sina(symbol="000852")
    # print(option_minute_em_df)
    dt = datetime.datetime.today().strftime("%Y%m%d")

    df = OptionMinutes.option_minute_all(prefixes=('MO', 'HO', 'IO'))
    with sqlite3.connect(f'hq_option_minutes_{dt}.sqlite') as conn:
        df.to_sql(df, conn)
    # df.to_excel('minutes.xlsx', index=False)
    print(1)
    pass
