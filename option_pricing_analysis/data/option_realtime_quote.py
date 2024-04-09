# coding=utf-8
from gevent import monkey

monkey.patch_all()
import akshare as ak

import pandas as pd
import json
import aiohttp
import gevent

def get_zz1000_code():
    option_cffex_zz1000_list_sina_df = ak.option_cffex_zz1000_list_sina()
    return option_cffex_zz1000_list_sina_df['中证1000指数']


def _parse_text(data_text):
    data_json = json.loads(
        data_text[data_text.find("{"): data_text.rfind("}") + 1]
    )

    ['call_contract_volume', 'call_contract_bid_price', 'call_contract_last_price', 'call_contract_ask_price',
     'call_contract_ask_volume', 'call_contract_open_interest', 'call_contract_change', 'strike_price',
     'call_contract_identifier', 'put_contract_volume', 'put_contract_bid_price', 'put_contract_last_price',
     'put_contract_ask_price', 'put_contract_ask_volume', 'put_contract_open_interest', 'put_contract_change',
     'put_contract_identifier', 'retrieval_time', 'data_source']

    # "看涨合约-买量",
    # "看涨合约-买价",
    # "看涨合约-最新价",
    # "看涨合约-卖价",
    # "看涨合约-卖量",
    # "看涨合约-持仓量",
    # "看涨合约-涨跌",
    # "行权价",
    # "看涨合约-标识",

    # "看跌合约-买量",
    # "看跌合约-买价",
    # "看跌合约-最新价",
    # "看跌合约-卖价",
    # "看跌合约-卖量",
    # "看跌合约-持仓量",
    # "看跌合约-涨跌",
    # "看跌合约-标识",

    option_call_df = pd.DataFrame(
        data_json["result"]["data"]["up"],
        columns=[
            "call_contract_volume",
            "call_contract_bid_price",
            "call_contract_last_price",
            "call_contract_ask_price",
            "call_contract_ask_volume",
            "call_contract_open_interest",
            "call_contract_change",
            "strike_price",
            "call_contract_identifier",
        ],
    )
    option_put_df = pd.DataFrame(
        data_json["result"]["data"]["down"],
        columns=[
            "put_contract_volume",
            "put_contract_bid_price",
            "put_contract_last_price",
            "put_contract_ask_price",
            "put_contract_ask_volume",
            "put_contract_open_interest",
            "put_contract_change",
            "put_contract_identifier",
        ],
    )
    data_df = pd.concat([option_call_df, option_put_df], axis=1)
    data_df["call_contract_volume"] = pd.to_numeric(data_df["call_contract_volume"], errors="coerce")
    data_df["call_contract_bid_price"] = pd.to_numeric(data_df["call_contract_bid_price"], errors="coerce")
    data_df["call_contract_last_price"] = pd.to_numeric(data_df["call_contract_last_price"], errors="coerce")
    data_df["call_contract_ask_price"] = pd.to_numeric(data_df["call_contract_ask_price"], errors="coerce")
    data_df["call_contract_ask_volume"] = pd.to_numeric(data_df["call_contract_ask_volume"], errors="coerce")
    data_df["call_contract_open_interest"] = pd.to_numeric(data_df["call_contract_open_interest"], errors="coerce")
    data_df["call_contract_change"] = pd.to_numeric(data_df["call_contract_change"], errors="coerce")
    data_df["strike_price"] = pd.to_numeric(data_df["strike_price"], errors="coerce")
    data_df["put_contract_volume"] = pd.to_numeric(data_df["put_contract_volume"], errors="coerce")
    data_df["put_contract_bid_price"] = pd.to_numeric(data_df["put_contract_bid_price"], errors="coerce")
    data_df["put_contract_last_price"] = pd.to_numeric(data_df["put_contract_last_price"], errors="coerce")
    data_df["put_contract_ask_price"] = pd.to_numeric(data_df["put_contract_ask_price"], errors="coerce")
    data_df["put_contract_ask_volume"] = pd.to_numeric(data_df["put_contract_ask_volume"], errors="coerce")
    data_df["put_contract_open_interest"] = pd.to_numeric(data_df["put_contract_open_interest"], errors="coerce")
    data_df["put_contract_change"] = pd.to_numeric(data_df["put_contract_change"], errors="coerce")
    return data_df



async def async_get(url):

    async with ClientSession() as session:
        async with session.get(url) as response:
            res = await response.text()
            return res

def option_cffex_zz1000_spot_sina(symbol_list: (list, tuple) = ["mo2208",]) -> pd.DataFrame:
    """
    中金所-中证 1000 指数-指定合约-实时行情
    https://stock.finance.sina.com.cn/futures/view/optionsCffexDP.php
    :param symbol: 合约代码; 用 option_cffex_zz1000_list_sina 函数查看
    :type symbol: str
    :return: 中金所-中证 1000 指数-指定合约-看涨看跌实时行情
    :rtype: pd.DataFrame
    """
    url = "https://stock.finance.sina.com.cn/futures/api/openapi.php/OptionService.getOptionData"
    params = [{"type": "futures", "product": "mo", "exchange": "cffex", "pinzhong": symbol, } for symbol in symbol_list]
    r = requests.get(url, params=params)
    data_text = r.text
    return _parse_text(data_text)




if __name__ == '__main__':
    option_cffex_zz1000_spot_sina_df = ak.option_cffex_zz1000_spot_sina(symbol="mo2208")
    print(option_cffex_zz1000_spot_sina_df)
    pass
