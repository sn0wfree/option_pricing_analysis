# coding = utf-8
from gevent import monkey

monkey.patch_all()

import akshare as ak

import pandas as pd
import json, datetime
import requests
import grequests
import os.path
import sqlite3
from glob import glob
import time
from CodersWheel.QuickTool.timer import timer
from ClickSQL import BaseSingleFactorTableNode


def get_zz1000_code():
    option_cffex_zz1000_list_sina_df = ak.option_cffex_zz1000_list_sina()
    code_list = option_cffex_zz1000_list_sina_df['中证1000指数']
    return code_list


def _parse_text(data_text):
    data_json = json.loads(
        data_text[data_text.find("{"): data_text.rfind("}") + 1]
    )

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
    data_df['retrieval_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_df['data_source'] = 'sina'

    return data_df


def err_handler(request, exception):
    print("请求出错")


@timer
def option_cffex_zz1000_spot_sina_async(symbol_list: (list, tuple) = None) -> pd.DataFrame:
    """
    中金所-中证 1000 指数-指定合约-实时行情
    https://stock.finance.sina.com.cn/futures/view/optionsCffexDP.php
    :param symbol: 合约代码; 用 option_cffex_zz1000_list_sina 函数查看
    :type symbol: str
    :return: 中金所-中证 1000 指数-指定合约-看涨看跌实时行情
    :rtype: pd.DataFrame
    """
    symbol_list = get_zz1000_code() if symbol_list is None else symbol_list
    print(symbol_list)

    url = "https://stock.finance.sina.com.cn/futures/api/openapi.php/OptionService.getOptionData"
    params_list = [{"type": "futures", "product": "mo", "exchange": "cffex", "pinzhong": symbol, } for symbol in
                   symbol_list]
    req_list = [grequests.get(url, params=param) for param in params_list]
    # loop = asyncio.get_event_loop()

    res_list = filter(None, grequests.map(req_list, size=len(symbol_list), exception_handler=err_handler))

    h = list((map(lambda x: _parse_text(x.text), res_list)))
    if len(h) != 0:
        return pd.concat(h)
    else:
        return pd.DataFrame()


@timer
def option_cffex_zz1000_spot_sina_recu(symbol_list: (list, tuple) = None) -> pd.DataFrame:
    """
    中金所-中证 1000 指数-指定合约-实时行情
    https://stock.finance.sina.com.cn/futures/view/optionsCffexDP.php
    :param symbol: 合约代码; 用 option_cffex_zz1000_list_sina 函数查看
    :type symbol: str
    :return: 中金所-中证 1000 指数-指定合约-看涨看跌实时行情
    :rtype: pd.DataFrame
    """
    symbol_list = get_zz1000_code() if symbol_list is None else symbol_list
    print(symbol_list)

    url = "https://stock.finance.sina.com.cn/futures/api/openapi.php/OptionService.getOptionData"
    params_list = [{"type": "futures", "product": "mo", "exchange": "cffex", "pinzhong": symbol, } for symbol in
                   symbol_list]
    h = []
    for param in params_list:
        resp = requests.get(url, params=param)
        h.append(_parse_text(resp.text))

    if len(h) != 0:
        return pd.concat(h)
    else:
        return pd.DataFrame()


cols = ['call_contract_volume', 'call_contract_bid_price', 'call_contract_last_price', 'call_contract_ask_price',
        'call_contract_ask_volume', 'call_contract_open_interest', 'call_contract_change', 'strike_price',
        'call_contract_identifier', 'put_contract_volume', 'put_contract_bid_price', 'put_contract_last_price',
        'put_contract_ask_price', 'put_contract_ask_volume', 'put_contract_open_interest', 'put_contract_change',
        'put_contract_identifier', 'retrieval_time', 'data_source']


def check_working_period(working_period, now=datetime.datetime.now().strftime("%H:%m:%S")):
    for (s, e) in working_period:
        if s <= now <= e:
            return True
    else:
        return False


def check_before_working_period(working_period, now=datetime.datetime.now().strftime("%H:%m:%S")):
    s_min = None
    for (s, e) in working_period:
        if s <= now <= e:
            return True
        else:
            s_min = min(s_min, s) if s_min is not None else s
    else:
        if s_min is not None and now <= s_min:
            return True
        else:
            return False


def store(df, conn):
    df.to_sql('option', conn, if_exists='append', index=False)


def create_file_name(store_path, today=datetime.datetime.now().strftime("%Y%m%d")):
    prefix = f"hq_mo_op_quote_{today}"
    file_name = f'{prefix}.sqlite'

    p = os.path.join(store_path, file_name)
    return prefix, file_name, p


def download_hq_quote(working_period=(('09:30:00', '11:30,00'),
                                      ('13:00:00', '15:00:00')),
                      store_path='./',
                      force=False,
                      conn=None
                      ):
    symbol_list = get_zz1000_code()
    today = datetime.datetime.now().strftime("%Y%m%d")

    prefix, file_name, p = create_file_name(store_path, today=today)


    tmp_cunt = 1

    with sqlite3.connect(p) as sqlite_conn:
        while True:
            now = datetime.datetime.now().strftime("%H:%m:%S")
            # print(now)
            if check_working_period(working_period, now=now):
                res = option_cffex_zz1000_spot_sina_async(symbol_list=symbol_list)
                if not res.empty:
                    store(res, sqlite_conn)
                print('will sleep 1s')
                time.sleep(1)
            elif force and tmp_cunt > 0:
                res = option_cffex_zz1000_spot_sina_async(symbol_list=symbol_list)
                if not res.empty:
                    store(res, sqlite_conn)
                    tmp_cunt -= 1
            elif now > '15:00:00':
                print(f'当前时间为{now},已经收盘,程序终止!')
                break
            elif now < '09:00:00':
                s = (pd.to_datetime('09:00:00') - pd.to_datetime(datetime.datetime.now().strftime("%H:%m:%S"))).seconds

                print(f'will sleep {s}s')
                time.sleep(s)
            else:
                print('will sleep 60s')
                time.sleep(60)
    if conn is not None:
        scan_and_create(conn, store_path=store_path, prefix="hq_mo_op_quote_*.sqlite")
    print('finish!')


def create_table_ck(ck_db_table='hq_quote.hq_mo_op_quote_today', sqlite_path='/home/ll/hq_data/MO',
                    sqlite_table='option'):
    """


    :return:
    """

    return f"""CREATE TABLE IF NOT EXISTS {ck_db_table} (
    call_contract_volume Int64 COMMENT 'Call Contract - Volume',
    call_contract_bid_price Float64 COMMENT 'Call Contract - Bid Price',
    call_contract_last_price Float64 COMMENT 'Call Contract - Last Price',
    call_contract_ask_price Float64 COMMENT 'Call Contract - Ask Price',
    call_contract_ask_volume Int64 COMMENT 'Call Contract - Ask Volume',
    call_contract_open_interest Int64 COMMENT 'Call Contract - Open Interest',
    call_contract_change Float64 COMMENT 'Call Contract - Change',
    strike_price Float64 COMMENT 'Strike Price',
    call_contract_identifier String COMMENT 'Call Contract - Identifier',
    put_contract_volume Int64 COMMENT 'Put Contract - Volume',
    put_contract_bid_price Float64 COMMENT 'Put Contract - Bid Price',
    put_contract_last_price Float64 COMMENT 'Put Contract - Last Price',
    put_contract_ask_price Float64 COMMENT 'Put Contract - Ask Price',
    put_contract_ask_volume Int64 COMMENT 'Put Contract - Ask Volume',
    put_contract_open_interest Int64 COMMENT 'Put Contract - Open Interest',
    put_contract_change Float64 COMMENT 'Put Contract - Change',
    put_contract_identifier String COMMENT 'Put Contract - Identifier',
    retrieval_time DateTime COMMENT 'Retrieval Time',
    data_source String COMMENT 'Data Source'
) ENGINE = SQLite('{sqlite_path}', '{sqlite_table}');"""


def scan_and_create(conn, store_path='/home/ll/hq_data/MO', prefix="hq_mo_op_quote_*.sqlite"):
    for path in glob(os.path.join(store_path, prefix)):
        file_name = os.path.split(path)[-1].split('.')[0]

        ck_db_table = f'hq_quote.{file_name}'
        create_table_sql = create_table_ck(ck_db_table=ck_db_table, sqlite_path=path, sqlite_table='option')
        conn(create_table_sql)


if __name__ == '__main__':
    conn = BaseSingleFactorTableNode("clickhouse://default:Imsn0wfree@0.0.0.0:8123/system")

    download_hq_quote(working_period=(('09:30:00', '11:30,00'),
                                      ('13:00:00', '15:00:00')),
                      store_path='/home/ll/hq_data/MO',
                      force=False,
                      conn=conn
                      )

    print(1)
    pass
