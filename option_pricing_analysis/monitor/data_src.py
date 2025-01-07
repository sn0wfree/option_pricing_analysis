# coding=utf-8

import warnings
from io import BytesIO

import akshare as ak
import pandas as pd
import requests
from CodersWheel.QuickTool.retry_it import conn_try_again

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决‘-’bug

from CodersWheel.QuickTool.timer import timer

import io


def parse_df(content):
    # 将文件内容转换为 BytesIO 对象
    buffer = io.BytesIO(content)
    return pd.read_parquet(buffer)


class RealTimeIdxQuote(object):
    @staticmethod
    @conn_try_again(max_retries=5, default_retry_delay=1, expect_error=Exception)
    def realtime_idx():
        realtime_idex = ak.stock_zh_index_spot_em(symbol="指数成份")

        rm_idex_selected = \
            realtime_idex[realtime_idex['名称'].isin(['沪深300', '中证500', '中证1000'])].drop_duplicates('名称')[
                ['名称', '最新价']]
        rm_idex_selected.columns = ['underlying', 'underlying_price']

        return rm_idex_selected


class NonRealTimeQuote(object):
    pass


# from WindPy import w
# w.start()





if __name__ == '__main__':
    import time
    for i in range(100):
        idx_dat = QuoteHolder.get_idx_minute_quote_via_ak('000852')

        op_dat = QuoteHolder.get_op_quote_via_cffex(symbol='mo', end_month="ALL")
        time.sleep(1)

    # print(op_dat)
    pass
