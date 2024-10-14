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

def query(url):
    x = requests.get(url)
    dat = parse_df(x.content)
    return dat


class QuoteHolder(object):
    @staticmethod
    @timer
    def get_op_quote_via_cffex(symbol='mo', end_month="ALL", drop_ym=True, base_host_port='47.104.186.157:3100'):
        url = f'http://{base_host_port}/ak/op/{symbol}/{end_month}'
        # 获取行情数据
        t_trading_board = query(url)

        if not drop_ym:
            return t_trading_board
        elif end_month != 'ALL':
            return t_trading_board.drop('end_month', axis=1)
        else:
            return t_trading_board

    @staticmethod
    @timer
    def get_idx_minute_quote_via_ak(symbol='000852', base_host_port='47.104.186.157:3100'):
        url = f'http://{base_host_port}/ak/idx/{symbol}/1'
        dat = query(url)
        dat['代码'] = symbol

        return dat


class PositionHolder(object):

    @staticmethod
    def get_current_position():
        # 假设的持仓数据
        positions_dict = {
            'MO2410-P-4300.CFE': {'Quantity': 10, 'Price': 10.75, 'Delta': 0.5, 'Gamma': 0.05, 'Theta': -0.05,
                                  'Vega': 0.1},
            'MO2410-P-4200.CFE': {'Quantity': 5, 'Price': 10.50, 'Delta': 0.5, 'Gamma': 0.05, 'Theta': -0.05,
                                  'Vega': 0.1}
        }

        positions = pd.DataFrame(positions_dict).T
        positions.index.name = 'code'
        positions = positions.reset_index()

        return positions


if __name__ == '__main__':
    import time
    for i in range(100):
        idx_dat = QuoteHolder.get_idx_minute_quote_via_ak('000852')

        op_dat = QuoteHolder.get_op_quote_via_cffex(symbol='mo', end_month="ALL")
        time.sleep(1)

    # print(op_dat)
    pass
