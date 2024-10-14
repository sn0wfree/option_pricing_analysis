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

from CodersWheel.QuickTool.file_cache import file_cache

symbol_2_key_dict = {'mo': '中证1000指数', 'ho': '上证50指数', 'io': '沪深300指数'}
symbol_2_api_symbol_dict = {'mo': 'zz1000', 'ho': 'sz50', 'io': 'hs300'}
symbol_2_board_symbol_dict = {'mo': '中证1000股指期权', 'ho': '上证50股指期权', 'io': '沪深300股指期权'}

# 中国金融期货交易所
CFFEX_OPTION_URL_50 = "http://www.cffex.com.cn/quote_HO.txt"
CFFEX_OPTION_URL_300 = "http://www.cffex.com.cn/quote_IO.txt"
CFFEX_OPTION_URL_1000 = "http://www.cffex.com.cn/quote_MO.txt"
# 深圳证券交易所

SZ_OPTION_URL_300 = "http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=ysplbrb&TABKEY=tab1&random=0.10432465776720479"

# 上海证券交易所

SH_OPTION_URL_50 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510050"
SH_OPTION_URL_KING_50 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510050_{}"

SH_OPTION_URL_300 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510300"
SH_OPTION_URL_KING_300 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510300_{}"

SH_OPTION_URL_500 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/510500"
SH_OPTION_URL_KING_500 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/510500_{}"

SH_OPTION_URL_KC_50 = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/588000"
SH_OPTION_URL_KC_KING_50 = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/588000_{}"

SH_OPTION_URL_KC_50_YFD = "http://yunhq.sse.com.cn:32041/v1/sh1/list/self/588080"
SH_OPTION_URL_KING_50_YFD = "http://yunhq.sse.com.cn:32041/v1/sho/list/tstyle/588080_{}"

SH_OPTION_PAYLOAD = {
    "select": "select: code,name,last,change,chg_rate,amp_rate,volume,amount,prev_close"
}

SH_OPTION_PAYLOAD_OTHER = {
    "select": "contractid,last,chg_rate,presetpx,exepx"
}


def option_finance_board(
        symbol: str = "嘉实沪深300ETF期权", end_month: str = "2306"
) -> pd.DataFrame:
    """
    期权当前交易日的行情数据
    主要为三个: 华夏上证50ETF期权, 华泰柏瑞沪深300ETF期权, 嘉实沪深300ETF期权,
    沪深300股指期权, 中证1000股指期权, 上证50股指期权, 华夏科创50ETF期权, 易方达科创50ETF期权
    http://www.sse.com.cn/assortment/options/price/
    http://www.szse.cn/market/product/option/index.html
    http://www.cffex.com.cn/hs300gzqq/
    http://www.cffex.com.cn/zz1000gzqq/
    :param symbol: choice of {"华夏上证50ETF期权", "华泰柏瑞沪深300ETF期权", "南方中证500ETF期权", "华夏科创50ETF期权", "易方达科创50ETF期权", "嘉实沪深300ETF期权", "沪深300股指期权", "中证1000股指期权", "上证50股指期权"}
    :type symbol: str
    :param end_month: 2003; 2020 年 3 月到期的期权
    :type end_month: str
    :return: 当日行情
    :rtype: pandas.DataFrame
    """
    end_month = end_month[-2:]
    if symbol == "华夏上证50ETF期权":
        r = requests.get(
            SH_OPTION_URL_KING_50.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = ["日期", "合约交易代码", "当前价", "涨跌幅", "前结价", "行权价", "数量"]
        return raw_data
    elif symbol == "华泰柏瑞沪深300ETF期权":
        r = requests.get(
            SH_OPTION_URL_KING_300.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = ["日期", "合约交易代码", "当前价", "涨跌幅", "前结价", "行权价", "数量"]
        return raw_data
    elif symbol == "南方中证500ETF期权":
        r = requests.get(
            SH_OPTION_URL_KING_500.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = ["日期", "合约交易代码", "当前价", "涨跌幅", "前结价", "行权价", "数量"]
        return raw_data
    elif symbol == "华夏科创50ETF期权":
        r = requests.get(
            SH_OPTION_URL_KC_KING_50.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = ["日期", "合约交易代码", "当前价", "涨跌幅", "前结价", "行权价", "数量"]
        return raw_data
    elif symbol == "易方达科创50ETF期权":
        r = requests.get(
            SH_OPTION_URL_KING_50_YFD.format(end_month),
            params=SH_OPTION_PAYLOAD_OTHER,
        )
        data_json = r.json()
        raw_data = pd.DataFrame(data_json["list"])
        raw_data.index = [str(data_json["date"]) + str(data_json["time"])] * data_json[
            "total"
        ]
        raw_data.columns = ["合约交易代码", "当前价", "涨跌幅", "前结价", "行权价"]
        raw_data["数量"] = [data_json["total"]] * data_json["total"]
        raw_data.reset_index(inplace=True)
        raw_data.columns = ["日期", "合约交易代码", "当前价", "涨跌幅", "前结价", "行权价", "数量"]
        return raw_data
    elif symbol == "嘉实沪深300ETF期权":
        url = "http://www.szse.cn/api/report/ShowReport/data"
        params = {
            "SHOWTYPE": "JSON",
            "CATALOGID": "ysplbrb",
            "TABKEY": "tab1",
            "PAGENO": "1",
            "random": "0.10642298535346595",
        }
        r = requests.get(url, params=params)
        data_json = r.json()
        page_num = data_json[0]["metadata"]["pagecount"]
        big_df = pd.DataFrame()
        for page in range(1, page_num + 1):
            params = {
                "SHOWTYPE": "JSON",
                "CATALOGID": "ysplbrb",
                "TABKEY": "tab1",
                "PAGENO": page,
                "random": "0.10642298535346595",
            }
            r = requests.get(url, params=params)
            data_json = r.json()
            temp_df = pd.DataFrame(data_json[0]["data"])
            big_df = pd.concat([big_df, temp_df], ignore_index=True)

        big_df.columns = [
            "合约编码",
            "合约简称",
            "标的名称",
            "类型",
            "行权价",
            "合约单位",
            "期权行权日",
            "行权交收日",
        ]
        big_df["期权行权日"] = pd.to_datetime(big_df["期权行权日"])
        big_df["end_month"] = big_df["期权行权日"].dt.month.astype(str).str.zfill(2)
        big_df = big_df[big_df["end_month"] == end_month]
        del big_df["end_month"]
        big_df.reset_index(inplace=True, drop=True)
        return big_df
    elif symbol == "沪深300股指期权":
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        }
        r = requests.get(CFFEX_OPTION_URL_300, headers=headers)
        raw_df = pd.read_table(BytesIO(r.content), sep=",")
        raw_df["end_month"] = (
            raw_df["instrument"]
            .str.split("-", expand=True)
            .iloc[:, 0]
            .str.slice(
                4,
            )
        )
        raw_df = raw_df[raw_df["end_month"] == end_month]
        del raw_df["end_month"]
        raw_df.reset_index(inplace=True, drop=True)
        return raw_df
    elif symbol == "中证1000股指期权":
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        }
        url = "http://www.cffex.com.cn/quote_MO.txt"
        r = requests.get(url, headers=headers)
        raw_df = pd.read_table(BytesIO(r.content), sep=",")
        raw_df["end_month"] = (
            raw_df["instrument"]
            .str.split("-", expand=True)
            .iloc[:, 0]
            .str.slice(
                4,
            )
        )
        raw_df = raw_df[raw_df["end_month"] == end_month]
        del raw_df["end_month"]
        raw_df.reset_index(inplace=True, drop=True)
        return raw_df
    elif symbol == "上证50股指期权":
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
        }
        url = "http://www.cffex.com.cn/quote_HO.txt"
        r = requests.get(url, headers=headers)
        raw_df = pd.read_table(BytesIO(r.content), sep=",")
        raw_df["end_month"] = (
            raw_df["instrument"]
            .str.split("-", expand=True)
            .iloc[:, 0]
            .str.slice(
                4,
            )
        )
        raw_df = raw_df[raw_df["end_month"] == end_month]
        del raw_df["end_month"]
        raw_df.reset_index(inplace=True, drop=True)
        return raw_df


def option_finance_board_CFFEX(symbol: str = "沪深300股指期权"):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
    }

    if symbol == "沪深300股指期权":
        url = CFFEX_OPTION_URL_300
    elif symbol == "中证1000股指期权":
        url = CFFEX_OPTION_URL_1000
    elif symbol == "上证50股指期权":
        url = CFFEX_OPTION_URL_50
    else:
        raise ValueError(f'unknown symbol:{symbol}')

    r = requests.get(url, headers=headers)
    raw_df = pd.read_table(BytesIO(r.content), sep=",")
    raw_df["end_month"] = (raw_df["instrument"].str.split("-", expand=True).iloc[:, 0].str.slice(4, ))

    raw_df.reset_index(inplace=True, drop=True)
    return raw_df
    # elif
    #
    #     r = requests.get(CFFEX_OPTION_URL_1000, headers=headers)
    #     raw_df = pd.read_table(BytesIO(r.content), sep=",")
    #     raw_df["end_month"] = (
    #         raw_df["instrument"]
    #         .str.split("-", expand=True)
    #         .iloc[:, 0]
    #         .str.slice(
    #             4,
    #         )
    #     )
    #     raw_df = raw_df[raw_df["end_month"] == end_month]
    #     del raw_df["end_month"]
    #     raw_df.reset_index(inplace=True, drop=True)
    #     return raw_df
    # elif symbol == "上证50股指期权":
    #
    #     r = requests.get(CFFEX_OPTION_URL_50, headers=headers)
    #     raw_df = pd.read_table(BytesIO(r.content), sep=",")
    #     raw_df["end_month"] = (
    #         raw_df["instrument"]
    #         .str.split("-", expand=True)
    #         .iloc[:, 0]
    #         .str.slice(
    #             4,
    #         )
    #     )
    #     raw_df = raw_df[raw_df["end_month"] == end_month]
    #     del raw_df["end_month"]
    #     raw_df.reset_index(inplace=True, drop=True)
    #     return raw_df


class DelayFutOptionQuote(object):
    @staticmethod
    @file_cache(enable_cache=True, granularity='d')
    @conn_try_again(max_retries=5, default_retry_delay=1, expect_error=Exception)
    def get_current_cffex_op_list(symbol='mo',
                                  symbol_2_key_dict={'mo': '中证1000指数', 'ho': '上证50指数', 'io': '沪深300指数'},
                                  symbol_2_api_symbol_dict={'mo': 'zz1000', 'ho': 'sz50', 'io': 'hs300'}):

        key = symbol_2_key_dict.get(symbol, None)
        api_symbol = symbol_2_api_symbol_dict.get(symbol, None)
        if key is None:
            raise ValueError(f'symbol only accept :{symbol_2_key_dict.keys()}')

        option_cffex_op_list_sina_df = getattr(ak, f'option_cffex_{api_symbol}_list_sina')()

        return option_cffex_op_list_sina_df[key]

    @staticmethod
    # @file_cache(enable_cache=True, granularity='M')
    @conn_try_again(max_retries=5, default_retry_delay=1, expect_error=Exception)
    def get_all_single_ym_symbol_op_quote(symbol='mo', end_month="2408",
                                          symbol_2_board_symbol_dic=symbol_2_board_symbol_dict):
        symbol = symbol_2_board_symbol_dict.get(symbol, None)
        if symbol is None:
            raise ValueError(f'symbol only accept :{symbol_2_board_symbol_dic.keys()}')
        raw_df = option_finance_board_CFFEX(symbol=symbol, )

        if end_month == 'ALL':
            option_finance_board_df = raw_df
        else:
            option_finance_board_df = raw_df[raw_df["end_month"] == end_month[-2:]]

        return option_finance_board_df

    @staticmethod
    def rename_df(call, direct='看涨'):

        cols = {'position': f'{direct}持仓量',
                'volume': f'{direct}成交量',
                'lastprice': f'{direct}最新成交价',
                'updown': f'{direct}涨跌',
                'pct': f'{direct}涨跌幅',
                'bprice': f'{direct}买一价格',
                'bamount': f'{direct}买一手数',
                'sprice': f'{direct}卖一价格',
                'samount': f'{direct}卖一手数',

                'k': '行权价', 'instrument': f'{direct}合约'}
        if direct == '看涨':
            col_order = ['instrument', 'updown', 'pct', 'volume', 'position', 'lastprice', 'end_month', 'k']
        else:
            col_order = ['k', 'end_month', 'lastprice', 'position', 'volume', 'pct', 'updown', 'instrument', ]
        return call[col_order].rename(columns=cols)

    @classmethod
    # @file_cache(enable_cache=True, granularity='H')
    def t_trading_board(cls, symbol='mo', end_month="2408"):
        option_finance_board_df = cls.get_all_single_ym_symbol_op_quote(symbol=symbol, end_month=end_month)

        option_finance_board_df['pct'] = option_finance_board_df['updown'] / (
                option_finance_board_df['lastprice'] - option_finance_board_df['updown'])

        option_finance_board_df['pct'] = option_finance_board_df['pct'].round(4)

        ym_d_k = option_finance_board_df['instrument'].apply(lambda x: x.split('-'))
        s_ym_d_k = pd.DataFrame(ym_d_k.values.tolist(), columns=['symbol_ym', 'direct', 'k'])
        s_ym_d_k['instrument'] = option_finance_board_df['instrument']

        merged_op_quote = pd.merge(s_ym_d_k, option_finance_board_df, left_on=['instrument'], right_on=['instrument'])

        call = merged_op_quote[merged_op_quote['direct'] == 'C']
        renamed_call = cls.rename_df(call, direct='看涨')
        put = merged_op_quote[merged_op_quote['direct'] == 'P']
        renamed_put = cls.rename_df(put, direct='看跌')
        return pd.merge(renamed_call, renamed_put, left_on=['行权价', 'end_month'],
                        right_on=['行权价', 'end_month'])


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


class QuoteHolder(object):
    @staticmethod
    def get_op_quote_via_cffex(symbol='mo', end_month="2410", drop_ym=True):
        # 获取行情数据
        t_trading_board = DelayFutOptionQuote.t_trading_board(symbol=symbol, end_month=end_month)

        if not drop_ym:
            return t_trading_board
        else:
            return t_trading_board.drop('end_month', axis=1)

    @staticmethod
    def get_idx_quote_via_ak():

        return RealTimeIdxQuote.realtime_idx()





class PositionHolder(object):


    @staticmethod
    def get_current_position():
        # 假设的持仓数据
        positions_dict = {
            'MO2410-P-4300.CFE': {'Quantity': 10, 'Price': 10.75,'Delta': 0.5, 'Gamma': 0.05, 'Theta': -0.05, 'Vega': 0.1},
            'MO2410-P-4200.CFE': {'Quantity': 5, 'Price': 10.50,'Delta': 0.5, 'Gamma': 0.05, 'Theta': -0.05, 'Vega': 0.1}
        }

        positions = pd.DataFrame(positions_dict).T
        positions.index.name = 'code'
        positions = positions.reset_index()

        return positions



if __name__ == '__main__':
    mo_q = QuoteHolder.get_op_quote_via_cffex(symbol='mo', end_month="ALL", drop_ym=True)

    print(1)
    pass