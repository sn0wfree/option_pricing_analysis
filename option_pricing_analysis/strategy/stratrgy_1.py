# coding=utf-8
import datetime
import os

import akshare as ak
import pandas as pd
from ClickSQL import BaseSingleFactorTableNode
from CodersWheel.QuickTool.boost_up import boost_up
from CodersWheel.QuickTool.file_cache import file_cache

from option_pricing_analysis.Model.BSM import ImpliedVolatility, OptionPricing


def get_info_last_delivery_multi(contracts, wind_helper: object):
    # contracts = self._transactions['委托合约'].unique().tolist() if contracts is None else contracts
    lastdeliv_and_multi = wind_helper.get_future_info_last_delivery_date_underlying(contracts)
    lastdeliv_and_multi.index.name = '委托合约'
    return lastdeliv_and_multi[['START_DATE', 'EXE_DATE', 'CONTRACTMULTIPLIER', 'UNDERLYINGWINDCODE']]


@file_cache(enable_cache=True)
def get_market_index_close_data_via_ak(w: object, code_list: list, code_name: list, start="2020-12-31",
                                       end=datetime.datetime.today().strftime("%Y-%m-%d"), period='W',

                                       comm_fut_code_dict={'NH0100.NHF': 'NHCI'}
                                       ):
    # code_strings = ','.join(code_list)

    # "000852.SH,000905.SH,000300.SH,NH0100.NHF,399372.SZ,399373.SZ,399376.SZ,399377.SZ",
    # error, idx = w.wsd(code_strings, "close", start, end, f"Period={period}", usedf=True)

    def dymanic_index_daily(symbol, start, end):

        if symbol in comm_fut_code_dict.keys():
            idx = ak.futures_return_index_nh(symbol=comm_fut_code_dict.get(symbol)).set_index('date')[
                'value'].to_frame(symbol)
            idx.index = pd.to_datetime(idx.index)
        else:
            code, exchangecode = symbol.split('.')
            try:
                idx = ak.stock_zh_index_daily(symbol=exchangecode.lower() + code).set_index('date')[
                    'close'].to_frame(symbol)
                idx.index = pd.to_datetime(idx.index)
            except KeyError as e:
                raise KeyError(f'unknown symbol for stock index:{symbol}')

        idx_mask = idx[(idx.index <= end) & (idx.index >= start)]
        return idx_mask[~idx_mask.index.duplicated()]

    h = [dymanic_index_daily(symbol, start, end) for symbol in code_list]

    idx_merged = pd.concat(h, axis=1)
    idx_merged['date'] = idx_merged.index

    idx_merged = idx_merged.resample(period).last().set_index('date').dropna()
    idx_merged.columns = code_name
    return idx_merged


class MOQuoteQuickMatrix(object):
    def __init__(self, conn, start: str = '2020-01-01', end='2999-12-31', underlying_endpos=2,
                 ym_length=4,
                 cp_length=3,

                 method='BSPricing',
                 ):
        _sub_sql = f"select * from quote.view_quote_mo where trade_date>='{start}' and trade_date<='{end}'"

        self._iv_func = ImpliedVolatility(pricing_f=OptionPricing, method=method).implied_volatility_brent

        self._conn = conn
        self._raw = self.query(_sub_sql)

        # self._underlying_quote = underlying_quote

        self._raw['prefix'] = self._raw['contract_code'].apply(lambda x: x[:underlying_endpos])

        self._raw['ym'] = self._raw['contract_code'].apply(lambda x: x[underlying_endpos:underlying_endpos + ym_length])

        self._raw['cp'] = self._raw['contract_code'].apply(
            lambda x: x[underlying_endpos + ym_length:underlying_endpos + ym_length + cp_length].strip('-').lower())

        self._raw['strike'] = self._raw['contract_code'].apply(
            lambda x: int(x[underlying_endpos + ym_length + cp_length:].split('.')[0]))

    @file_cache(enable_cache=True, granularity='d')
    def query(self, sql):
        return self._conn(sql)

    def _fast_group_iter_(self, by=['trade_date', 'ym'], dt_start=None, ym_start=None, strike_start=None,
                          cp=['p', 'c'], prefix='mo'):
        if dt_start is None:
            dt_start = self._raw['trade_date'].min()

        if ym_start is None:
            ym_start = self._raw['ym'].min()

        if strike_start is None:
            strike_start = self._raw['strike'].min()

        dt_mask = self._raw['trade_date'] >= f'{dt_start}'
        ym_mask = self._raw['ym'] >= f'{ym_start}'
        strike_mask = self._raw['strike'] >= strike_start
        cp_mask = self._raw['cp'].isin(cp)
        prefix_mask = self._raw['prefix'].str.lower() == prefix.lower()

        if 'cp' not in by:
            by = by + ['cp']

        data = self._raw[dt_mask & ym_mask & strike_mask & cp_mask & prefix_mask]

        for by_item, item in data.groupby(by):
            yield by_item, item

    def _diff_k_by_ym_(self, underlying_quote, lastdel_multi,
                       dt_start=None,
                       ym_start=None,
                       strike_start=None,
                       cp=['p', 'c'],
                       prefix='mo',
                       r=0.015, g=0, ):
        for by_item, df in self._fast_group_iter_(by=['trade_date', 'ym'], dt_start=dt_start, ym_start=ym_start,
                                                  strike_start=strike_start,
                                                  cp=cp, prefix=prefix):
            dt = df['trade_date'].unique().tolist()[0]
            ym = df['ym'].unique().tolist()[0]
            cp_sign = df['cp'].replace({"c": 1, "p": -1}).unique().tolist()[0]

            contracts = df['contract_code'].tolist()

            EXE_DATE = \
                lastdel_multi[lastdel_multi.index.isin(contracts)].reindex(index=contracts)[
                    'EXE_DATE'].unique().tolist()[0]

            t = underlying_quote[(underlying_quote.index <= EXE_DATE) & (underlying_quote.index > dt)].shape[0] / 250

            s = underlying_quote.loc[dt, '中证1000']

            func = self._iv_func

            for k, fee in df[['strike', 'close']].values:
                iv, diff = func(s, k, r, t, fee, cp_sign, g)

                # tasks = [(s, k, r, t, fee, cp_sign, g) for k, fee in df[['strike', 'close']].values]

                # result = boost_up(func, tasks, star=True)
                # for (k, fee), (iv, diff) in zip(df[['strike', 'close']].values, result):
                # iv, diff = self._iv_func(s, k, r, t, fee, cp_sign, g, method='BSPricing')
                # # dt ym cp k fee iv
                yield *by_item, k, fee, iv, diff


def load_lastdel_multi(path: (str, list)):
    if isinstance(path, str) and os.path.exists(path):
        return pd.read_excel(path)
    else:
        from option_pricing_analysis.analysis.option_analysis_monitor import WindHelper

        wh = WindHelper()

        lastdel_multi = get_info_last_delivery_multi(path, wh)
        return lastdel_multi


if __name__ == '__main__':
    #
    node = BaseSingleFactorTableNode('clickhouse://default:Imsn0wfree@47.104.186.157:8123/system')

    sql = "select * from quote.view_quote_mo where trade_date>='2020-01-01'"

    zz1000 = get_market_index_close_data_via_ak(None, ['000852.SH'], ['中证1000'], start="2020-12-31",
                                                end=datetime.datetime.today().strftime("%Y-%m-%d"), period='D',
                                                comm_fut_code_dict={'NH0100.NHF': 'NHCI'})

    moqqm = MOQuoteQuickMatrix(node, start='2020-01-01', end='2099-12-31')

    contract_list = moqqm._raw['contract_code'].unique().tolist()

    lastdel_multi = load_lastdel_multi('lastdel_multi.xlsx')

    # lastdel_multi = get_info_last_delivery_multi(contract_list, wh)
    # lastdel_multi.to_excel('lastdel_multi.xlsx')

    m = pd.DataFrame(list(moqqm._diff_k_by_ym_(zz1000, lastdel_multi, )),
                     columns=['dt', 'ym', 'cp', 'k', 'fee', 'iv', 'diff'])
    m.to_excel('test_iv.xlsx')
    print(1)
    pass
