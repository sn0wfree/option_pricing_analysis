# coding=utf-8
import datetime

import pandas as pd
from ClickSQL import BaseSingleFactorTableNode
from CodersWheel.QuickTool.file_cache import file_cache


class WindHelper(object):
    """
        A helper class for interacting with the Wind financial database.
    """

    def __init__(self):
        from WindPy import w
        w.start()
        self.w = w

    @file_cache(enable_cache=True, granularity='d')
    def get_data(self, code, start, end, fields):
        """
        Generic method to fetch data from the Wind database.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :param fields: The fields to retrieve (e.g., 'close', 'high').
        :return: DataFrame with the requested data.
        """
        error, data_frame = self.w.wsd(code, fields, start, end, "", usedf=True)
        if error != 0:
            raise ValueError(f"Wind API error: {error}")
        return data_frame

    def get_close_prices(self, code, start, end):
        """
        Retrieve closing prices for a given code and date range.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :return: DataFrame with closing prices.
        """
        return self.get_data(code, start, end, 'close')

    def wind_wsd_high(self, code: str, start: str, end: str):
        return self.get_data(code, start, end, "high")

    def wind_wsd_low(self, code: str, start: str, end: str):
        return self.get_data(code, start, end, "low")

    def wind_wsd_open(self, code: str, start: str, end: str):
        """

        :param code:
        :param start:
        :param end:
        :return:
        """
        return self.get_data(code, start, end, "open")

    def wind_wsd_volume(self, code: str, start: str, end: str):
        """

        :param code:
        :param start:
        :param end:
        :return:
        """
        return self.get_data(code, start, end, "volume")

    def wind_wsd_quote(self, code: str, start: str, end: str, required_cols=('open', 'high', 'low', 'close', 'volume')):
        """

        :param code:
        :param start:
        :param end:
        :param required_cols:
        :return:
        """

        res_generator = (self.get_data(code, start, end, col) for col in required_cols)
        df = pd.concat(res_generator, axis=1)
        df['symbol'] = code

        return df

    def wind_wsd_quote_reduce(self, code: str, start: str, end: str, required_cols=('close', 'volume')):
        df = self.wind_wsd_quote(code, start, end, required_cols=required_cols)
        return df

    @file_cache(enable_cache=True, granularity='d')
    def get_future_info_last_delivery_date_underlying(self, code_list=None,
                                                      date_fields=['EXE_ENDDATE', 'LASTDELIVERY_DATE', 'FTDATE_NEW',
                                                                   'STARTDATE'],
                                                      multiplier_fields=['CONTRACTMULTIPLIER'],
                                                      underlying_code=['UNDERLYINGWINDCODE'],
                                                      futures_margin=['MARGIN']
                                                      ):
        """
        Retrieve future information including last delivery dates and contract multipliers.

        :param code_list: List of future contract codes.
        :return: DataFrame with future information.
        """
        if code_list is None:
            code_list = ["IF2312.CFE"]

        cols_str = ",".join(date_fields + multiplier_fields + underlying_code + futures_margin).lower()
        # "ftdate_new,startdate,lastdelivery_date,exe_enddate,contractmultiplier"

        err, last_deliv_and_multi = self.w.wss(','.join(code_list), cols_str, usedf=True)

        if err != 0:
            raise ValueError(f"Wind API error: {err}")

        for field in date_fields:
            last_deliv_and_multi[field] = last_deliv_and_multi[field].replace('1899-12-30', None)
            # last_deliv_and_multi[field] = pd.to_datetime(last_deliv_and_multi[field], errors='coerce')

        last_deliv_and_multi['EXE_DATE'] = last_deliv_and_multi['EXE_ENDDATE'].combine_first(
            last_deliv_and_multi['LASTDELIVERY_DATE'])
        last_deliv_and_multi['START_DATE'] = last_deliv_and_multi['STARTDATE'].combine_first(
            last_deliv_and_multi['FTDATE_NEW'])

        return last_deliv_and_multi

    def _get_quote_info(self, lastdeliv_and_multi, quote_start_with: str, existed_dt_dict=None):
        today = datetime.datetime.today()

        existed_dt_dict = {} if existed_dt_dict is None else existed_dt_dict

        quote_start_with = pd.to_datetime(quote_start_with).strftime("%Y-%m-%d")

        min_start_dt = pd.to_datetime(min(quote_start_with, today.strftime("%Y-%m-%d")))
        got = []

        # 获取合约行情
        for contract, params in lastdeliv_and_multi.iterrows():
            init_start = max(params['START_DATE'], existed_dt_dict.get(contract, min_start_dt)) - pd.DateOffset(days=10)
            # start = min(init_start, params['START_DATE'])
            end = min(today, params['EXE_DATE']).strftime("%Y-%m-%d")
            q = self.wind_wsd_quote_reduce(contract, init_start, end, required_cols=('close',)).dropna()[
                'CLOSE'].to_frame(contract)
            q.index.name = 'date'
            yield q
            got.append(contract)

        # 获取underlyling 行情
        for underlying in lastdeliv_and_multi['UNDERLYINGWINDCODE'].dropna().unique():
            if underlying not in got:
                init_start = max(min_start_dt, existed_dt_dict.get(underlying, min_start_dt)) - pd.DateOffset(days=10)
                q = \
                    self.wind_wsd_quote_reduce(underlying, init_start, today.strftime("%Y-%m-%d"),
                                               required_cols=('close',)).dropna()[
                        'CLOSE'].to_frame(underlying)
                q.index.name = 'date'
                yield q

    def get_data_fast(self, contracts, quote_start_with, existed_dt_dict=None):

        # contracts = self._transactions['委托合约'].unique().tolist() if contracts is None else contracts
        lastdel_multi = self.get_future_info_last_delivery_date_underlying(contracts)
        lastdel_multi.index.name = '委托合约'
        lastdel_multi = lastdel_multi[['START_DATE', 'EXE_DATE', 'CONTRACTMULTIPLIER', 'UNDERLYINGWINDCODE', 'MARGIN']]

        h = self._get_quote_info(lastdel_multi, quote_start_with, existed_dt_dict=existed_dt_dict)

        quote = pd.concat(h, axis=1).sort_index()
        quote.index = pd.to_datetime(quote.index)
        return lastdel_multi, quote


class WindBaseDatabase(WindHelper):
    def __init__(self, src: str, db='wind_data'):
        super(WindBaseDatabase, self).__init__()
        self.db = db
        self.src = src

    @staticmethod
    def cached_quote(contract_code_list, db, query):
        code_str = " ',' ".join(contract_code_list)
        # wind_data
        sql = f"select date,code,close from {db}.quote where code in ('{code_str}')  "
        return query(sql)

    @staticmethod
    def cached_derivative_info(contract_code_list, db, query):
        code_str = " ',' ".join(contract_code_list)
        # wind_data
        sql = f"select * from {db}.derivative_info where contract_code in ('{code_str}')  "
        return query(sql)

    def get_new_quote_info(self, contract_code_list, db, query, quote_start_with='2022-06-04'):
        code_str = " ',' ".join(contract_code_list)
        sql = f"select code,max(date) as max_date from {db}.quote where code in ('{code_str}')  group by code "
        quote_max_dt_df = query(sql)
        existed_dt_dict = dict(quote_max_dt_df.values)

        derivative_info = query(
            f"select contract_code, exe_date from {db}.derivative_info where contract_code in ('{code_str}')   ")

        merged = pd.merge(quote_max_dt_df, derivative_info, left_on='code', right_on='contract_code')
        if merged.empty:
            quote_start_with = quote_start_with
        else:
            merged['expired'] = merged['exe_date'] < merged['max_date']
            if merged[~merged['expired']].empty:
                quote_start_with = quote_start_with
            else:
                quote_start_with = merged[~merged['expired']]['max_date'].min()
        lastdel_multi, quote = self.get_data_fast(contract_code_list, quote_start_with, existed_dt_dict=existed_dt_dict)
        return lastdel_multi, quote

    @staticmethod
    def parse_quote_and_lastdel_multi(lastdel_multi, quote):
        q = quote.stack(-1).reset_index()
        q.columns = ['date', 'code', 'close']

        lastdel_multi_q = lastdel_multi.reset_index()
        lastdel_multi_q.columns = ['contract_code', 'start_date', 'exe_date', 'CONTRACTMULTIPLIER'.lower(),
                                   'UNDERLYINGWINDCODE'.lower(), 'MARGIN'.lower()]

        return q, lastdel_multi_q

    def _get_data_fusion_(self, contract_code_list, db, query, insert, quote_start_with='2022-06-04'):
        new_lastdel_multi, new_quote = self.get_new_quote_info(contract_code_list, db, query)

        q, lastdel_multi_q = self.parse_quote_and_lastdel_multi(new_lastdel_multi, new_quote)

        insert(q, lastdel_multi_q)

        lastdel_multi, quote = self.get_data_fast(contract_code_list, quote_start_with)

        return lastdel_multi, quote

    def get_data_fusion(self, contract_code_list, quote_start_with='2022-06-04'):
        query, insert = self.create_query_insert_func(self.src, self.db)
        return self._get_data_fusion_(contract_code_list, self.db, query, insert, quote_start_with=quote_start_with)

    @staticmethod
    def create_query_insert_func(src, db):
        node_local = BaseSingleFactorTableNode(src)

        def insert(quote, lastdel_multi):
            node_local.insert_df(quote, db, 'quote')
            node_local.insert_df(lastdel_multi, db, 'derivative_info')
            node_local(f'optimize table {db}.quote final')
            node_local(f'optimize table {db}.derivative_info final')

        return node_local, insert


if __name__ == '__main__':
    src = 'clickhouse://default:Imsn0wfree@10.67.20.52:8123/system'

    contract_code_list = ['MO2409-P-4500.CFE', 'MO2409-P-4450.CFE']

    wh = WindBaseDatabase(src=src, db='wind_data')

    lastdel_multi, quote = wh.get_data_fusion(contract_code_list, quote_start_with='2022-06-04')

    print(1)

    pass
