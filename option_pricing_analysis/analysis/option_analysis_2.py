# coding=utf-8
import pandas as pd
import numpy as np
import re, datetime
from pf_analysis.utils.file_cache import file_cache
# from pf_analysis.load_data.load_data_from_gaoyu import cal_mdd_rolling_with_ir_calmar
import os
from WindPy import w
from glob import glob

import warnings
warnings.filterwarnings("ignore")

w.start()


class WindHelper(object):
    """
        A helper class for interacting with the Wind financial database.
    """

    @staticmethod
    def get_data(code, start, end, fields):
        """
        Generic method to fetch data from the Wind database.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :param fields: The fields to retrieve (e.g., 'close', 'high').
        :return: DataFrame with the requested data.
        """
        error, data_frame = w.wsd(code, fields, start, end, "", usedf=True)
        if error != 0:
            raise ValueError(f"Wind API error: {error}")
        return data_frame

    @classmethod
    def get_close_prices(cls, code, start, end):
        """
        Retrieve closing prices for a given code and date range.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :return: DataFrame with closing prices.
        """
        return cls.get_data(code, start, end, 'close')

    @classmethod
    def wind_wsd_high(cls, code: str, start, end):
        return cls.get_data(code, start, end, "high")

    @classmethod
    def wind_wsd_low(cls, code: str, start, end):
        return cls.get_data(code, start, end, "low")

    @classmethod
    def wind_wsd_open(cls, code: str, start, end):
        return cls.get_data(code, start, end, "open")

    @classmethod
    def wind_wsd_volume(cls, code: str, start, end):
        return cls.get_data(code, start, end, "volume")

    @classmethod
    def wind_wsd_quote(cls, code: str, start, end, required_cols=('open', 'high', 'low', 'close', 'volume')):
        res_generator = (cls.get_data(code, start, end, col) for col in required_cols)
        df = pd.concat(res_generator, axis=1)
        df['symbol'] = code

        return df

    @classmethod
    def wind_wsd_quote_reduce(cls, code: str, start, end, required_cols=('close', 'volume')):
        res_generator = (cls.get_data(code, start, end, col) for col in required_cols)
        df = pd.concat(res_generator, axis=1)
        df['symbol'] = code

        return df

    @staticmethod
    def get_future_info_last_delivery_date(code_list=None):
        """
        Retrieve future information including last delivery dates and contract multipliers.

        :param code_list: List of future contract codes.
        :return: DataFrame with future information.
        """
        if code_list is None:
            code_list = ["IF2312.CFE"]

        err, last_deliv_and_multi = w.wss(','.join(code_list),
                                          "ftdate_new,startdate,lastdelivery_date,exe_enddate,contractmultiplier,margin",#,undelyingwindcode
                                          usedf=True)

        if err != 0:
            raise ValueError(f"Wind API error: {err}")

        date_fields = ['EXE_ENDDATE', 'LASTDELIVERY_DATE', 'FTDATE_NEW', 'STARTDATE', 'MARGIN']
        for field in date_fields:
            last_deliv_and_multi[field] = last_deliv_and_multi[field].replace('1899-12-30', None)
            # last_deliv_and_multi[field] = pd.to_datetime(last_deliv_and_multi[field], errors='coerce')

        # last_deliv_and_multi['EXE_DATE'] = last_deliv_and_multi['EXE_ENDDATE'].combine_first(
        #     last_deliv_and_multi['LASTDELIVERY_DATE'])
        # last_deliv_and_multi['START_DATE'] = last_deliv_and_multi['STARTDATE'].combine_first(
        #     last_deliv_and_multi['FTDATE_NEW'])

        last_deliv_and_multi['EXE_DATE'] = [x['LASTDELIVERY_DATE'] if x['EXE_ENDDATE'] is pd.NaT else x['EXE_ENDDATE']
                                            for r, x in
                                            last_deliv_and_multi[['EXE_ENDDATE', 'LASTDELIVERY_DATE']].iterrows()]

        last_deliv_and_multi['START_DATE'] = [x['FTDATE_NEW'] if x['STARTDATE'] is pd.NaT else x['STARTDATE']
                                              for r, x in
                                              last_deliv_and_multi[['FTDATE_NEW', 'STARTDATE']].iterrows()]

        return last_deliv_and_multi

    @staticmethod
    def get_future_info_lastdelivery_date(code_list=["IF2312.CFE"]):
        err, lastdeliv_and_multi = w.wss(','.join(code_list),
                                         "ftdate_new,startdate,lastdelivery_date,exe_enddate,contractmultiplier",#,underlyingwindcode
                                         usedf=True)

        lastdeliv_and_multi['EXE_ENDDATE'] = lastdeliv_and_multi['EXE_ENDDATE'].replace('1899-12-30', None)
        lastdeliv_and_multi['LASTDELIVERY_DATE'] = lastdeliv_and_multi['LASTDELIVERY_DATE'].replace('1899-12-30', None)

        lastdeliv_and_multi['FTDATE_NEW'] = lastdeliv_and_multi['FTDATE_NEW'].replace('1899-12-30', None)
        lastdeliv_and_multi['STARTDATE'] = lastdeliv_and_multi['STARTDATE'].replace('1899-12-30', None)

        lastdeliv_and_multi['EXE_DATE'] = [x['LASTDELIVERY_DATE'] if x['EXE_ENDDATE'] is pd.NaT else x['EXE_ENDDATE']
                                           for r, x in
                                           lastdeliv_and_multi[['EXE_ENDDATE', 'LASTDELIVERY_DATE']].iterrows()]

        lastdeliv_and_multi['START_DATE'] = [x['FTDATE_NEW'] if x['STARTDATE'] is pd.NaT else x['STARTDATE']
                                             for r, x in
                                             lastdeliv_and_multi[['FTDATE_NEW', 'STARTDATE']].iterrows()]
        #lastdeliv_and_multi['UNDERLYING_CODE'] =

        return lastdeliv_and_multi


class OptionItem(object):

    def __init__(self, data_dict: (dict, pd.Series)):
        if isinstance(data_dict, dict) or isinstance(data_dict, pd.Series):
            self._single_transaction = data_dict

    @property
    def CONTRACTMULTIPLIER(self):
        return self._single_transaction['CONTRACTMULTIPLIER']

    @property
    def UNDERLYINGWINDCODE(self):
        return self._single_transaction['UNDERLYINGWINDCODE']

    @property
    def EXE_DATE(self):
        return self._single_transaction['EXE_DATE']

    @property
    def contract(self):
        return self._single_transaction['委托合约']

    @property
    def buy_sell(self):
        return self._single_transaction['买卖']

    @property
    def open_close(self):
        return self._single_transaction['开平']

    @property
    def number(self):
        return self._single_transaction['手数']

    @property
    def deal_price(self):
        return self._single_transaction['成交均价']

    @property
    def time(self):
        return self._single_transaction['报单时间']

    @property
    def date(self):
        return self._single_transaction['报单日期']

    # @property
    # def SETTLE(self):
    #     return self._single_transaction['SETTLE']

    def to_series(self):
        if self.LASTDELIVERY_DATE is None:
            raise ValueError('LASTDELIVERY_DATE is not defined!')
        if self.EXE_ENDDATE is None:
            raise ValueError('EXE_ENDDATE is not defined!')
        return self._single_transaction


class OptionItemHolder(object):

    def __init__(self, transactions: pd.DataFrame):
        self._holder = [OptionItem(series) for row, series in transactions.iterrows()]

    def __add__(self, other: (dict, pd.Series)):
        item = OptionItem(other)

        self._holder.append(item)

    def count(self):
        return len(self._holder)

    @property
    def transacted_contracts(self):
        return sorted(set(map(lambda x: x.contract, self._holder)))


def code_type_detect(code,
                     futures_pattern=re.compile(r'^[A-Z]{2}\d{4}\.\w{3}$'),
                     option_pattern=re.compile(r'^[A-Z]+[0-9]+-[CP]-[0-9]+\.\w+$'),
                     ):
    if futures_pattern.match(code):
        return 'Future'
    elif option_pattern.match(code):
        return 'Option'
    else:
        return 'Unknown'


class ProcessReportSingle(object):
    def __init__(self, dt, report_file_path: str = 'report.csv'):
        if isinstance(report_file_path, str) and report_file_path.endswith('csv'):
            self._report = pd.read_csv(report_file_path, encoding='GBK')
            self._report['报单日期'] = pd.to_datetime(str(dt), format='%Y%m%d') if isinstance(dt, str) else dt
        elif isinstance(report_file_path, pd.DataFrame):
            self._report = report_file_path
        else:
            raise ValueError(f'report_file_path only accept str or pd.DataFrame; but got {type(report_file_path)}')

        self.parse(self._report)

    def parse(self, _report, target_cols=['委托合约', '买卖', '开平', '手数', '成交均价', '报单时间', '报单日期']):
        ## process 买卖
        buysell_cols_replace = {'买\u3000': '买', '\u3000卖': '卖'}
        openclose_cols_replace = {'平今': '平','平昨':'平'}
        _report['买卖'] = _report['买卖'].replace(buysell_cols_replace)
        _report['开平'] = _report['开平'].replace(openclose_cols_replace)
        ## 全部成交
        all_settledown = _report[
            (_report['挂单状态'] == '全部成交') & (_report['未成交'] == 0)]
        ## 已撤单-部分成交
        partial_settledown = _report[
            (_report['挂单状态'] == '已撤单') & (_report['手数'] != _report['未成交'])].copy(deep=True)

        partial_settledown['手数'] = partial_settledown['手数'] - partial_settledown['未成交']
        partial_settledown['未成交'] = 0
        partial_settledown['挂单状态'] = '全部成交'

        self._parsed_report = pd.concat([all_settledown, partial_settledown])

        self._parsed_report['成交均价'] = self._parsed_report['成交均价'].astype(float)

        contracts = self._parsed_report[target_cols[0]].unique().tolist()

        contracts_translated_dict = ProcessReportSingle.parse_name_process(contracts)

        self._parsed_report[target_cols[0]] = self._parsed_report[target_cols[0]].replace(contracts_translated_dict)

        self._transactions = self._parsed_report[target_cols]

    @classmethod
    def parse_name_process(cls, contracts, rule: dict = {'MO\d{4}-[CP]-[0-9]+': 'CFE',
                                                         'HO\d{4}-[CP]-[0-9]+': 'CFE',
                                                         'IO\d{4}-[CP]-[0-9]+': 'CFE',
                                                         'IH\d{4}': 'CFE',
                                                         'IF\d{4}': 'CFE',
                                                         'IM\d{4}': 'CFE',
                                                         'AG\d{4}': 'SHF',
                                                         'AU\d{4}': 'SHF',
                                                         'AL\d{4}': 'SHF',
                                                         'CU\d{4}': 'SHF'}, ):

        suffix = tuple(set(rule.values()))
        h = {}

        for contract in contracts:
            if contract.endswith(suffix):
                pass
            else:
                for pattern, suffix_name in rule.items():

                    if re.compile(pattern).match(contract):
                        h[contract] = contract + '.' + suffix_name
                        break
        return h

    def reduced_contracts(self, ):

        return self._transactions['委托合约'].unique().tolist()

    @property
    def reduced_transactions(self):

        transactions = self._transactions

        transactions['amt_100'] = transactions['手数'] * transactions['成交均价']

        out = transactions.groupby(['委托合约', '买卖', '开平', '报单日期']).sum()[['手数', 'amt_100']].reset_index()
        out['成交均价'] = out['amt_100'] / out['手数']

        output_cols = ['委托合约', '买卖', '开平', '报单日期', '手数', '成交均价']

        return out[output_cols]

    def create_items(self, lastdeliv_and_multi, reduced=False, return_df=False):

        if reduced:
            transactions = self.reduced_transactions
        else:
            transactions = self._transactions

        res = pd.merge(transactions, lastdeliv_and_multi.reset_index(), left_on='委托合约', right_on='委托合约')

        if return_df:
            return res

        else:

            transaction_holders = OptionItemHolder(res)

            return transaction_holders

    def map_contracts_to_persons(self):

        mapping = {'ll': [], 'wj': [], 'gr': []}
        # 假设合约前缀与人员的映射规则
        prefix_rules = {
            'AU': 'wj', 'IF': 'wj', 'IH': 'wj','CU': 'wj',
            'AG': 'gr', 'AL': 'gr'
        }

        # 遍历所有合约
        for contract in self.reduced_contracts():
            matched = False
            # 应用映射规则
            for prefix, person in prefix_rules.items():
                if contract.startswith(prefix):
                    mapping[person].append(contract)
                    matched = True
                    break
            if not matched:
                mapping['ll'].append(contract)

        return mapping


class ProcessReport(ProcessReportSingle):

    def __init__(self,
                 report_file_path: str = 'C:\\Users\\linlu\\Documents\\GitHub\\pf_analysis\\pf_analysis\\optionanalysis\\report_file',
                 person_contract_mapping=None):#
        # self.person_contract_mapping = person_contract_mapping or self.default_person_contract_mapping
        _parsed_report = self.load_report(report_file_path)
        #self.contracts = self.reduced_contracts()


        super().__init__(None, _parsed_report)

    @staticmethod
    def load_report(
            report_file_path: str = 'C:\\Users\\linlu\\Documents\\GitHub\\pf_analysis\\pf_analysis\\optionanalysis\\report_file',
            daily_report='report*.csv',
            period_report='report*-*.xlsx',
            target_cols=['委托合约', '买卖', '开平', '手数', '成交均价', '报单时间', '报单日期']):
        status = os.path.isdir(report_file_path)
        if not status:
            raise ValueError(f'{report_file_path} is not folder')

        report_daily = os.path.join(report_file_path, daily_report)
        daily_holder = []

        buysell_cols_replace = {'买\u3000': '买', '\u3000卖': '卖'}

        for daily_file in glob(report_daily):
            report = pd.read_csv(daily_file, encoding='GBK')

            match = re.search(r'report(\d{8})\.csv', os.path.split(daily_file)[-1])
            dt = int(match.group(1))  # pd.to_datetime(match.group(1), format='%Y%m%d')
            report['报单日期'] = dt
            report['买卖'] = report['买卖'].replace(buysell_cols_replace)
            daily_holder.append(report)
        daily = pd.concat(daily_holder)

        report_period = os.path.join(report_file_path, period_report)
        report_period_all = pd.concat([pd.read_excel(period, sheet_name='report') for period in glob(report_period)])
        # report_period_all

        merged_report = pd.concat([report_period_all, daily])

        ## 全部成交
        all_settledown = merged_report[
            (merged_report['挂单状态'] == '全部成交') & (merged_report['未成交'] == 0)]
        ## 已撤单-部分成交
        partial_settledown = merged_report[
            (merged_report['挂单状态'].isin(('已撤单', '部分成交'))) & (
                        merged_report['手数'] != merged_report['未成交'])].copy(
            deep=True)
        # partial_settledown = merged_report[
        #     (merged_report['挂单状态'] == '已撤单') & (merged_report['手数'] != merged_report['未成交'])].copy(
        #     deep=True)

        partial_settledown['手数'] = partial_settledown['手数'] - partial_settledown['未成交']
        partial_settledown['未成交'] = 0
        partial_settledown['挂单状态'] = '全部成交'

        _parsed_report = pd.concat([all_settledown, partial_settledown])
        _parsed_report['开平'] = _parsed_report['开平'].replace({'开仓': '开', '平仓': '平'})
        _parsed_report['委托合约'] = _parsed_report['委托合约'].apply(lambda x: x.strip()).str.upper()

        return _parsed_report

    @staticmethod
    @file_cache(enable_cache=True, granularity='d')
    def get_info_last_delivery_multi(contracts):
        # contracts = self._transactions['委托合约'].unique().tolist() if contracts is None else contracts
        lastdeliv_and_multi = WindHelper.get_future_info_last_delivery_date(contracts)
        lastdeliv_and_multi.index.name = '委托合约'
        return lastdeliv_and_multi[['START_DATE', 'EXE_DATE', 'CONTRACTMULTIPLIER']]  #,'UNDERLYINGWINDCODE'

    @staticmethod
    def get_quote(lastdeliv_and_multi: pd.DataFrame, today=None):

        today = datetime.datetime.today() if today is None else today
        h = []
        for contract in lastdeliv_and_multi.index:
            params = lastdeliv_and_multi.loc[contract, :].to_dict()
            start = min(today, params['START_DATE']).strftime("%Y-%m-%d")
            end = min(today, params['EXE_DATE']).strftime("%Y-%m-%d")
            q = WindHelper.wind_wsd_quote_reduce(contract, start, end).dropna()
            q.index.name = 'date'
            for k, v in params.items():
                q[k] = v
            h.append(q)

        return pd.concat(h)

    @classmethod
    @file_cache(enable_cache=True, granularity='d')
    def get_quote_and_info(cls, contracts, today=datetime.datetime.today()):
        lastdeliv_and_multi = cls.get_info_last_delivery_multi(contracts)
        quote = cls.get_quote(lastdeliv_and_multi, today=today)
        quote = quote.reset_index().pivot_table(index='date', columns='symbol', values='CLOSE')

        quote.index = pd.to_datetime(quote.index)

        return quote

    @staticmethod
    def code_type_detect(code,
                         futures_pattern=re.compile(r'^[A-Z]{2}\d{4}\.\w{3}$'),
                         option_pattern=re.compile(r'^[A-Z]+[0-9]+-[CP]-[0-9]+\.\w+$'),
                         ):
        if futures_pattern.match(code):
            return 'Future'
        elif option_pattern.match(code):
            return 'Option'
        else:
            return 'Unknown'

    @staticmethod
    def deriviative_to_index(code,
                             ):
        if code.startswith("IM"):
            return "000852.SH"
        elif code.startswith("IC"):
            return "000905.SH"
        else:
            return "OTHERS"


    @staticmethod
    def _cal_buy2open_cost(transactions, option_list, dt_list):
        m1 = transactions['委托合约'].isin(option_list)
        mask_buy2open = transactions['买卖开平'] == '买开'

        m1_buy2open_transactions = transactions[m1 & mask_buy2open]
        buy2open = m1_buy2open_transactions.pivot_table(index='报单日期', values='cost',
                                                        columns='委托合约').reindex(index=dt_list)
        buy2open_unit = m1_buy2open_transactions.pivot_table(index='报单日期', values='unit',
                                                             columns='委托合约').reindex(index=dt_list)

        cum_cost_df = buy2open.reindex(index=quote.index, columns=option_list).fillna(0).cumsum()

        return cum_cost_df, buy2open_unit, buy2open

    @staticmethod
    def _cal_sell2close_cost(transactions, option_list, dt_list):
        m1 = transactions['委托合约'].isin(option_list)
        mask_sell2close = transactions['买卖开平'] == '卖平'

        m1_sell2close_transactions = transactions[m1 & mask_sell2close]
        sell2close = m1_sell2close_transactions.pivot_table(index='报单日期', values='cost',
                                                            columns='委托合约').reindex(index=dt_list)
        sell2close_unit = m1_sell2close_transactions.pivot_table(index='报单日期', values='unit',
                                                                 columns='委托合约').reindex(index=dt_list)

        cum_sold_value_df = sell2close.reindex(index=quote.index, columns=option_list).fillna(0).cumsum()

        return cum_sold_value_df, sell2close_unit, sell2close

    @staticmethod
    def _cal_sell2open_cost(transactions, future_list, dt_list):
        m1 = transactions['委托合约'].isin(future_list)
        mask_sell2open = transactions['买卖开平'] == '卖开'

        m1_sell2open_transactions = transactions[m1 & mask_sell2open]
        sell2open = m1_sell2open_transactions.pivot_table(index='报单日期', values='cost',
                                                            columns='委托合约').reindex(index=dt_list)
        sell2open_unit = m1_sell2open_transactions.pivot_table(index='报单日期', values='unit',
                                                                 columns='委托合约').reindex(index=dt_list)

        cum_sold_value_df = sell2open.reindex(index=quote.index, columns=future_list).fillna(0).cumsum()

        return cum_sold_value_df, sell2open_unit, sell2open

    @staticmethod
    def _cal_buy2close_cost(transactions, future_list, dt_list):
        m1 = transactions['委托合约'].isin(future_list)
        mask_buy2close = transactions['买卖开平'] == '买平'

        m1_buy2close_transactions = transactions[m1 & mask_buy2close]
        buy2close = m1_buy2close_transactions.pivot_table(index='报单日期', values='cost',
                                                        columns='委托合约').reindex(index=dt_list)
        buy2close_unit = m1_buy2close_transactions.pivot_table(index='报单日期', values='unit',
                                                             columns='委托合约').reindex(index=dt_list)

        cum_cost_df = buy2close.reindex(index=quote.index, columns=future_list).fillna(0).cumsum()

        return cum_cost_df, buy2close_unit, buy2close

    @staticmethod
    def prepare_transaction(transactions, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, }):
        transactions['买卖开平'] = transactions['买卖'] + transactions['开平']
        transactions['trade_type'] = transactions['买卖开平'].replace(trade_type_mark)
        transactions['unit'] = transactions['手数'] * transactions['CONTRACTMULTIPLIER']
        transactions['cost'] = transactions['unit'] * transactions['成交均价'] * transactions['trade_type']
        transactions['报单日期'] = pd.to_datetime(transactions['报单日期'], format='%Y%m%d')
        # transactions['UNDERLYINGPRICE'] = transactions['手数'] * transactions['CONTRACTMULTIPLIER']
        # ['委托合约', '买卖', '开平', '报单日期', '手数', '成交均价', 'START_DATE', 'EXE_DATE', 'CONTRACTMULTIPLIER']
        return transactions


    def calculate_executed_profit(self,future, net_unit, cum_cost_df, end_dt, holding_value):
        if holding_value.index.max() >= end_dt:
            # 获取标的资产的收盘价
            index = self.deriviative_to_index(future)
            spot_price = w.wsd(index, "close", end_dt, end_dt, "")

            # 计算交割收益
            spot_value = net_unit * spot_price.Data[0]
            value_of_executed_profit = spot_value[-1] - cum_cost_df[[future]].iloc[-1] # 假设cum_cost_df是单列DataFrame
            dt_list = holding_value.index[holding_value.index > end_dt]
            executed_profit = pd.DataFrame(index=dt_list)
            executed_profit[future] = value_of_executed_profit
        else:
            executed_profit = pd.DataFrame(index=cum_cost_df.index)

        return executed_profit



    def parse_transactions_with_quote_buy(self, quote, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, }):

        lastdeliv_and_multi = self.get_info_last_delivery_multi(self.reduced_contracts())
        transactions = self.create_items(lastdeliv_and_multi, reduced=True, return_df=True)
        transactions = self.prepare_transaction(transactions, trade_type_mark=trade_type_mark)

        option_list = list(filter(lambda x: self.code_type_detect(x) == 'Option', transactions['委托合约'].unique()))

        # avg_holding_cost = self.cal_average_cost(transactions,)

        # m1 = transactions['委托合约'].isin(option_list)
        ## 累计开仓成本,开仓张数，开仓金额

        cum_cost_df, buy2open_unit, buy2open = self._cal_buy2open_cost(transactions, option_list, quote.index)
        ## 累计平仓金额,平仓张数，平仓金额
        _, sell2close_unit, sell2close = self._cal_sell2close_cost(transactions, option_list,
                                                                   quote.index)

        cum_buy_unit = buy2open_unit.cumsum().ffill()  # 累计开仓张数
        # cumsum_sell2close_unit = sell2close_unit.cumsum()

        # 有卖出的期权
        have_sold_options = sell2close.columns.tolist()

        holder = []

        for option in option_list:
            end_dt = lastdeliv_and_multi.loc[option, 'EXE_DATE']

            if option in have_sold_options:  # 如果有平仓交易
                cum_sold_value = sell2close[sell2close.index <= end_dt][[option]].fillna(0).cumsum()
                cum_sold_unit = sell2close_unit[sell2close_unit.index <= end_dt][[option]].fillna(0).cumsum()
                net_unit = cum_buy_unit[option] - cum_sold_unit.fillna(0)[option]

                holding_value = net_unit * quote[option]
                if not isinstance(holding_value, pd.DataFrame):
                    holding_value = holding_value.to_frame(option)

                res_sub = cum_sold_value * -1 + holding_value - cum_cost_df[[option]]

            else:  # 无平仓交易
                cum_sold_value = pd.DataFrame(index=cum_cost_df.index)
                cum_sold_unit = pd.DataFrame()
                net_unit = cum_buy_unit[option]

                holding_value = net_unit * quote[option]
                if not isinstance(holding_value, pd.DataFrame):
                    holding_value = holding_value.to_frame(option)

                res_sub = holding_value - cum_cost_df[[option]]

            ## 需要行权收益
            if quote[quote.index > end_dt].empty:
                executed_profit = pd.DataFrame(index=cum_cost_df.index)
            else:
                value_of_executed_profit = max(0, holding_value.loc[end_dt].values[0])
                dt_list = quote.index[quote.index > end_dt]
                executed_profit = pd.DataFrame(index=dt_list, )
                executed_profit[option] = value_of_executed_profit

            holder.append([cum_sold_value, holding_value, net_unit, executed_profit, res_sub])

        value_holder, holding_value_holder, unit_holder, executed_holder, result_holder = list(zip(*holder))

        holding_df = pd.concat(holding_value_holder, axis=1).reindex(index=quote.index, columns=option_list)
        unit_df = pd.concat(unit_holder, axis=1).reindex(index=quote.index, columns=option_list)
        # cum_cost_df
        value_df = pd.concat(value_holder, axis=1).reindex(index=quote.index, columns=option_list).ffill().fillna(0)
        executed_df = pd.concat(executed_holder, axis=1).reindex(index=quote.index, columns=option_list).fillna(0)
        res = pd.concat(result_holder, axis=1).reindex(index=quote.index, columns=option_list).ffill()

        sum_realized_value = value_df.sum(axis=1).to_frame('累计平仓价值') * -1
        sum_cost = cum_cost_df.sum(axis=1).to_frame('累计开仓成本')
        cross_resid_value = holding_df.sum(axis=1).to_frame('期权残值')
        sum_executed = executed_df.sum(axis=1).to_frame('行权收益')
        sum_res = res.sum(axis=1).to_frame('累计净损益(右轴)')

        summary_info = pd.concat([sum_realized_value, sum_cost, cross_resid_value, sum_executed, sum_res], axis=1)

        summary_info['期权累计价值（残值+行权收益+平仓收益）'] = summary_info['累计平仓价值'] + summary_info['期权残值'] + summary_info['行权收益']

        summary_info['累计持仓收益率'] = summary_info['累计净损益(右轴)'] / summary_info['累计开仓成本']

        return summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df

    def parse_transactions_with_quote_sell(self, quote, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, }):

        lastdeliv_and_multi = self.get_info_last_delivery_multi(self.reduced_contracts())
        transactions = self.create_items(lastdeliv_and_multi, reduced=True, return_df=True)
        transactions = self.prepare_transaction(transactions, trade_type_mark=trade_type_mark)

        mask = (transactions['买卖开平'] == "卖开") | (transactions['买卖开平'] == "买平")
        # mask_short = ~mask_long
        # transactions_short = transactions[mask_short]

        option_list = list(filter(lambda x: self.code_type_detect(x) == 'Option', transactions[mask]['委托合约'].unique()))

        # avg_holding_cost = self.cal_average_cost(transactions,)

        # m1 = transactions['委托合约'].isin(option_list)
        ## 累计开仓成本,开仓张数，开仓金额

        cum_cost_df, sell2open_unit, sell2open = self._cal_sell2open_cost(transactions[mask], option_list, quote.index)
        ## 累计平仓金额,平仓张数，平仓金额
        _, buy2close_unit, buy2close = self._cal_buy2close_cost(transactions[mask], option_list,
                                                                   quote.index)

        cum_buy_unit = sell2open_unit.cumsum().ffill()  # 累计开仓张数
        # cumsum_sell2close_unit = sell2close_unit.cumsum()

        # 有卖出的期权
        have_sold_options = buy2close.columns.tolist()

        holder = []

        for option in option_list:
            end_dt = lastdeliv_and_multi.loc[option, 'EXE_DATE']

            if option in have_sold_options:  # 如果有平仓交易
                cum_sold_value = buy2close[buy2close.index <= end_dt][[option]].fillna(0).cumsum()
                cum_sold_unit = buy2close_unit[buy2close_unit.index <= end_dt][[option]].fillna(0).cumsum()
                net_unit = cum_buy_unit[option] - cum_sold_unit.fillna(0)[option]

                holding_value = net_unit * quote[option]
                if not isinstance(holding_value, pd.DataFrame):
                    holding_value = holding_value.to_frame(option)

                res_sub = (cum_sold_value * -1 + holding_value - cum_cost_df[[option]])*-1

            else:  # 无平仓交易
                cum_sold_value = pd.DataFrame(index=cum_cost_df.index)
                # cum_sold_unit = pd.DataFrame()
                net_unit = cum_buy_unit[option]

                holding_value = net_unit * quote[option]
                if not isinstance(holding_value, pd.DataFrame):
                    holding_value = holding_value.to_frame(option)

                res_sub = (holding_value - cum_cost_df[[option]])*-1

            ## 需要行权收益
            if quote[quote.index > end_dt].empty:
                executed_profit = pd.DataFrame(index=cum_cost_df.index)
            else:
                value_of_executed_profit = max(0, holding_value.loc[end_dt].values[0])
                dt_list = quote.index[quote.index > end_dt]
                executed_profit = pd.DataFrame(index=dt_list, )
                executed_profit[option] = value_of_executed_profit

            holder.append([cum_sold_value, holding_value, net_unit, executed_profit, res_sub])

        value_holder, holding_value_holder, unit_holder, executed_holder, result_holder = list(zip(*holder))

        holding_df = pd.concat(holding_value_holder, axis=1).reindex(index=quote.index, columns=option_list)
        unit_df = pd.concat(unit_holder, axis=1).reindex(index=quote.index, columns=option_list)
        # cum_cost_df
        value_df = pd.concat(value_holder, axis=1).reindex(index=quote.index, columns=option_list).ffill().fillna(0)
        executed_df = pd.concat(executed_holder, axis=1).reindex(index=quote.index, columns=option_list).fillna(0)
        res = pd.concat(result_holder, axis=1).reindex(index=quote.index, columns=option_list).ffill()

        sum_realized_value = value_df.sum(axis=1).to_frame('累计平仓价值') * -1
        sum_cost = cum_cost_df.sum(axis=1).to_frame('累计开仓成本')
        cross_resid_value = holding_df.sum(axis=1).to_frame('期权残值')
        sum_executed = executed_df.sum(axis=1).to_frame('行权收益')
        sum_res = res.sum(axis=1).to_frame('累计净损益(右轴)')

        summary_info = pd.concat([sum_realized_value, sum_cost, cross_resid_value, sum_executed, sum_res], axis=1)

        summary_info['期权累计价值（残值+行权收益+平仓收益）'] = summary_info['累计平仓价值'] + summary_info['期权残值'] + summary_info['行权收益']

        summary_info['累计持仓收益率'] = summary_info['累计净损益(右轴)'] / summary_info['累计开仓成本']

        return summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df

    # def get_futures_list(self, transactions, code_type_detect_func, list_type='long'):
    #     # 区分做多和做空
    #     mask_long = (transactions['买卖开平'] == "买开") | (transactions['买卖开平'] == "卖平")
    #
    #     if list_type == 'long':
    #         transactions_filtered = transactions[mask_long]
    #     else:
    #         transactions_filtered = transactions[~mask_long]
    #
    #     future_list = list(
    #         filter(lambda x: code_type_detect_func(x) == 'Future', transactions_filtered['委托合约'].unique()))
    #
    #     return future_list

    # def cal_output(self,):
    #     value_holder, holding_value_holder, unit_holder, result_holder = list(zip(*holder))
    #
    #     holding_df = pd.concat(holding_value_holder, axis=1).reindex(index=quote.index, columns=future_list)
    #     unit_df = pd.concat(unit_holder, axis=1).reindex(index=quote.index, columns=future_list)
    #     # cum_cost_df
    #     value_df = pd.concat(value_holder, axis=1).reindex(index=quote.index, columns=future_list).ffill().fillna(0)
    #     # executed_df = pd.concat(executed_holder, axis=1).reindex(index=quote.index, columns=future_list).fillna(0)
    #     res = pd.concat(result_holder, axis=1).reindex(index=quote.index, columns=future_list).ffill()
    #
    #     sum_realized_value = value_df.sum(axis=1).to_frame('累计平仓价值') * -1
    #     sum_cost = cum_cost_df.sum(axis=1).to_frame('累计开仓成本')
    #     cross_resid_value = holding_df.sum(axis=1).to_frame('期权残值')
    #     # sum_executed = executed_df.sum(axis=1).to_frame('行权收益')
    #     sum_res = res.sum(axis=1).to_frame('累计净损益(右轴)')
    #
    #     summary_info = pd.concat([sum_realized_value, sum_cost, cross_resid_value, sum_res], axis=1)
    #
    #     summary_info['期权累计价值（残值+平仓收益）'] = summary_info['累计平仓价值'] + summary_info['期权残值']
    #
    #     summary_info['累计持仓收益率'] = summary_info['累计净损益(右轴)'] / summary_info['累计开仓成本']
    def parse_transactions_with_quote_long(self, quote, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, }):

        lastdeliv_and_multi = self.get_info_last_delivery_multi(self.reduced_contracts())
        transactions = self.create_items(lastdeliv_and_multi, reduced=True, return_df=True)
        transactions = self.prepare_transaction(transactions, trade_type_mark=trade_type_mark)

        #future_long = self.get_futures_list(transactions, self.code_type_detect_func, list_type='long')

        # 区分做多还是做空
        mask_long = (transactions['买卖开平'] == "买开") | (transactions['买卖开平'] == "卖平")
        mask_short = ~mask_long
        transactions_long = transactions[mask_long]
        #transactions_short = transactions[mask_short]

        future_long = list(filter(lambda x: self.code_type_detect(x) == 'Future', transactions_long['委托合约'].unique()))
        #m1 = transactions['委托合约'].isin(option_list)


        ## 累计开仓成本,开仓张数，开仓金额
        cum_cost_df, buy2open_unit, buy2open = self._cal_buy2open_cost(transactions, future_long, quote.index)
        ## 累计平仓金额,平仓张数，平仓金额
        _, sell2close_unit, sell2close = self._cal_sell2close_cost(transactions, future_long, quote.index)
        cum_buy_unit = buy2open_unit.cumsum().ffill()  # 累计开仓张数
        # cumsum_sell2close_unit = sell2close_unit.cumsum()

        # 有平仓的交易
        have_sold_future = sell2close.columns.tolist()

        holder = []

        for future in future_long:
            end_dt = lastdeliv_and_multi.loc[future, 'EXE_DATE']
            last_netunits = {}

            if future in have_sold_future:  # 如果有平仓交易
                cum_sold_value = sell2close[sell2close.index <= end_dt][[future]].fillna(0).cumsum()
                cum_sold_unit = sell2close_unit[sell2close_unit.index <= end_dt][[future]].fillna(0).cumsum()
                net_unit = cum_buy_unit[future] - cum_sold_unit.fillna(0)[future]

                holding_value = net_unit * quote[future]
                if not isinstance(holding_value, pd.DataFrame):
                    holding_value = holding_value.to_frame(future)

                res_sub = cum_sold_value * -1 + holding_value - cum_cost_df[[future]]

            else:  # 无平仓交易
                cum_sold_value = pd.DataFrame(index=cum_cost_df.index)
                #cum_sold_unit = pd.DataFrame()
                net_unit = cum_buy_unit[future]

                holding_value = net_unit * quote[future]
                if not isinstance(holding_value, pd.DataFrame):
                    holding_value = holding_value.to_frame(future)

                res_sub = holding_value - cum_cost_df[[future]]
                last_netunits[future] = net_unit.iloc[-1]
            if future.startswith('IM' or 'IH' or 'IF'):
                if holding_value.index.max() >= end_dt:
                    # executed_profit = pd.DataFrame(index=cum_cost_df.index)
                    index = self.deriviative_to_index(future)
                    spot_price = w.wsd(index, "close", end_dt, end_dt, "")

                    spot_value = net_unit * spot_price.Data[0]
                    # value_of_executed_profit = spot_value[-1] #- cum_cost_df[future].iloc[-1]#- holding_value.values[0]  # future-->spot
                    dt_list = holding_value.index[holding_value.index > end_dt]
                    executed_profit = pd.DataFrame(index=dt_list, )
                    executed_profit[future] = spot_value[-1]  # value_of_executed_profit
                else:
                    executed_profit = pd.DataFrame(index=cum_cost_df.index)
            else:
                executed_profit = pd.DataFrame(index=cum_cost_df.index)

            last_netunits_df = pd.DataFrame(last_netunits, index=[cum_buy_unit.index.max()])
            holder.append([cum_sold_value, holding_value, net_unit, executed_profit, res_sub])

        value_holder, holding_value_holder, unit_holder, executed_holder, result_holder = list(zip(*holder))

        holding_df = pd.concat(holding_value_holder, axis=1).reindex(index=quote.index, columns=future_long)
        unit_df = pd.concat(unit_holder, axis=1).reindex(index=quote.index, columns=future_long)
        value_df = pd.concat(value_holder, axis=1).reindex(index=quote.index, columns=future_long).ffill().fillna(
            0)
        executed_df = pd.concat(executed_holder, axis=1).reindex(index=quote.index, columns=future_long).fillna(0)
        res = pd.concat(result_holder, axis=1).reindex(index=quote.index, columns=future_long).ffill()

        sum_realized_value = value_df.sum(axis=1).to_frame('累计平仓价值') * -1
        sum_cost = cum_cost_df.sum(axis=1).to_frame('累计开仓成本')
        cross_resid_value = holding_df.sum(axis=1).to_frame('期货残值')
        sum_executed = executed_df.sum(axis=1).to_frame('交割收益')
        sum_res = res.sum(axis=1).to_frame('累计净损益(右轴)')

        summary_info = pd.concat([sum_realized_value, sum_cost, cross_resid_value, sum_executed, sum_res], axis=1)

        summary_info['期货累计价值（平仓收益+期货残值+交割收益）'] = summary_info['累计平仓价值'] + summary_info[
            '期货残值'] + \
                                                                   summary_info['交割收益']

        summary_info['累计持仓收益率'] = summary_info['累计净损益(右轴)'] / summary_info['累计开仓成本']
        summary_info['收益净值'] = 1 + summary_info['累计持仓收益率']

        return summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df



    #
    # def cal_deliv_profit_short(self,future,end_dt,buy2close,buy2close_unit,cum_sell_unit):
    #     cum_sold_value = buy2close[buy2close.index <= end_dt][[future]].fillna(0).cumsum()
    #     cum_sold_unit = buy2close_unit[buy2close_unit.index <= end_dt][[future]].fillna(0).cumsum()
    #     net_unit = cum_sell_unit[future] - cum_sold_unit.fillna(0)[future]
    #     holding_value = net_unit * quote[future]
    #     return holding_value
    # def cal_closeout_profit_short(self,future,end_dt,buy2close,buy2close_unit,cum_sell_unit):
    #     cum_sold_value = buy2close[buy2close.index <= end_dt][[future]].fillna(0).cumsum()
    #     cum_sold_unit = buy2close_unit[buy2close_unit.index <= end_dt][[future]].fillna(0).cumsum()
    #     net_unit = cum_sell_unit[future] - cum_sold_unit.fillna(0)[future]
    #     holding_value = net_unit * quote[future]
    #     return holding_value
    def parse_transactions_with_quote_short(self, quote, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, }):
        lastdeliv_and_multi = self.get_info_last_delivery_multi(self.reduced_contracts())
        transactions = self.create_items(lastdeliv_and_multi, reduced=True, return_df=True)
        transactions = self.prepare_transaction(transactions, trade_type_mark=trade_type_mark)
        # future_short = self.get_futures_list(transactions, self.code_type_detect_func)

        # 区分做多还是做空
        mask_long = (transactions['买卖开平'] == "买开") | (transactions['买卖开平'] == "卖平")
        mask_short = ~mask_long
        #transactions_long = transactions[mask_long]
        transactions_short = transactions[mask_short]


        future_short = list(
            filter(lambda x: self.code_type_detect(x) == 'Future', transactions_short['委托合约'].unique()))


        cum_cost_df, sell2open_unit, sell2open = self._cal_sell2open_cost(transactions, future_short, quote.index)
        result_holder, buy2close_unit, buy2close = self._cal_buy2close_cost(transactions, future_short, quote.index)
        cum_sell_unit = sell2open_unit.cumsum().ffill()  # 累计开仓张数
        have_sold_future = buy2close.columns.tolist()

        holder = []

        for future in future_short:

            end_dt = lastdeliv_and_multi.loc[future, 'EXE_DATE']
            #post_expiry_data = quote[quote.index > end_dt]
            #delivery_occurred = not post_expiry_data.empty

            if future in have_sold_future:  # 如果有平仓交易
                cum_sold_value = buy2close[buy2close.index <= end_dt][[future]].fillna(0).cumsum()
                cum_sold_unit = buy2close_unit[buy2close_unit.index <= end_dt][[future]].fillna(0).cumsum()
                net_unit = cum_sell_unit[future] - cum_sold_unit.fillna(0)[future]

                holding_value = net_unit * quote[future]
                if not isinstance(holding_value, pd.DataFrame):
                    holding_value = holding_value.to_frame(future)

                res_sub = cum_sold_value - holding_value + cum_cost_df[[future]]


            else:  # 无平仓交易
                cum_sold_value = pd.DataFrame(index=cum_cost_df.index)
                # cum_sold_unit = pd.DataFrame()
                net_unit = cum_sell_unit[future]

                holding_value = net_unit * quote[future]
                if not isinstance(holding_value, pd.DataFrame):
                    holding_value = holding_value.to_frame(future)
                res_sub = cum_cost_df[[future]] - holding_value
            # #如果交割了    交割价格是现货价

            #executed_profit = self.calculate_executed_profit(future, net_unit, cum_cost_df, end_dt, holding_value)
            if future.startswith('IM' or 'IH' or 'IF'):
                if holding_value.index.max() >= end_dt:
                    # executed_profit = pd.DataFrame(index=cum_cost_df.index)
                    index = self.deriviative_to_index(future)
                    spot_price = w.wsd(index, "close", end_dt, end_dt, "")

                    spot_value = net_unit * spot_price.Data[0]
                    #value_of_executed_profit = spot_value[-1] #- cum_cost_df[future].iloc[-1]#- holding_value.values[0]  # future-->spot
                    dt_list = holding_value.index[holding_value.index > end_dt]
                    executed_profit = pd.DataFrame(index=dt_list, )
                    executed_profit[future] = spot_value[-1]#value_of_executed_profit
                else:
                    executed_profit = pd.DataFrame(index=cum_cost_df.index)
            else:
                executed_profit = pd.DataFrame(index=cum_cost_df.index)

                #
            holder.append([cum_sold_value, holding_value, net_unit, executed_profit, res_sub])

        value_holder, holding_value_holder, unit_holder, executed_holder, result_holder = list(zip(*holder))

        holding_df = pd.concat(holding_value_holder, axis=1).reindex(index=quote.index, columns=future_short)
        unit_df = pd.concat(unit_holder, axis=1).reindex(index=quote.index, columns=future_short)
        value_df = pd.concat(value_holder, axis=1).reindex(index=quote.index, columns=future_short).ffill().fillna(0)
        executed_df = pd.concat(executed_holder, axis=1).reindex(index=quote.index, columns=future_short).fillna(0)
        res = pd.concat(result_holder, axis=1).reindex(index=quote.index, columns=future_short).ffill()

        sum_realized_value = value_df.sum(axis=1).to_frame('累计平仓价值') * -1
        sum_cost = cum_cost_df.sum(axis=1).to_frame('累计开仓成本')
        cross_resid_value = holding_df.sum(axis=1).to_frame('期货残值')
        sum_executed = executed_df.sum(axis=1).to_frame('交割收益')
        sum_res = res.sum(axis=1).to_frame('累计净损益(右轴)')

        summary_info = pd.concat([sum_realized_value, sum_cost, cross_resid_value,sum_executed, sum_res], axis=1)

        summary_info['期货累计价值（平仓收益+期货残值+交割收益）'] = summary_info['累计平仓价值'] + summary_info['期货残值'] + summary_info['交割收益']

        summary_info['累计持仓收益率'] = summary_info['累计净损益(右轴)'] / summary_info['累计开仓成本']
        summary_info['收益净值'] = 1 + summary_info['累计持仓收益率']
        return summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df



    def cal_category_contract(self, res, cum_cost_df):
        nv_df = pd.DataFrame(index=res.index)
        returns = pd.DataFrame(index=res.index)
        profit = pd.DataFrame(index=res.index)

        periods = {'当日': 1, '近一周': 5, '近一月': 20}
        contract_prefixes = {}
        for col in res.columns:
            match = re.match(r'^[A-Za-z]+', col)
            if match:
                prefix = match.group(0)
                if prefix not in contract_prefixes:
                    contract_prefixes[prefix] = []
                contract_prefixes[prefix].append(col)

        for prefix, contract_list in contract_prefixes.items():
            sum_res = res.loc[:, contract_list].sum(axis=1, min_count=1)  # 至少有一个非NA值才进行求和
            sum_cum_cost = cum_cost_df.loc[:, contract_list].sum(axis=1, min_count=1)
            sum_cum_cost.replace(0, pd.NA, inplace=True)
            nv = 1 + sum_res / sum_cum_cost
            nv_df[prefix + '损益'] = sum_res.fillna(method='ffill', inplace=False).fillna(0)
            nv_df[prefix + '净值'] = nv.fillna(method='ffill', inplace=False).fillna(1)

            profit.at[prefix,'累计损益']=nv_df[prefix + '损益'].iloc[-1]
            returns.at[prefix,'累计收益率']=nv_df[prefix + '净值'].iloc[-1]-1
            for period, offset in periods.items():
                profit.at[prefix, f'{period}收益'] = nv_df[prefix + '损益'].diff(periods=offset).iloc[-1]
                returns.at[prefix, f'{period}收益率'] = nv_df[prefix + '净值'].pct_change(periods=offset).iloc[-1]
            profit = profit.dropna()
            returns = returns.dropna()

        return nv_df,profit,returns

    def cal_return(self,data):
        data['当日收益率'] = data['IM净值'].pct_change()

        # Calculate weekly return for the latest date
        data['IM近一周收益率'] = data['IM净值'].pct_change(periods=5)

        # Calculate monthly return for the latest date
        data['IM近一月收益率'] = data['IM净值'].pct_change(periods=20)

        # Select the last row for the most recent date's data
        latest_data = data.iloc[-1][['当日收益率', 'IM近一周收益率', 'IM近一月收益率']]


    def get_person_output(self,res,executed_df,value_df,cum_cost_df,holding_df,unit_df):
        today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')
        person_contract_mapping =self.map_contracts_to_persons()

        for person, contracts in person_contract_mapping.items():  #
            _contracts = [contract for contract in contracts if contract in res.columns]
            if not _contracts:
                continue
            person_res = res[_contracts]
            person_executed_df = executed_df[_contracts]
            person_value_df = value_df[_contracts]
            person_cum_cost_df = cum_cost_df[_contracts]
            person_holding_df = holding_df[_contracts]
            person_unit_df = unit_df[_contracts]
            sum_realized_value = person_value_df.sum(axis=1).to_frame('累计平仓价值') * -1
            sum_cost = person_cum_cost_df.sum(axis=1).to_frame('累计开仓成本')
            cross_resid_value = person_holding_df.sum(axis=1).to_frame('期货残值')
            sum_executed = person_executed_df.sum(axis=1).to_frame('交割收益')
            sum_res = person_res.sum(axis=1).to_frame('累计净损益(右轴)')
            nv_df,profits,returns= self.cal_category_contract(person_res,person_cum_cost_df)
            summary =pd.concat([profits,returns],axis=1)

            person_summary = pd.concat([sum_realized_value, sum_cost, cross_resid_value, sum_executed, sum_res], axis=1)
            # person_summary = summary_info[summary_info['委托合约'].isin(future_contracts)]
            person_summary['期货累计价值（平仓收益+期货残值+交割收益）'] = person_summary['累计平仓价值'] + person_summary['期货残值'] + person_summary['交割收益']
            person_summary['累计持仓收益率'] = person_summary['累计净损益(右轴)'] / person_summary['累计开仓成本']
            person_summary['收益净值'] = person_summary['累计持仓收益率']+1
            person_summary = pd.concat([person_summary,nv_df],axis=1)


            file_name = f'E:\\prt\\pf_analysis\\pf_analysis\\optionanalysis\\output_file\\期货日收益率统计及汇总@{today_str}_{person}.xlsx'
            with pd.ExcelWriter(file_name) as f:
                summary.to_excel(f, sheet_name='分品种统计')
                person_summary.to_excel(f, sheet_name='输出')
                person_res.to_excel(f, sheet_name='分合约累计交易损益')
                person_executed_df.to_excel(f, sheet_name='交割收益')
                person_value_df.to_excel(f, sheet_name='平仓价值累计值')
                person_cum_cost_df.to_excel(f, sheet_name='开仓成本累计值')
                person_holding_df.to_excel(f, sheet_name='持仓价值截面')
                person_unit_df.to_excel(f, sheet_name='持仓合约数')


        # return person_summary,person_res,person_executed_df,person_value_df,


    def parse_transactions_with_quote(self, quote, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, }):

        lastdeliv_and_multi = self.get_info_last_delivery_multi(self.reduced_contracts())
        transactions = self.create_items(lastdeliv_and_multi, reduced=True, return_df=True)

        transactions['买卖开平'] = transactions['买卖'] + transactions['开平']
        transactions['trade_type'] = transactions['买卖开平'].replace(trade_type_mark)
        transactions['unit'] = transactions['手数'] * transactions['CONTRACTMULTIPLIER']
        transactions['cost'] = transactions['unit'] * transactions['成交均价'] * transactions['trade_type']
        transactions['报单日期'] = pd.to_datetime(transactions['报单日期'], format='%Y%m%d')
        # ['委托合约', '买卖', '开平', '报单日期', '手数', '成交均价', 'START_DATE', 'EXE_DATE', 'CONTRACTMULTIPLIER']

        option_list = list(filter(lambda x: self.code_type_detect(x) == 'Option', transactions['委托合约'].unique()))

        m1 = transactions['委托合约'].isin(option_list)

        mask_buy2open = transactions['买卖开平'] == '买开'
        mask_sell2close = transactions['买卖开平'] == '卖平'

        buy2open = transactions[m1 & mask_buy2open].pivot_table(index='报单日期', values='cost',
                                                                columns='委托合约').reindex(index=quote.index)
        sell2close = transactions[m1 & mask_sell2close].pivot_table(index='报单日期', values='cost',
                                                                    columns='委托合约').reindex(index=quote.index)

        buy2open_unit = transactions[m1 & mask_buy2open].pivot_table(index='报单日期', values='unit',
                                                                     columns='委托合约').reindex(
            index=quote.index)
        sell2close_unit = transactions[m1 & mask_sell2close].pivot_table(index='报单日期', values='unit',
                                                                         columns='委托合约').reindex(
            index=quote.index)

        cumsum_buy2open_unit = buy2open_unit.cumsum()
        cum_buy_unit = cumsum_buy2open_unit.ffill()
        cumsum_sell2close_unit = sell2close_unit.cumsum()

        holding_value_holder = []
        unit_holder = []
        executed_holder = []
        value_holder = []
        result_holder = []
        # 有卖出的期权
        sell2close_columns = sell2close.columns.tolist()

        cum_cost_df = buy2open.reindex(index=quote.index, columns=option_list).fillna(0).cumsum()

        for option in option_list:
            end_dt = lastdeliv_and_multi.loc[option, 'EXE_DATE']

            if option in sell2close_columns:
                cum_sellvalue = sell2close[sell2close.index <= end_dt][[option]].fillna(0).cumsum()
                cum_sellvalue_unit = sell2close_unit[sell2close_unit.index <= end_dt][[option]].fillna(0).cumsum()
                net_unit = cum_buy_unit[option] - cum_sellvalue_unit.fillna(0)[option]

            else:
                cum_sellvalue = pd.DataFrame(index=cum_cost_df.index)
                cum_sellvalue_unit = pd.DataFrame()
                net_unit = cum_buy_unit[option]

            holding_value = net_unit * quote[option]
            if not isinstance(holding_value, pd.DataFrame):
                holding_value = holding_value.to_frame(option)

            if cum_sellvalue.empty:
                res_sub = holding_value - cum_cost_df[[option]]
            else:
                res_sub = cum_sellvalue * -1 + holding_value - cum_cost_df[[option]]

            ## 需要行权收益
            if quote[quote.index > end_dt].empty:
                executed_profit = pd.DataFrame(index=cum_cost_df.index)
            else:

                value_of_executed_profit = max(0, holding_value.loc[end_dt].values[0])
                dt_list = quote.index[quote.index > end_dt]
                executed_profit = pd.DataFrame(index=dt_list, )
                executed_profit[option] = value_of_executed_profit
                # executed_profit = holding_value.loc[end_dt].to_frame().T.reindex(index=quote.index).ffill()

            executed_holder.append(executed_profit)
            result_holder.append(res_sub)
            holding_value_holder.append(holding_value)
            unit_holder.append(net_unit)
            # cost_holder.append(cum_cost)
            value_holder.append(cum_sellvalue)

        holding_df = pd.concat(holding_value_holder, axis=1).reindex(index=quote.index, columns=option_list)
        unit_df = pd.concat(unit_holder, axis=1).reindex(index=quote.index, columns=option_list)
        # cum_cost_df
        value_df = pd.concat(value_holder, axis=1).reindex(index=quote.index, columns=option_list).ffill().fillna(0)
        executed_df = pd.concat(executed_holder, axis=1).reindex(index=quote.index, columns=option_list).fillna(0)
        res = pd.concat(result_holder, axis=1).reindex(index=quote.index, columns=option_list).ffill()
        sum_realized_value = value_df.sum(axis=1).to_frame('累计平仓价值') * -1
        sum_cost = cum_cost_df.sum(axis=1).to_frame('累计开仓成本')
        cross_resid_value = holding_df.sum(axis=1).to_frame('期权残值')
        sum_executed = executed_df.sum(axis=1).to_frame('行权收益')
        sum_res = res.sum(axis=1).to_frame('累计净损益(右轴)')

        summary_info = pd.concat([sum_realized_value, sum_cost, cross_resid_value, sum_executed, sum_res], axis=1)

        summary_info['期权累计价值（残值+行权收益+平仓收益）'] = summary_info['累计平仓价值'] + summary_info['期权残值'] + \
                                                               summary_info['行权收益']

        summary_info['累计持仓收益率'] = summary_info['累计净损益(右轴)'] / summary_info['累计开仓成本']

        return summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df

    #@staticmethod


    def get_individual_returns(self, holding_df, value_df, res, cum_cost_df, unit_df, executed_df=None):
        # latest_date = quote.index.max()
        # latest_quotes = quote.loc[latest_date]
        individual_returns = []
        person_contract_mapping = self.map_contracts_to_persons()
        # lastdeliv_and_multi = self.get_info_last_delivery_multi(self.reduced_contracts())

        for person, contracts in person_contract_mapping.items():  #
            for contract in contracts:
                # 针对每个人和他们的合约，提取相关数据
                if contract in holding_df.columns:
                    # 提取相关信息
                    latest_holding_value = holding_df[contract].iloc[-1]
                    latest_unit = unit_df[contract].iloc[-1] #/ lastdeliv_and_multi[contract]['CONTRACTMULTIPLIER']
                    latest_cum_cost = cum_cost_df[contract].iloc[-1]
                    latest_realized_value =value_df[contract].iloc[-1]
                    latest_res = res[contract].iloc[-1]
                    if len(res[contract]) > 1:
                        td_res = res[contract].iloc[-1] - res[contract].iloc[-2]
                    else:
                        td_res = 0

                    if latest_cum_cost != 0:
                        td_pctchg = td_res / latest_cum_cost
                    else:
                        td_pctchg = 0
                    # td_res = res[contract].iloc[-1] - res[contract].iloc[-2]
                    # td_pctchg = td_res/latest_cum_cost

                    if executed_df is not None and contract in executed_df.columns:
                        latest_executed_value = executed_df[contract].iloc[-1]
                        latest_res = res[contract].iloc[-1] + latest_executed_value
                        latest_return_rate = latest_res / latest_cum_cost
                    else:
                        latest_executed_value = None
                        latest_res = res[contract].iloc[-1]
                        latest_return_rate = latest_res / latest_cum_cost

                    # weekly_profit_amount, weekly_profit_rate, monthly_profit_amount, monthly_profit_rate=self.get_periodic_profit(summary_info)

                    individual_return = {
                        '姓名': person,
                        '合约': contract,
                        '手数': latest_unit,
                        '累计开仓成本': latest_cum_cost,
                        '累计平仓价值': latest_realized_value,
                        '持仓价值截面': latest_holding_value,
                        '当日损益': td_res,
                        '当日涨跌': td_pctchg,
                        '分合约累计损益': latest_res,
                        '分合约收益率': latest_return_rate,

                        # '周度收益' : weekly_profit_amount,
                        # '周度收益率': weekly_profit_rate,
                        # '月度收益' : monthly_profit_amount,
                        # '月度收益率': monthly_profit_rate,
                    }
                    if latest_executed_value is not None:
                        individual_return['交割/行权收益'] = latest_executed_value

                    individual_returns.append(individual_return)

        return pd.DataFrame(individual_returns)





if __name__ == '__main__':
    today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')

    PR = ProcessReport(
        report_file_path='E:\\prt\\pf_analysis\\pf_analysis\\optionanalysis\\report_file')

    contracts = PR.reduced_contracts()

    quote = PR.get_quote_and_info(contracts)
    quote = quote[quote.index >= '2023-1-1']  # reduce useless quote data



    #期权
    summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df = PR.parse_transactions_with_quote_buy(quote,trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1,})

    individual_returns = PR.get_individual_returns( holding_df, value_df, res, cum_cost_df, unit_df, executed_df)
    person_summary = PR.get_person_output(res,executed_df,value_df,cum_cost_df,holding_df,unit_df)
    with pd.ExcelWriter(f'期权日收益率统计及汇总@{today_str}.xlsx') as f:
        individual_returns.to_excel(f, sheet_name='分合约统计')
        summary_info.to_excel(f, sheet_name='输出')
        res.to_excel(f, sheet_name='分合约累计交易损益')
        executed_df.to_excel(f,sheet_name='行权收益')
        value_df.to_excel(f, sheet_name='平仓价值累计值')
        cum_cost_df.to_excel(f, sheet_name='开仓成本累计值')
        holding_df.to_excel(f, sheet_name='持仓价值截面')
        unit_df.to_excel(f, sheet_name='持仓合约数')

    summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df = PR.parse_transactions_with_quote_sell(
        quote, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, })

    individual_returns = PR.get_individual_returns(holding_df, value_df, res, cum_cost_df, unit_df, executed_df)
    person_summary = PR.get_person_output(res, executed_df, value_df, cum_cost_df, holding_df, unit_df)
    with pd.ExcelWriter(f'期权日收益率统计及汇总@{today_str}_sellput.xlsx') as f:
        individual_returns.to_excel(f, sheet_name='分合约统计')
        summary_info.to_excel(f, sheet_name='输出')
        res.to_excel(f, sheet_name='分合约累计交易损益')
        executed_df.to_excel(f, sheet_name='行权收益')
        value_df.to_excel(f, sheet_name='平仓价值累计值')
        cum_cost_df.to_excel(f, sheet_name='开仓成本累计值')
        holding_df.to_excel(f, sheet_name='持仓价值截面')
        unit_df.to_excel(f, sheet_name='持仓合约数')

    # 期货
    summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df = PR.parse_transactions_with_quote_long(quote,trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1,})
    individual_returns = PR.get_individual_returns(  holding_df, value_df, res, cum_cost_df, unit_df, executed_df)
    person_summary = PR.get_person_output(res,executed_df,value_df,cum_cost_df,holding_df,unit_df)
    with pd.ExcelWriter(f'E:\\prt\\pf_analysis\\pf_analysis\\optionanalysis\\output_file\\期货日收益率统计及汇总@{today_str}_long.xlsx') as f:

        individual_returns.to_excel(f, sheet_name='分合约统计')
        summary_info.to_excel(f, sheet_name='输出')
        res.to_excel(f, sheet_name='分合约累计交易损益')
        executed_df.to_excel(f,sheet_name='交割收益')
        value_df.to_excel(f, sheet_name='平仓价值累计值')
        cum_cost_df.to_excel(f, sheet_name='开仓成本累计值')
        holding_df.to_excel(f, sheet_name='持仓价值截面')
        unit_df.to_excel(f, sheet_name='持仓合约数')

    # 期货short
    summary_info, res, value_df, cum_cost_df, holding_df, executed_df, unit_df = PR.parse_transactions_with_quote_short(
        quote, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, })
    individual_returns = PR.get_individual_returns( holding_df, value_df, res, cum_cost_df, unit_df, executed_df)
    person_summary = PR.get_person_output(res,executed_df,value_df,cum_cost_df,holding_df,unit_df)
    with pd.ExcelWriter(
            f'E:\\prt\\pf_analysis\\pf_analysis\\optionanalysis\\output_file\\期货日收益率统计及汇总@{today_str}_short.xlsx') as f:
        individual_returns.to_excel(f, sheet_name='分合约统计')
        summary_info.to_excel(f, sheet_name='输出')
        res.to_excel(f, sheet_name='分合约累计交易损益')
        executed_df.to_excel(f, sheet_name='交割收益')
        value_df.to_excel(f, sheet_name='平仓价值累计值')
        cum_cost_df.to_excel(f, sheet_name='开仓成本累计值')
        holding_df.to_excel(f, sheet_name='持仓价值截面')
        unit_df.to_excel(f, sheet_name='持仓合约数')




    print(1)
    pass
