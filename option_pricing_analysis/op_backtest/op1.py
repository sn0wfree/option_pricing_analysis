# coding=utf-8
import datetime
import warnings
from functools import partial

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from collections import ChainMap

from CodersWheel.QuickTool.file_cache import file_cache

from Indicators import Sensibility

sensi_f = Sensibility()

from scipy.optimize import minimize, LinearConstraint

from option_analysis_monitor import ProcessReport, DerivativesItem, ReportAnalyst, ProcessReportSingle
import matplotlib.pyplot as plt


def cal_greek_with_given_f(row, f, DividendYield=0):
    UnderlyingPrice = f
    Strike = row['k']
    Volatility = row['bs_iv']
    Time2Maturity = row['t']
    RiskFreeRate = row['r']
    OptionType = row['cp']
    return sensi_f.getGreeks(UnderlyingPrice,
                             Strike,
                             Volatility,
                             Time2Maturity,
                             DividendYield,
                             RiskFreeRate,
                             OptionType, )


def cal_pnl(f, selected_with_weight):
    # call

    f_k = f - selected_with_weight['k']

    call_mask = selected_with_weight['cp'] == 'C'

    call_pnl = np.dot((f_k[call_mask] >= 0) * f_k[call_mask], selected_with_weight[call_mask]['weight'])

    # put
    put_mask = selected_with_weight['cp'] == 'P'
    put_pnl = np.dot((f_k[put_mask] <= 0) * f_k[put_mask] * -1, selected_with_weight[put_mask]['weight'])

    return call_pnl + put_pnl


def opposite(input, opposite_tuple):
    if input in opposite_tuple:
        if input == opposite_tuple[0]:
            return opposite_tuple[-1]
        elif input == opposite_tuple[-1]:
            return opposite_tuple[0]
        else:
            raise ValueError(f'opposite_tuple and input got wrong, {input} in opposite_tuple but not at last ot first')
    else:
        return input


class OptionPortfolioWithDT(object):
    __slots__ = ('_selected', '_multiply_num', '_weight',)

    def __init__(self, selected, multiply_num=100):
        if len(selected['dt'].unique().tolist()) != 1:
            raise ValueError('only accept cross data! dt must be same!')
        self._selected = selected
        self._multiply_num = multiply_num

        self._weight = []

    @property
    def _fake_f(self):
        return CachedData.create_fake_greek_for_contract(self._selected)

    def __len__(self):
        return len(self.available_contract)

    def get(self, contract_code):
        mask = self._selected['contract_code'] == contract_code
        return self._selected[mask]

    @property
    def call(self):
        call_mask = self._selected['cp'] == 'C'
        return self._selected[call_mask]

    @property
    def put(self):
        put_mask = self._selected['cp'] == 'P'
        return self._selected[put_mask]

    def level_put(self, level=1):
        put = self.put.sort_values('f_k_diff')
        cp = 'P'
        otm_mask = put['f_k_diff'] > 0 if cp == 'P' else put['f_k_diff'] < 0
        tt = put[otm_mask]
        return tt.head(level)['contract_code'].values[-1]

    def level_call(self, level=1):
        cp = 'C'
        call = self.call.sort_values('f_k_diff', ascending=False)

        otm_mask = call['f_k_diff'] > 0 if cp == 'P' else call['f_k_diff'] < 0
        tt = call[otm_mask]
        return tt.head(level)['contract_code'].values[-1]

    def get_level_put(self, level=1):
        code = self.level_put(level=level)
        return self.get(code)

    def get_level_call(self, level=1):
        code = self.level_call(level=level)
        return self.get(code)

    @property
    def f(self):
        return self._selected['f'].unique().tolist()[0]

    @property
    def available_contract(self):
        return self._selected['contract_code'].unique().tolist()

    @property
    def available_call_contract(self):

        return self.call['contract_code'].unique().tolist()

    @property
    def available_put_contract(self):

        return self.put['contract_code'].unique().tolist()

    @property
    def available_main_put_contract(self):
        put_mask = self._selected['cp'] == 'P'
        main_mask = self._selected['main_mark'] != 0
        #         avail_put = self.available_put_contract
        return self.reduce_quote(self._selected[put_mask & main_mask])

    @property
    def available_main_call_contract(self):
        call_mask = self._selected['cp'] == 'C'
        main_mask = self._selected['main_mark'] != 0

        return self.reduce_quote(self._selected[call_mask & main_mask])

    @staticmethod
    def reduce_quote(quote,
                     q_cols=['contract_code', 'f', 'k', 'fee', 'bs_iv', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', ]):

        return quote[q_cols].set_index('contract_code')

    def _cal_delta_size(self, share, contract_code, f):
        code_mask = self._selected['contract_code'] == contract_code
        Delta = self._selected[code_mask]['Delta']
        return share * self._multiply_num * 100 * Delta * f

    def _cal_gamma_size(self, share, contract_code):
        code_mask = self._selected['contract_code'] == contract_code
        Gamma = self._selected[code_mask]['Gamma']
        return share * self._multiply_num * 100 * Gamma

    def _cal_vega_size(self, share, contract_code):
        code_mask = self._selected['contract_code'] == contract_code
        Vega = self._selected[code_mask]['Vega']
        return share * self._multiply_num * 100 * Vega

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, w):
        self._weight = w

    def portfolio_delta(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected['Delta']) * self._multiply_num * self.f

    def portfolio_gamma(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected['Gamma']) * self._multiply_num

    def portfolio_vega(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected['Vega']) * self._multiply_num

    def cal_greek_with_given_f(self, f, DividendYield=0):
        for _, row in self._selected.iterrows():
            yield cal_greek_with_given_f(row, f, DividendYield=DividendYield)

    def _create_portfolio(self, hedge_size, p1=0.5):

        p1q = self.get_level_put(level=1)
        p1s = required_share(hedge_size * p1, p1q['Delta'], p1q['f']).astype(int)
        yield p1q['contract_code'].values.tolist()[0], p1s.values.tolist()[0]

        p2q = self.get_level_put(level=2)
        p2s = required_share(hedge_size * p1 * 0.1 * -1, p2q['Delta'], p2q['f']).astype(int)
        yield p2q['contract_code'].values.tolist()[0], p2s.values.tolist()[0]

        c1q = self.get_level_call(level=1)
        c1s = required_share(hedge_size * p1 * 0.6 * 1, c1q['Delta'], c1q['f']).astype(int)
        yield c1q['contract_code'].values.tolist()[0], c1s.values.tolist()[0]

        c2q = self.get_level_call(level=2)
        c2s = required_share(hedge_size * p1 * 0.6 * 1, c2q['Delta'], c2q['f']).astype(int)
        yield c2q['contract_code'].values.tolist()[0], c2s.values.tolist()[0]

        c3q = self.get_level_call(level=3)
        c3s = required_share(hedge_size * p1 * 0.5 * 0.6 * -1, c3q['Delta'], c3q['f']).astype(int)
        yield c3q['contract_code'].values.tolist()[0], c3s.values.tolist()[0]

        c4q = self.get_level_call(level=4)
        c4s = required_share(hedge_size * p1 * 0.5 * 0.6 * -1, c3q['Delta'], c3q['f']).astype(int)
        yield c4q['contract_code'].values.tolist()[0], c4s.values.tolist()[0]

    def create_init_weight(self, hedge_size, p1=0.5):

        otm_selected = self._selected

        weight_df = pd.DataFrame(list(self._create_portfolio(hedge_size, p1=p1)), columns=['contract_code', 'weight'])
        for code in weight_df['contract_code']:
            w = weight_df[weight_df['contract_code'] == code]['weight'].values[0]
            otm_selected.loc[code, 'init_weight'] = w
        otm_selected['init_weight'] = otm_selected['init_weight'].fillna(0)

        # otm_selected['init_weight'] = self.create_init_weight(hedge_size)

        return otm_selected['init_weight']

    def create_greek_matrix_all(self):
        otm_selected = self._selected.copy(deep=True)
        Delta, Gamma, Vega, Theta, Rho = CachedData.create_greek_matrix_all(otm_selected, self._fake_f)

        return Delta, Gamma, Vega, Theta, Rho

    def run_opt(self, hedge_size, Delta, Gamma, Vega, Theta, Rho, verbose=True):
        # Delta, Gamma, Vega, Theta, Rho = self.create_greek_matrix_all()
        otm_selected = self._selected

        initial_weights = self.create_init_weight(hedge_size)

        res = OptBundle.run_opt(initial_weights, otm_selected, hedge_size, Delta, Gamma, Vega, Rho, method='SLSQP',
                                verbose=verbose)

        otm_selected['weight'] = res.x

        return res


class PortfolioHolder(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def long_side_reduce(df, direct_dict={'卖平': -1, '买开': 1}):

        c = df['买卖'] + df['开平']
        c_replaced = c.replace(direct_dict)
        mask = c.isin(direct_dict.keys())

        sub = df[mask]

        if sub.empty:
            return pd.DataFrame()
        elif sub.shape[0] == 1:
            return sub
        else:
            sub_df = df.head(1).copy(deep=True)
            net_share = (df['手数'] * c_replaced).sum(axis=0)
            if net_share == 0:
                return pd.DataFrame()
            else:

                sub_df['成交均价'] = (df['手数'] * df['成交均价'] * c_replaced).sum(axis=0) / net_share
                sub_df['买卖'] = '买' if net_share > 0 else '卖'
                sub_df['开平'] = '开' if net_share > 0 else '平'
                sub_df['手数'] = np.abs(net_share)
                sub_df['手续费'] = sub_df['手数'] * sub_df['成交均价'] * 100 * (0.00432 / 100)
                return sub_df

    @staticmethod
    def short_side_reduce(df, direct_dict={'买平': -1, '卖开': 1}):

        c = df['买卖'] + df['开平']
        c_replaced = c.replace(direct_dict)
        mask = c.isin(direct_dict.keys())

        sub = df[mask]

        if sub.empty:
            return pd.DataFrame()
        elif sub.shape[0] == 1:
            return sub
        else:
            sub_df = df.head(1).copy(deep=True)
            net_share = (df['手数'] * c_replaced).sum(axis=0)

            if net_share == 0:
                return pd.DataFrame()
            else:
                sub_df['成交均价'] = (df['手数'] * df['成交均价'] * c_replaced).sum(axis=0) / net_share
                sub_df['买卖'] = '卖' if net_share > 0 else '买'
                sub_df['开平'] = '平' if net_share > 0 else '开'
                sub_df['手数'] = np.abs(net_share)
                sub_df['手续费'] = sub_df['手数'] * sub_df['成交均价'] * 100 * (0.00432 / 100)
                return sub_df

    def to_frame(self, reduce=False):
        if len(self) != 0:
            raw = pd.concat(self)
            if reduce:
                h = []

                for _, df in raw.groupby(['委托合约', '报单时间']):
                    c = df['买卖'] + df['开平']
                    mask = c.isin(['买开', '卖平'])

                    long = self.long_side_reduce(df[mask])

                    if not long.empty:
                        h.append(long)

                    short = self.short_side_reduce(df[~mask])

                    if not short.empty:
                        h.append(short)

                target_cols = ['报单日期', '委托合约', '手数', '未成交', '买卖', '开平', '成交均价', '挂单状态',
                               '报单时间',
                               '详细状态', '盈亏', '手续费', '成交号']

                return pd.concat(h)[target_cols]
            else:
                return raw
        else:
            return pd.DataFrame()


class OptionPortfolio(object):
    def __init__(self, selected_matrix, multiply_num=100, dt_col='dt', code_col='contract_code'):
        self._selected_matrix = selected_matrix
        self._multiply_num = multiply_num
        self._dt_col = dt_col
        self._code_col = code_col
        self._holder = PortfolioHolder()

        self._expire_dict = dict(self._selected_matrix[['contract_code', 'expire_day']].drop_duplicates().values)

    def records(self, reduce=False):

        raw = self._holder.to_frame(reduce=reduce)

        return raw

    def holdings(self, end_dt='2099-07-12', ):
        reduced_records = self.records(reduce=True)
        h = []

        long = self._records_to_holdings(reduced_records, end_dt, trade_type_selected=['买开', '卖平'])
        if not long.empty:
            long['方向'] = 'long'
            h.append(long)

        short = self._records_to_holdings(reduced_records, end_dt, trade_type_selected=['卖开', '买平'])
        if not short.empty:
            short['方向'] = 'short'
            h.append(short)

        return pd.concat(h)

    @staticmethod
    def _records_to_holdings(reduced_records, end_dt='2023-07-12',
                             trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, },
                             trade_type_selected=['买开', '卖平']):

        test_data = reduced_records[reduced_records['报单时间'] <= end_dt].copy(deep=True)

        test_data['买卖开平'] = test_data['买卖'] + test_data['开平']

        test_data = test_data[test_data['买卖开平'].isin(trade_type_selected)]

        test_data['手数方向'] = test_data['手数'] * test_data['买卖开平'].replace(trade_type_mark)

        res = test_data.groupby('委托合约')['手数方向'].sum().to_frame('手数')

        res['统计截至时间'] = end_dt

        return res[res['手数'] != 0].reset_index()

    @staticmethod
    def mapping_records_to_holdings(reduced_records,
                                    trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, },
                                    trade_type_selected=['买开', '卖平']):

        test_data = reduced_records.copy(deep=True)

        test_data['买卖开平'] = test_data['买卖'] + test_data['开平']
        test_data = test_data[test_data['买卖开平'].isin(trade_type_selected)]
        test_data['手数方向'] = test_data['手数'] * test_data['买卖开平'].replace(trade_type_mark)

        last = test_data.pivot_table(index='报单时间', columns='委托合约', values='手数方向', aggfunc='sum')

        return last.fillna(0).cumsum()

    @staticmethod
    def status_to_translations(dd):
        col_rename = {'dt': '报单日期', 'contract_code': '委托合约', 'fee': '成交均价'}

        dd = dd.rename(columns=col_rename)
        dd['买卖'] = ((dd['weight'] > 0) * 1).replace(1, '买').replace(0, '卖')
        dd['开平'] = '开'
        dd['手数'] = np.int64(dd['weight'].abs())
        dd['挂单状态'] = '全部成交'
        dd['报单时间'] = dd['报单日期']
        dd['详细状态'] = '全部成交'
        dd['盈亏'] = 0
        dd['手续费'] = dd['手数'] * dd['成交均价'] * 100 * (0.00432 / 100)
        dd['成交号'] = 'hedge1'
        dd['未成交'] = 0
        dd['remarks'] = 0
        target_cols = ['报单日期', '委托合约', '手数', '未成交', '买卖', '开平', '成交均价', '挂单状态', '报单时间',
                       '详细状态', '盈亏', '手续费', '成交号']

        dd['报单日期'] = pd.to_datetime(dd['报单日期']).dt.strftime('%Y-%m-%d')
        return dd[target_cols]

    @classmethod
    def _2_records(cls, dd):
        dd['weight'] = dd['weight'].fillna(0)
        dd_ = dd[dd['weight'] != 0]
        raw_records = cls.status_to_translations(dd_)
        raw_records.index = raw_records['委托合约']
        return raw_records

    @staticmethod
    def add_sell_records(raw_records, next_quote_dict, next_dt, expire_dict):
        h = []

        for _, _d in raw_records.copy(deep=True).iterrows():

            next_fee = next_quote_dict.get(_d['委托合约'])

            if _d['报单日期'] <= expire_dict.get(_d['委托合约']) and next_fee is not None:
                _d['成交均价'] = next_fee
                _d['报单时间'] = next_dt

                _d['买卖'] = opposite(_d['买卖'], ('买', '卖'))
                _d['开平'] = opposite(_d['开平'], ('开', '平'))

                _d['手续费'] = _d['手数'] * _d['成交均价'] * 100 * (0.00432 / 100)

                _d['报单日期'] = pd.to_datetime(_d['报单时间']).strftime('%Y-%m-%d')

                if _d.shape[0] == 13:
                    h.append(_d.to_frame().T)
                else:
                    h.append(_d.to_frame())
        if len(h) != 0:
            sell_records = pd.concat(h).convert_dtypes()

            dtypes_dict = raw_records.dtypes.to_dict()
            return sell_records.astype(dtypes_dict)
        else:
            return None

    @property
    def _index(self):
        return sorted(self._selected_matrix[self._dt_col].dt.strftime("%Y-%m-%d %H:%M:%S").unique().tolist())

    def next_dt_dict(self):
        return dict(zip(self._index[:-1], self._index[1:]))

    def _get_given_dt_matrix(self, dt):

        return self._selected_matrix[self._selected_matrix[self._dt_col] == dt]

    def _mapping_index(self):
        dts = self._index
        for dt in dts:
            yield dt, self._selected_matrix[self._dt_col] == dt

    def __iter__(self):
        return self  # 返回迭代器对象本身

    def __next__(self):

        t_mask = self._selected_matrix['t'] >= 0.01
        for dt, dt_selected in self._selected_matrix[t_mask].groupby(self._dt_col):
            ym_mask = dt_selected['ym'] == dt_selected['ym'].min()

            yield dt, dt_selected[ym_mask]

        # else:
        #     raise StopIteration  # 没有更多元素时停止迭代

    @staticmethod
    def strategy(dt_selected, *args, **kwargs):
        raise ValueError('not defined')

    def mapping_strategy(self, *args, **kwargs):

        for dt, dt_selected in self.__next__():
            trading_selected = self.strategy(dt_selected, *args, **kwargs)
            yield trading_selected

    @staticmethod
    def create_transactions(transactions, last_deliv_and_multi,
                            trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, },
                            reduced=False, return_df=False, ):
        merged_transactions = pd.merge(transactions, last_deliv_and_multi.reset_index(),
                                       left_on='委托合约', right_on='委托合约')
        transactions = ProcessReportSingle.prepare_transactions(merged_transactions, trade_type_mark=trade_type_mark)
        return transactions

    @classmethod
    def parse_transactions_with_quote_v2(cls, transactions, quote, lastdel_multi,
                                         trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, "买平今": -1, },
                                         ):
        transactions = cls.create_transactions(transactions, lastdel_multi, reduced=False, return_df=True,
                                               trade_type_mark=trade_type_mark)

        result_holder = [
            DerivativesItem.parse_bo2sc_so2pc(lastdel_multi, contract, sub_transaction, quote, return_dict=False) for
            contract, sub_transaction in
            transactions.groupby('委托合约')]

        l1, l2, l3, l4, l5, l6, s1, s2, s3, s4, s5, s6 = list(zip(*result_holder))
        # long
        long_info_dict = ProcessReport._merged_function(l3, l4, l6, l5, l1, l2, quote, direct='long')
        # short
        short_info_dict = ProcessReport._merged_function(s3, s4, s6, s5, s1, s2, quote, direct='short')

        return ChainMap(long_info_dict, short_info_dict)

    @classmethod
    def fast(cls, transactions, quote, lastdel_multi_hedge):

        info_dict = cls.parse_transactions_with_quote_v2(transactions, quote, lastdel_multi_hedge.set_index('委托合约'))

        commodity_contracts = transactions['委托合约'].unique().tolist()
        x_long_summary_info, x_short_summary_info = ReportAnalyst.create_summary_info(commodity_contracts,
                                                                                      info_dict,
                                                                                      symbol='hedge')
        summary_ls_merged = ReportAnalyst.long_short_merge('hedge', x_long_summary_info, x_short_summary_info)
        summary_ls_merged['zz1000'] = quote['000852.SH']
        # summary_ls_merged['msg'] = strategy_result.groupby('dt')['msg'].last()
        return summary_ls_merged


class CachedData(object):
    def __init__(self):
        pass

    @staticmethod
    def load_quote_greek(select='select.parquet', main_mark=1):
        selected = pd.read_parquet(select)
        masks = selected['main_mark'] >= main_mark
        mask_selected = selected[masks]
        mask_selected.index = mask_selected['contract_code']
        return mask_selected

    @staticmethod
    @file_cache(enable_cache=True, granularity='d', )
    def create_fake_greek_for_contract(mask_selected, fake_f_list=range(4100, 8000, 50)):
        # fake_f_list = sorted(mask_selected['k'].unique())
        h = []
        for fake_f in fake_f_list:
            for _, row_data in mask_selected.iterrows():
                dt = row_data['dt']
                contract_code = row_data['contract_code']
                current_fee = row_data['fee']
                fake_greeks = cal_greek_with_given_f(row_data, fake_f, DividendYield=0)
                h.append((fake_f, dt, contract_code, current_fee, *fake_greeks))

        fake_contract_code = pd.DataFrame(h,
                                          columns=['fake_f', 'dt', 'contract_code', 'current_fee', 'Delta', 'Gamma',
                                                   'Vega', 'Theta', 'Rho'])
        return fake_contract_code

    @staticmethod
    def create_fake_greek_matrix(selected_fake_contract_code, f_cols='fake_f'):
        fake_delta = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Delta')
        fake_gamma = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Gamma')
        fake_vega = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Vega')
        fake_rho = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Rho')
        fake_theta = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Theta')

        return fake_delta.T, fake_gamma.T, fake_vega.T, fake_theta.T, fake_rho.T

    @staticmethod
    def create_greek_matrix(selected_contract_code):

        # selected_contract_code = selected_contract_code
        # print(selected_contract_code)
        delta = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                          values='Delta')
        gamma = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                          values='Gamma')
        vega = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                         values='Vega')
        rho = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                        values='Rho')
        theta = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                          values='Theta')

        return delta.T, gamma.T, vega.T, theta.T, rho.T

    @classmethod
    def create_greek_matrix_all(cls, real, fake):
        r1 = real[['contract_code', 'f', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']]

        f1 = fake[['contract_code', 'fake_f', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']]
        f1.columns = ['contract_code', 'f', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']

        greek_df = pd.concat([f1, r1], axis=0)
        return cls.create_fake_greek_matrix(greek_df, f_cols='f')

        pass

    @classmethod
    def create_fake_portfolio_greek(cls, fake_contract_code, weights, code_list, crrt_f):
        # crrt_f = fake_contract_code['crrt_f'].unique().tolist()[0]
        fake_delta, fake_gamma, fake_vega, fake_theta, fake_rho = cls.create_fake_greek_matrix(fake_contract_code)

        portfolio_delta = fake_delta.reindex(columns=code_list).dot(weights) * 100 * crrt_f
        portfolio_gamma = fake_gamma.reindex(columns=code_list).dot(weights) * 100
        portfolio_vega = fake_vega.reindex(columns=code_list).dot(weights) * 100
        portfolio_rho = fake_rho.reindex(columns=code_list).dot(weights) * 100
        portfolio_theta = fake_theta.reindex(columns=code_list).dot(weights) * 100

        return portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta, portfolio_rho


class OptBundle(object):

    @staticmethod
    def delta_exposure_constraint(weights, Delta, f, target_delta_exposure):
        return np.dot(weights, Delta.loc[f, :]) * 100 * f - target_delta_exposure

    @staticmethod
    def delta(weights, Delta, f, ):
        return np.dot(weights, Delta.loc[f, :]) * 100 * f

    @staticmethod
    def gamma_constraint(weights, Gamma, f):
        return np.sum(weights * Gamma.loc[f, :] * 100)

    @staticmethod
    def vega_constraint(weights, Vega, f):
        return np.sum(weights * Vega.loc[f, :] * 100)

    @staticmethod
    def fee(weights, fee):
        cost = np.sum(weights * fee * 100) * -1
        return cost

    # 定义目标函数和约束条件
    # 定义Delta曲线约束
    @staticmethod
    def delta_curve_stability(weights, delta_martix, f, price_range):
        merged = delta_martix.reindex(index=price_range).dot(weights)

        # print(merged, price_range)

        return np.mean(np.power(np.gradient(merged, price_range - f), 2))

    #     @staticmethod
    #     def delta_curve_stability(weights, delta_martix, f, price_range):
    #         merged = delta_martix.reindex(index=price_range).dot(weights)
    #         return  np.mean( np.power(np.gradient(merged, price_range - f),2))

    # 定义合约数约束条件
    @staticmethod
    def contract_count_constraint(lots, max_contracts):
        return max_contracts - np.count_nonzero(lots)

    @staticmethod
    # 定义PNL计算
    def calculate_pnl(lots, deltas, prices, current_price):
        return np.sum(deltas * (prices - current_price) * lots)

    @classmethod
    # 定义PNL曲线稳定性约束
    def pnl_stability_constraint(cls, lots, fee, f, selected, k_list, price_range, tolerance=10):
        selected['weight'] = np.int32(lots)

        pnls = pd.Series({f_: cal_pnl(f_, selected) - fee for f_ in price_range}).to_frame('pnl')
        # print(fee, pnls)

        # 定义行权价附近的价格区间
        sub_pnls = pnls[pnls.index.isin(price_range)]['pnl']
        # print('sub_pnls:', sub_pnls.shape)
        # dx = np.diff(price_range).mean()  # 计算平均间隔
        # 计算价格间隔
        if isinstance(price_range, (list, np.ndarray)) and len(price_range) > 1:
            dx = np.diff(price_range).mean()  # 使用价格的平均间隔
            stability_measure = np.abs(np.gradient(sub_pnls, dx))
        else:
            # 价格间隔为1的默认情况
            stability_measure = np.abs(np.gradient(sub_pnls, edge_order=1))

        return tolerance - np.max(stability_measure)

    @classmethod
    def objective(cls, weights, fee, delta, f, mask_selected, price_range):
        cost = cls.fee(weights, fee)

        delta_curve = cls.delta_curve_stability(weights, delta, f, price_range)

        pnl_curve = cls.pnl_stability_constraint(weights, cost, f, mask_selected, delta.index, price_range)

        num_contracts = np.count_nonzero(weights)  # 合约数量
        total_share = np.sum(np.abs(weights))

        #         print('num_contracts: ', num_contracts, 'total_share: ', total_share)

        return np.abs(cost) + (np.abs(delta_curve) + np.abs(pnl_curve)) ** 2

    @classmethod
    def run_opt(cls, initial_weights, mask_selected, target_delta_exposure, Delta, Gamma, Vega, Rho, method='SLSQP',
                verbose=True):
        # create_fake_portfolio_greek(fake_contract_code, weight_df)

        f = mask_selected['f'].unique().tolist()[0]

        price_range = Delta[(Delta.index <= f + 100) & (Delta.index >= f - 200)].index

        fee = mask_selected['fee']

        total_share_constraint = LinearConstraint(np.ones(len(fee)), lb=-50, ub=50)

        # gamma_constraint_c = NonlinearConstraint(lambda w: cls.gamma_constraint(w, Gamma, f), lb=0, ub=10)

        # stability_constraint = {'type': 'ineq', 'fun': cls.delta_curve_stability, 'args': (Delta, f, price_range)}
        #
        # max_contracts = 8  # 设定最大合约数量

        c1 = (
            {'type': 'eq', 'fun': cls.delta_exposure_constraint, 'args': (Delta, f, target_delta_exposure)},
            # gamma_constraint_c,
            {'type': 'eq', 'fun': cls.gamma_constraint, 'args': (Gamma, f)},
            {'type': 'ineq', 'fun': cls.vega_constraint, 'args': (Vega, f)},
            total_share_constraint,
            #             stability_constraint,
            # {'type': 'ineq', 'fun': cls.fake_delta_upper_constraint, 'args': (Delta, f, price_range)}
        )

        objective = lambda w: cls.objective(w, fee, Delta, f, mask_selected, price_range)

        # 边界条件
        bounds = [(-50, 50)] * len(initial_weights)

        # 执行优化
        result = minimize(objective, initial_weights, constraints=c1, bounds=bounds, method=method)

        opt_weight = np.int32(result.x)

        if verbose:
            # 打印结果
            print(result)
            print('损失函数：', objective(opt_weight))
            print('f:', f)

            print("优化的合约:", mask_selected.index.tolist())
            print("优化前的权重:", initial_weights.values)
            print("优化后的权重:", opt_weight)
            print("Delta:", cls.delta(opt_weight, Delta, f))
            print("Gamma:", cls.gamma_constraint(opt_weight, Gamma, f))
            print("Vega:", cls.vega_constraint(opt_weight, Vega, f))
            print('cost:', cls.fee(opt_weight, fee))
        return result


def required_share(require_hedged_size, delta, f):
    return require_hedged_size / delta / f / 100


def _cal_portfolio_greek(weight, Delta, Gamma, Vega, Theta, Rho):
    d = Delta.dot(weight).to_frame('Delta')
    g = Gamma.dot(weight).to_frame('Gamma')
    v = Vega.dot(weight).to_frame('Vega')
    t = Theta.dot(weight).to_frame('Theta')
    r = Rho.dot(weight).to_frame('Rho')

    return pd.concat([d, g, v, t, r], axis=1)


def draw_greek_surface_pic2(res, selected, Delta, Gamma, Vega, Theta, Rho, num_plots=6, num_cols=3,
                            greek_alphabet=('Delta', 'Gamma', 'Vega', 'Theta', 'Rho')):
    f = selected['f'].unique().tolist()[0]
    fee = np.dot(res.x, selected['fee'])

    d = _cal_portfolio_greek(res.x, Delta, Gamma, Vega, Theta, Rho)

    num_rows = (num_plots + num_cols - 1) // num_cols  # 计算行数，确保每行5张图
    # 使用subplot创建图形排列
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_cols * num_rows), sharex=True, sharey=False)
    # pnl

    pnl = pd.Series({f_: cal_pnl(f_, selected) - fee for f_ in d.index}).to_frame('pnl')
    ax = axs.flatten()[0]
    ax.plot(pnl['pnl'])
    ax.set_title(f'PnL Surface-Portfolio', fontsize=10)
    ax.set_xlim(pnl.index.min(), pnl.index.max())
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('PnL')
    ax.axvline(x=f, color='r')  # 'r' 表示红色
    ax.axhline(y=0, color='b')  # 'b' 表示蓝色

    for c, greek in enumerate(greek_alphabet):
        ax = axs.flatten()[c + 1]
        ax.plot(d[greek])
        ax.set_title(f'{greek} Surface-Portfolio', fontsize=10)
        # ax.set_ylim(min_iv,max_iv)
        ax.set_xlim(d.index.min(), d.index.max())
        ax.set_xlabel('Strike Price')
        ax.set_ylabel(greek)
        ax.axvline(x=f, color='r')  # 'r' 表示红色
    #         ax.axhline(y=0, color='b')  # 'b' 表示蓝色

    # 隐藏多余的子图
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axs.flatten()[i])
    plt.tight_layout()
    plt.show()


def load_full_greek_data(name='full_greek_caled_marked_60m.parquet'):
    full_greek_caled_marked_60m = pd.read_parquet(name)
    full_greek_caled_marked_60m['contract_code'] = full_greek_caled_marked_60m['contract_code'].astype(str) + '.CFE'
    full_greek_caled_marked_60m['OTM'] = ((full_greek_caled_marked_60m['f'] - full_greek_caled_marked_60m['k']) *
                                          full_greek_caled_marked_60m['cp'].replace({'C': -1, 'P': 1}).astype(int)) >= 0

    return full_greek_caled_marked_60m


def find_third_friday(year, month):
    # 创建一个当月第一天的日期对象
    first_day_of_month = datetime.date(year, month, 1)

    # 找到当月第一天是星期几
    first_day_weekday = first_day_of_month.weekday()

    # 计算到第一个周五需要的天数差
    # 如果第一天是周五，weekday()将返回4，这样我们就不会进入循环

    days_to_first_friday = 4 - first_day_weekday

    # 计算第一个周五的日期
    first_friday = first_day_of_month + datetime.timedelta(days=days_to_first_friday)

    # 计算第三周五的日期
    # 两个周五之间相隔14天，因此第三周五是第一个周五之后的14天
    third_friday = first_friday + datetime.timedelta(weeks=2)

    return third_friday


def ym_to_expire_day(ym: str, working_days, year_prefix='20'):
    year = int(year_prefix + str(ym)[:2])
    m = int(str(ym)[2:])

    day = pd.to_datetime(find_third_friday(year, m))

    if day in working_days:
        return day
    else:
        return working_days[working_days >= day].head(1).dt.strftime('%Y-%m-%d').values[0]


if __name__ == '__main__':
    # select one day

    tool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()
    working_days = pd.to_datetime(tool_trade_date_hist_sina_df['trade_date'])

    #
    start_dt = '2023-01-01'

    hedge_size = -500 * 10000  # 设定对冲市值
    draw_pic = False

    full_greek_caled_marked_60m = load_full_greek_data(name='full_greek_caled_marked_rmed_60m.parquet')
    full_greek_caled_marked_60m_rd = full_greek_caled_marked_60m[full_greek_caled_marked_60m['main_mark'] >= 1]

    full_greek_caled_marked = load_full_greek_data(name='full_greek_caled_marked.parquet')

    ym_2_expire_func = partial(ym_to_expire_day, working_days=working_days)

    ym_2_expire_dict = dict(list(map(lambda x: (x, ym_2_expire_func(x)), full_greek_caled_marked['ym'].unique())))

    full_greek_caled_marked['expire_day'] = full_greek_caled_marked['ym'].replace(ym_2_expire_dict)
    full_greek_caled_marked_60m['expire_day'] = full_greek_caled_marked_60m['ym'].replace(ym_2_expire_dict)
    full_greek_caled_marked_60m_rd['expire_day'] = full_greek_caled_marked_60m_rd['ym'].replace(ym_2_expire_dict)

    lastdel_multi_hedge = pd.read_excel('lastdel_multi_hedge.xlsx')

    zz1000 = ak.stock_zh_index_daily(symbol="sh000852")

    quote = full_greek_caled_marked.pivot_table(index='dt', columns='contract_code', values='fee')
    quote['000852.SH'] = zz1000.pivot_table(index='date', values='close')

    zz1000_15m = pd.read_excel('zz1000_15m_v1.xlsx')
    zz1000_60m = zz1000_15m.set_index('dt').resample("H").last().dropna()
    zz1000_60m = zz1000_60m[['close']]
    zz1000_60m.columns = ['000852.SH']

    merged_quote_s4_matrix = pd.read_parquet('merged_quote_s4_matrix.parquet')
    s22 = merged_quote_s4_matrix['bais_a_std40'].ffill()
    upper = s22.rolling(4 * 125).std() * 1 + s22.rolling(4 * 125).median()


    def strategy(dt_selected, hedge_size, min_put_level=3, max_cost=200):

        cp_mask = dt_selected['cp'] == 'P'
        fee_mask = dt_selected['fee'] <= max_cost
        otm_mask = dt_selected['OTM']

        c_mask = dt_selected['main_mark'] >= min_put_level

        a_put = dt_selected[cp_mask & c_mask & fee_mask & otm_mask].sort_values('fee')  # .head(1)
        if a_put.empty:
            if min_put_level < dt_selected['main_mark'].max():
                return strategy(dt_selected, hedge_size, min_put_level=min_put_level - 1, max_cost=max_cost)
            else:
                raise ValueError('no option available!')
        else:
            a_put['weight'] = np.round(hedge_size / a_put['Delta'] / 100 / a_put['f'], 0)
            less_100_mask = a_put['weight'] <= 100
            if a_put[less_100_mask].empty:
                return strategy(dt_selected, hedge_size, min_put_level=min_put_level + 1, max_cost=max_cost)
            else:
                return a_put[less_100_mask].head(1)


    op1 = OptionPortfolio(full_greek_caled_marked_60m_rd)

    op1.strategy = strategy

    result = pd.concat(list(op1.mapping_strategy(hedge_size, min_put_level=4, max_cost=200)))

    dt_tick_60m = sorted(result['dt'].unique())
    res_matrix = result.pivot_table(index='dt', columns='contract_code', values='weight')

    w_diff = res_matrix.fillna(0).diff(1)
    w_diff.loc[w_diff.index[0], :] = res_matrix.loc[w_diff.index[0], :]

    t = w_diff.stack(-1).to_frame('weight').reset_index()
    # for s, e in zip(dt_tick_60m[:-1], dt_tick_60m[1:]):
    #     weight_matrix = result[(result['dt'] >= s) & (result['dt'] <= e)]
    #     weight_matrix.diff(1)
    print(2)

    # reduce result

    # reduced_records = op1.records(reduce=False).reset_index(drop=True)
    # # reduced_records
    # reduced_records['报单日期'] = pd.to_datetime(reduced_records['报单日期']).dt.strftime('%Y%m%d')
    # summary_ls_merged = op1.fast(reduced_records, quote, lastdel_multi_hedge)
    # draw_pnl_pic(summary_ls_merged, title='Short Pure-Put PNL@500w')
