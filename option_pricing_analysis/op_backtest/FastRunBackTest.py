# coding=utf-8

import warnings

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from collections import ChainMap

from op1 import OptionPortfolioWithDT
from option_analysis_monitor import ProcessReport, DerivativesItem, ReportAnalyst, ProcessReportSingle
import matplotlib.pyplot as plt
from CodersWheel.QuickTool.boost_up import boost_up
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决‘-’bug
from CodersWheel.QuickTool.file_cache import file_cache


class FastRunBackTest(object):
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

        #         print(sub_test1
        info_dict = cls.parse_transactions_with_quote_v2(transactions, quote, lastdel_multi_hedge.set_index('委托合约'))

        commodity_contracts = transactions['委托合约'].unique().tolist()
        x_long_summary_info, x_short_summary_info = ReportAnalyst.create_summary_info(commodity_contracts,
                                                                                      info_dict,
                                                                                      symbol='hedge')
        summary_ls_merged = ReportAnalyst.long_short_merge('hedge', x_long_summary_info, x_short_summary_info)
        summary_ls_merged['zz1000'] = quote['000852.SH']
        summary_ls_merged['msg'] = strategy_result.groupby('dt')['msg'].last()
        return summary_ls_merged

    @staticmethod
    def status_to_transcations(dd):
        col_rename = {'dt': '报单日期', 'contract_code': '委托合约', 'fee': '成交均价'}

        dd = dd.rename(columns=col_rename)
        dd['买卖'] = ((dd['weight'] > 0) * 1).replace(1, '买').replace(0, '卖')
        dd['开平'] = '开'
        dd['手数'] = np.int32(dd['weight'].abs())
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
        return dd[target_cols]

    @staticmethod
    def create_sell_records(another_records, next_quote):
        another_records['报单日期'] = pd.to_datetime(next_quote.index[0])
        for s in another_records['委托合约'].unique():
            another_records.loc[s, '成交均价'] = next_quote[s].values
        another_records['开平'] = '平'
        another_records['买卖'] = another_records['买卖'].replace({'买': '卖', '卖': '买'})
        return another_records

    @classmethod
    def strategy_result_2_translation(cls, strategy_result, quote):
        h = []
        for dt, dd in strategy_result.groupby('dt'):
            records = cls.status_to_transcations(dd)
            c_dt = pd.to_datetime(records['报单日期'].unique()[0])
            h.append(records)
            l = quote[quote.index > c_dt].reindex(columns=records['委托合约'].unique())
            if not l.empty:
                another_records = records.copy(deep=True)
                another_records = cls.create_sell_records(another_records, l.head(1))
                h.append(another_records)
        test1 = pd.concat(h)

        test1['remarks'] = '0'

        sub_test1 = test1[~test1['成交均价'].isna()].reset_index(drop=True)

        # sub_test1['order_dt'] = sub_test1['order_dt'].dt.strftime('%Y-%m-%d')
        sub_test1['order_time'] = sub_test1['报单日期']
        sub_test1['报单日期'] = pd.to_datetime(sub_test1['报单日期'].dt.strftime('%Y-%m-%d'))
        return sub_test1

    @classmethod
    def strategy_result_2_v3(cls, strategy_result, quote, signal):

        h = []

        for dt, dd in strategy_result.groupby('dt'):
            records = cls.status_to_transcations(dd)
            c_dt = pd.to_datetime(records['报单日期'].unique()[0])
            h.append(records)
            l = quote[quote.index > c_dt].reindex(columns=records['委托合约'].unique())
            if not l.empty:
                another_records = records.copy(deep=True)
                another_records = cls.create_sell_records(another_records, l.head(1))
                h.append(another_records)
        test1 = pd.concat(h)

        test1['remarks'] = '0'

        sub_test1 = test1[~test1['成交均价'].isna()].reset_index(drop=True)

        # sub_test1['order_dt'] = sub_test1['order_dt'].dt.strftime('%Y-%m-%d')
        sub_test1['order_time'] = sub_test1['报单日期']
        sub_test1['报单日期'] = pd.to_datetime(sub_test1['报单日期'].dt.strftime('%Y-%m-%d'))
        return sub_test1


def load_full_greek_data(name='full_greek_caled_marked_60m.parquet'):
    full_greek_caled_marked_60m = pd.read_parquet(name)
    full_greek_caled_marked_60m['contract_code'] = full_greek_caled_marked_60m['contract_code'].astype(str) + '.CFE'
    full_greek_caled_marked_60m['OTM'] = ((full_greek_caled_marked_60m['f'] - full_greek_caled_marked_60m['k']) *
                                          full_greek_caled_marked_60m['cp'].replace({'C': -1, 'P': 1}).astype(int)) >= 0

    return full_greek_caled_marked_60m


class FastRunBackTest2(FastRunBackTest):

    @classmethod
    def create_transactions_by_strategy_result_v3(cls, strategy_result, hedge_size, otm_selected,
                                                  signal, ):

        strategy = {'低波&不跌': 'put_spread_call',
                    '低波&跌': 'optimized',
                    '高波&不跌': 'put_spread_call',
                    '高波&跌': 'optimized'}

        h = []

        quote = otm_selected.pivot_table(index='dt', columns='contract_code', values='fee').reindex(
            columns=otm_selected['contract_code'].unique())

        signal_v2 = cls.create_scenario(signal, si_cols=('P_IV_1m_high', 'predicted_short'))

        for scenario, sign in signal_v2.groupby('scenario'):
            if scenario in ('低波&跌', '高波&跌'):
                sub_test2 = cls.optimized_2_records(sign.index, strategy_result, quote)
                if sub_test2 is not None:
                    h.append(sub_test2)
            elif scenario in ('低波&不跌', '高波&不跌'):

                sub_test2 = cls.put_spread_call_2_records(sign.index, hedge_size, otm_selected, quote)
                if sub_test2 is not None:
                    h.append(sub_test2)

            else:
                pass
        sub_test1 = pd.concat(h)

        return sub_test1

    @staticmethod
    def create_scenario(signal, si_cols=('P_IV_1m_high', 'predicted_short')):
        s1, s2 = si_cols
        s1_mask = signal[s1] == 1  # 高波
        s2_mask = signal[s2] == 1  # 跌
        ss = s1_mask * 2 + s2_mask * 1
        replace_dict = {0: '低波&不跌', 1: '低波&跌', 2: '高波&不跌', 3: '高波&跌'}
        signal['scenario'] = ss.replace(replace_dict)
        return signal

    @classmethod
    def w_2_records(cls, dd, quote):
        dd['weight'] = dd['weight'].fillna(0)
        dd_ = dd[dd['weight'] != 0]

        records = cls.status_to_transcations(dd_)
        c_dt = pd.to_datetime(dd_['dt'].unique()[0])
        records.index = records['委托合约']
        yield records
        l = quote[quote.index > c_dt].reindex(columns=records['委托合约'].unique())
        if not l.empty:
            another_records = records.copy(deep=True)
            another_records = cls.create_sell_records(another_records, l.head(1))
            yield another_records

    @classmethod
    def put_call(cls, hedge_size, otm_selected, quote):
        op_portfolio = OptionPortfolioWithDT(otm_selected)
        a_lv2_call = op_portfolio.get_level_call(level=2)
        a_put = op_portfolio.get_level_put(level=2)
        _selected = op_portfolio._selected.copy()

        _selected['weight'] = 0
        # Delta, Gamma, Vega, Theta, Rho = op_portfolio.create_greek_matrix_all()

        w = hedge_size / a_put['Delta'] / 100 / a_put['f']
        _selected.loc[w.index, 'weight'] = w.values[0]
        _selected.loc[a_lv2_call.index, 'weight'] = w.values[0]
        yield from cls.w_2_records(_selected.copy(deep=True), quote)
    @classmethod
    def f(cls, *args):
        return list(cls.put_call(*args))

    @classmethod
    def put_spread_call_2_records(cls, dt_list, hedge_size, otm_selected, quote):
        h = []
        mask = otm_selected['dt'].isin(dt_list)
        tasks = [(hedge_size, dd_selected, quote) for dt, dd_selected in otm_selected[mask].groupby('dt')]

        for s in boost_up(cls.f, tasks, star=True):
            h.extend(s)

        if len(h) == 0:
            return None

        test1 = pd.concat(h)

        test1['remarks'] = '0'

        sub_test1 = test1[~test1['成交均价'].isna()].reset_index(drop=True)
        sub_test1['order_time'] = sub_test1['报单日期']
        sub_test1['报单日期'] = pd.to_datetime(sub_test1['报单日期'].dt.strftime('%Y-%m-%d'))
        return sub_test1

    @classmethod
    def optimized_2_records(cls, dt_list, strategy_result, quote):
        h = []
        mask = strategy_result['dt'].isin(dt_list)

        for dt, dd_selected in strategy_result[mask].groupby('dt'):
            h.extend(list(cls.w_2_records(dd_selected, quote)))

        if len(h) == 0:
            return None

        test1 = pd.concat(h)

        test1['remarks'] = '0'

        sub_test1 = test1[~test1['成交均价'].isna()].reset_index(drop=True)

        # sub_test1['order_dt'] = sub_test1['order_dt'].dt.strftime('%Y-%m-%d')
        sub_test1['order_time'] = sub_test1['报单日期']
        sub_test1['报单日期'] = pd.to_datetime(sub_test1['报单日期'].dt.strftime('%Y-%m-%d'))
        return sub_test1

    @classmethod
    def strategy_result_2_translation(cls, strategy_result, quote, status):
        h = []
        for dt, dd in strategy_result.groupby('dt'):
            records = cls.status_to_transcations(dd)

            c_dt = pd.to_datetime(records['报单日期'].unique()[0])
            h.append(records)
            l = quote[quote.index > c_dt].reindex(columns=records['委托合约'].unique())
            if not l.empty:
                another_records = records.copy(deep=True)
                another_records = cls.create_sell_records(another_records, l.head(1))
                h.append(another_records)
        test1 = pd.concat(h)

        test1['remarks'] = '0'

        sub_test1 = test1[~test1['成交均价'].isna()].reset_index(drop=True)

        # sub_test1['order_dt'] = sub_test1['order_dt'].dt.strftime('%Y-%m-%d')
        sub_test1['order_time'] = sub_test1['报单日期']
        sub_test1['报单日期'] = pd.to_datetime(sub_test1['报单日期'].dt.strftime('%Y-%m-%d'))
        return sub_test1


@file_cache(enable_cache=True, granularity='d')
def load_data(s=1):
    full_greek_caled_marked_60m = load_full_greek_data(name='full_greek_caled_marked_rmed_60m.parquet')

    full_greek_caled_marked = load_full_greek_data(name='full_greek_caled_marked.parquet')

    lastdel_multi_hedge = pd.read_excel('lastdel_multi_hedge.xlsx')

    zz1000 = ak.stock_zh_index_daily(symbol="sh000852")

    strategy_result = pd.read_parquet('strategy_result_60m_v2.parquet')

    merged_quote_s4_matrix = pd.read_parquet('merged_quote_s4_matrix.parquet')
    zz1000_15m = pd.read_excel('zz1000_15m_v1.xlsx')

    return full_greek_caled_marked, full_greek_caled_marked_60m, lastdel_multi_hedge, zz1000, strategy_result, merged_quote_s4_matrix, zz1000_15m


def my_function():
    start_dt = '2023-01-01'

    hedge_size = -500 * 10000  # 设定对冲市值
    draw_pic = False

    full_greek_caled_marked, full_greek_caled_marked_60m, lastdel_multi_hedge, zz1000, strategy_result, merged_quote_s4_matrix, zz1000_15m = load_data(
        s=1)
    signal = merged_quote_s4_matrix[['P_IV_1m_high', 'predicted_short']].dropna()
    signal = signal[signal.index >= start_dt]

    quote = full_greek_caled_marked.pivot_table(index='dt', columns='contract_code', values='fee')
    quote['000852.SH'] = zz1000.pivot_table(index='date', values='close')

    zz1000_60m = zz1000_15m.set_index('dt').resample("H").last().dropna()
    zz1000_60m = zz1000_60m[['close']]
    zz1000_60m.columns = ['000852.SH']

    # 只选主力合约
    main_contract_mask = full_greek_caled_marked_60m['t'] >= 0.02
    main_mark = full_greek_caled_marked_60m['main_mark'] >= 2  # select main contract
    # main_mark2 = full_greek_caled_marked_60m['main_mark'] < 5 # select main contract
    start_dt_mask = full_greek_caled_marked_60m['dt'] >= start_dt

    sub_full_greek_caled_marked_60m = full_greek_caled_marked_60m[main_mark & main_contract_mask & start_dt_mask]

    transactions = FastRunBackTest2.create_transactions_by_strategy_result_v3(strategy_result, hedge_size,
                                                                              sub_full_greek_caled_marked_60m, signal, )

    # summary_ls_merged = FastRunBackTest.fast(transactions, quote, lastdel_multi_hedge)

    # # 创建一个图形和一组坐标轴
    # fig, ax = plt.subplots(figsize=(10, 6))
    # a1 = ax.plot(summary_ls_merged.index, summary_ls_merged['累计净损益(右轴)'], 'r-',
    #              label='期权组合损益')  # 'r-' 表示红色线
    # ax.set_ylabel('期权组合损益@保护500w')
    # ax.legend(prop={'size': 10})
    # # 创建次坐标轴
    # ax2 = ax.twinx()
    # # 设置次坐标轴的标签
    # ax2.set_ylabel('指数点位')
    #
    # # 绘制次坐标轴上的数据
    #
    # a2 = ax2.plot(summary_ls_merged.index, summary_ls_merged['zz1000'], 'b-', label='中证1000')  # 'b-' 表示蓝色线
    #
    # # 添加图例
    # lines = [a1[0], a2[0]]
    # plt.legend(lines, ['期权组合损益', '中证1000'])
    # # 显示图形
    # plt.show()
    # 你的代码
    pass


if __name__ == '__main__':
    from cProfile import Profile

    prof = Profile()
    with prof:
        my_function()

    prof.print_stats()

    #

    pass
