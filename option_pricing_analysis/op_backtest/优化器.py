# coding: utf-8


import warnings
from collections import ChainMap

import akshare as ak
import numpy as np
import pandas as pd

# from QuantOPT.constraints.relaxer import RunOpt
# from QuantOPT.constraints.constraints import create_constraints_holder
# from QuantOPT.core.model_core import Holder

warnings.filterwarnings('ignore')

from op1 import OptionPortfolioWithDT, draw_greek_surface_pic2
from option_analysis_monitor import ProcessReport, DerivativesItem, ReportAnalyst, ProcessReportSingle
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决‘-’bug


def cal_core(otm_selected, hedge_size, draw_pic=False):
    op_portfolio = OptionPortfolioWithDT(otm_selected)
    Delta, Gamma, Vega, Theta, Rho = op_portfolio.create_greek_matrix_all()
    # res = OptBundle.run_opt(initial_weights, otm_selected, hedge_size, Delta, Gamma, Vega, Rho, method='SLSQP')
    res = op_portfolio.run_opt(hedge_size, Delta, Gamma, Vega, Theta, Rho, verbose=False)

    op_portfolio._selected['success'] = res.success
    op_portfolio._selected['msg'] = res.message

    if draw_pic:
        draw_greek_surface_pic2(res, op_portfolio._selected, num_plots=6, num_cols=3, )

    return op_portfolio._selected.copy(deep=True)


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
        transactions = cls.create_transactions(transactions, lastdel_multi, reduced=True, return_df=True,
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
    def fast(cls, strategy_result, quote, lastdel_multi_hedge):

        sub_test1 = cls.strategy_result_2_transcation(strategy_result, quote)


        info_dict = cls.parse_transactions_with_quote_v2(sub_test1, quote, lastdel_multi_hedge.set_index('委托合约'))

        commodity_contracts = sub_test1['委托合约'].unique().tolist()
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
    def strategy_result_2_transcation(cls, strategy_result, quote):
        h = []
        for dt, dd in strategy_result.groupby('dt'):
            records = cls.status_to_transcations(dd)
            h.append(records)
            l = quote[quote.index > records['报单日期'].unique()[0]].reindex(columns=records['委托合约'].unique())
            if not l.empty:
                #             print(l.head(1)[records['委托合约'].unique()])
                another_records = records.copy(deep=True)
                another_records = cls.create_sell_records(another_records, l.head(1))
                h.append(another_records)
        test1 = pd.concat(h)

        test1['remarks'] = '0'

        sub_test1 = test1[~test1['成交均价'].isna()].reset_index(drop=True)

        # sub_test1['order_dt'] = sub_test1['order_dt'].dt.strftime('%Y-%m-%d')
        sub_test1['order_time'] = '00:00:00'
        return sub_test1


def single_run(otm_selected):
    op_portfolio = OptionPortfolioWithDT(otm_selected)
    Delta, Gamma, Vega, Theta, Rho = op_portfolio.create_greek_matrix_all()
    # res = OptBundle.run_opt(initial_weights, otm_selected, hedge_size, Delta, Gamma, Vega, Rho, method='SLSQP')
    res = op_portfolio.run_opt(hedge_size, Delta, Gamma, Vega, Theta, Rho, verbose=False)

    op_portfolio._selected['success'] = res.success
    op_portfolio._selected['msg'] = res.message
    return op_portfolio._selected.copy(deep=True)


def single_run_v2(otm_selected, hedge_size):
    op_portfolio = OptionPortfolioWithDT(otm_selected)
    Delta, Gamma, Vega, Theta, Rho = op_portfolio.create_greek_matrix_all()
    # res = OptBundle.run_opt(initial_weights, otm_selected, hedge_size, Delta, Gamma, Vega, Rho, method='SLSQP')
    res = op_portfolio.run_opt(hedge_size, Delta, Gamma, Vega, Theta, Rho, verbose=False)
    op_portfolio._selected['success'] = res.success
    op_portfolio._selected['msg'] = res.message
    if not res.success:
        a_put = op_portfolio.get_level_put(level=2)
        op_portfolio._selected['weight'] = 0
        w = hedge_size / a_put['Delta'] / 100 / a_put['f']
        op_portfolio._selected.loc[w.index, 'weight'] = w
        return op_portfolio._selected.copy(deep=True)

    else:

        return op_portfolio._selected.copy(deep=True)


def create_tasks_data(full_greek_caled_marked, hedge_size, start_dt):
    dt_mask = full_greek_caled_marked['dt'] >= pd.to_datetime(start_dt)
    otm_mask = full_greek_caled_marked['OTM']

    for dt, df in full_greek_caled_marked[dt_mask & otm_mask].groupby('dt'):
        otm_selected = df[df['ym'] == df['ym'].min()]
        otm_selected.index = otm_selected['contract_code'].astype(str)

        yield otm_selected, hedge_size


if __name__ == '__main__':
    pass

    #
    start_dt = '2023-01-01'

    hedge_size = -500 * 10000  # 设定对冲市值
    draw_pic = False

    full_greek_caled_marked = pd.read_parquet('full_greek_caled_marked.parquet')
    full_greek_caled_marked['contract_code'] = full_greek_caled_marked['contract_code'].astype(str) + '.CFE'
    full_greek_caled_marked['OTM'] = ((full_greek_caled_marked['f'] - full_greek_caled_marked['k']) *
                                      full_greek_caled_marked['cp'].replace({'C': -1, 'P': 1}).astype(int)) >= 0
    #
    # # 只选主力合约
    # main_contract_mask = full_greek_caled_marked['t'] >= 0.02
    # main_mark = full_greek_caled_marked['main_mark'] >= 3  # select main contract
    # start_dt_mask = full_greek_caled_marked['dt'] >= start_dt
    #
    #
    #
    # sub_full_greek_caled_marked = full_greek_caled_marked[main_mark & main_contract_mask & start_dt_mask]
    # h = boost_up(single_run_v2, create_tasks_data(sub_full_greek_caled_marked, hedge_size, start_dt), star=True, core=19)
    # strategy_result = pd.concat(h)

    # In[38]:

    strategy_result = pd.read_parquet('strategy_result_v2.parquet')

    lastdel_multi_hedge = pd.read_excel('lastdel_multi_hedge.xlsx')

    zz1000 = ak.stock_zh_index_daily(symbol="sh000852")

    quote = full_greek_caled_marked.pivot_table(index='dt', columns='contract_code', values='fee')
    quote['000852.SH'] = zz1000.pivot_table(index='date', values='close')

    # strategy_result['contract_code'] = strategy_result['contract_code'].astype(str) + '.CFE'

    summary_ls_merged = FastRunBackTest.fast(strategy_result, quote, lastdel_multi_hedge)

    print(summary_ls_merged)
