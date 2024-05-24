# coding=utf-8

# from pycaret.regression import *

import datetime

import numpy as np
import pandas as pd
from ClickSQL import BaseSingleFactorTableNode

from option_pricing_analysis.analysis.option_analysis_monitor import WindHelper, Configs
from option_pricing_analysis.analysis.summary import DerivativeSummary
from option_pricing_analysis.strategy.stratrgy_1 import get_market_index_close_data_via_ak


class StrategyVolArb(object):
    @staticmethod
    def rename_cn_en():
        report_cols = ['报单日期', '委托合约', '成交均价', '报单时间', '手数', '未成交', '买卖', '开平', '盈亏',
                       '手续费',
                       '成交号', '挂单状态', '详细状态', '备注']

        rename_cols = ["order_dt", "contract_code", "average_trade_price", "order_time", "volume", "unexecuted_volume",
                       "buy_sell_indicator", "open_close_indicator", "profit_loss", "commission", "trade_number",
                       "order_status",
                       "detailed_status", "remarks"]

        report_name_dict = dict(zip(report_cols, rename_cols))
        return report_name_dict

    @staticmethod
    def rename_en_cn():
        report_cols = ['报单日期', '委托合约', '成交均价', '报单时间', '手数', '未成交', '买卖', '开平', '盈亏',
                       '手续费',
                       '成交号', '挂单状态', '详细状态', '备注']

        rename_cols = ["order_dt", "contract_code", "average_trade_price", "order_time", "volume", "unexecuted_volume",
                       "buy_sell_indicator", "open_close_indicator", "profit_loss", "commission", "trade_number",
                       "order_status",
                       "detailed_status", "remarks"]

        report_name_dict = dict(zip(rename_cols, report_cols))
        return report_name_dict

    @staticmethod
    def create_xxxx_report(transac_quote_1day, ls='买', oc='开', transac_name='op_arb_strategy', price_col='high',
                           code_col='contract_code', defualt_share=1, trading_fee=lambda x: x * 15 * 1.01):
        report = transac_quote_1day[[code_col, price_col]]
        report.columns = ['委托合约', '成交均价']
        report.index.name = '报单日期'
        report['报单时间'] = '00:00:00'

        report['手数'] = defualt_share
        report['未成交'] = 0
        report['买卖'] = ls
        report['开平'] = oc
        report['盈亏'] = 0
        report['手续费'] = trading_fee(defualt_share)
        report['成交号'] = transac_name
        report['盈亏'] = 0
        report['挂单状态'] = '全部成交'

        report['详细状态'] = '全部成交'
        report['备注'] = 'test'
        return report

    @classmethod
    def create_long_open_report(cls, transac_quote_1day, transac_name='op_arb_strategy', price_col='high',
                                code_col='contract_code', defualt_share=1, trading_fee=lambda x: x * 15 * 1.01):
        report = cls.create_xxxx_report(transac_quote_1day, ls='买', oc='开', transac_name=transac_name,
                                        price_col=price_col,
                                        code_col=code_col, defualt_share=defualt_share, trading_fee=trading_fee)
        return report

    @classmethod
    def create_long_close_report(cls, transac_quote_1day, transac_name='op_arb_strategy', price_col='high',
                                 code_col='contract_code', defualt_share=1, trading_fee=lambda x: x * 15 * 1.01):
        report = cls.create_xxxx_report(transac_quote_1day, ls='卖', oc='平', transac_name=transac_name,
                                        price_col=price_col,
                                        code_col=code_col, defualt_share=defualt_share, trading_fee=trading_fee)
        return report

    @classmethod
    def create_short_open_report(cls, transac_quote_1day, transac_name='op_arb_strategy', price_col='low',
                                 code_col='contract_code', defualt_share=1, trading_fee=lambda x: x * 15 * 1.01):
        report = cls.create_xxxx_report(transac_quote_1day, ls='卖', oc='开', transac_name=transac_name,
                                        price_col=price_col,
                                        code_col=code_col, defualt_share=defualt_share, trading_fee=trading_fee)
        return report

    @classmethod
    def create_short_close_report(cls, transac_quote_1day, transac_name='op_arb_strategy', price_col='low',
                                  code_col='contract_code', defualt_share=1, trading_fee=lambda x: x * 15 * 1.01):
        report = cls.create_xxxx_report(transac_quote_1day, ls='买', oc='平', transac_name=transac_name,
                                        price_col=price_col,
                                        code_col=code_col, defualt_share=defualt_share, trading_fee=trading_fee)
        return report

    @staticmethod
    def opt_selected_func_to_signal(ym_data, top_n=5):
        cp_mask = ym_data['cp'] == 'p'
        # 由于是套利不买实值期权
        otm_mask = ym_data['otm'] == 1
        # 只买当月和次月合约
        # 提前一周换仓
        week_end_mask = ym_data['log_days_to_maturity'] >= 0.02
        # 只看当月和近月合约
        month_end_mask = ym_data['log_days_to_maturity'] <= 0.15

        otm_10_mask = (1 - ym_data['moneyness']) <= 0.06

        resid_mask = ym_data['resid'].abs() <= 0  # 剔除bs无法计算的情况

        # bs_iv  > SABR_iv ,认为是高估波动率 否则是低估波动率
        signal_dict = {}
        for dt, data in ym_data[cp_mask & otm_mask & week_end_mask & month_end_mask & otm_10_mask & resid_mask].groupby(
                'dt'):

            fee300_mask = (data['fee'] <= 200) & (data['fee'] >= 10)

            selected_data = data[fee300_mask]

            over_vol = selected_data['sabr_bs_loss'] < 0
            under_vol = selected_data['sabr_bs_loss'] > 0

            signal_ = over_vol.sum() >= top_n

            if not selected_data.empty:

                sabr_bs_loss = selected_data['sabr_bs_loss'].values.tolist()

                over_pos = np.argmin(sabr_bs_loss)
                under_pos = np.argmax(sabr_bs_loss)
                # 卖,高估的波动率合约
                sell_d = selected_data[selected_data['sabr_bs_loss'] == sabr_bs_loss[over_pos]]
                # 买，买低估波动率合约
                buy_d = selected_data[selected_data['sabr_bs_loss'] == sabr_bs_loss[under_pos]]

                tradable_signal = (sell_d['sabr_iv'].values >= buy_d['sabr_iv'].values) and (
                        sell_d['bs_iv'].values >= buy_d['bs_iv'].values)

                sell = sell_d['contract_code'].values[0]
                buy = buy_d['contract_code'].values[0]

                if tradable_signal[0] and signal_:
                    #             print(signal_dict.keys())
                    signal_dict[dt] = (buy, sell)

        signal2 = pd.DataFrame(signal_dict, index=['buy', 'sell']).T
        signal2['traded_dt'] = signal2.index
        signal2['traded_dt'] = signal2['traded_dt'].shift(-1)
        return signal2

    @classmethod
    def signal_2_transactions(cls, signal, put_quote, defualt_share=10, transac_name='op_arb_strategy',
                              no_overnight=False):
        # contract_code = set(signal['buy'].unique().tolist() + signal['sell'].unique().tolist())

        # sub_put_quote = put_quote[put_quote['contract_code'].isin(contract_code)]
        h = []
        for dt, sub_signal in signal.dropna().groupby('traded_dt'):
            buy_code_list = sub_signal['buy'].unique().tolist()

            temp_quote = put_quote.loc[dt]


            t_buy_quote = temp_quote[temp_quote['contract_code'].isin(buy_code_list)]
            try:
                buy_delta = t_buy_quote['delta'].values[0]
            except Exception as e:
                print(1)

            buy_report = cls.create_long_open_report(t_buy_quote, price_col='open', transac_name=transac_name,
                                                     defualt_share=defualt_share)
            h.append(buy_report)

            if no_overnight:
                buy_close_report = cls.create_long_close_report(t_buy_quote, price_col='close',
                                                                transac_name=transac_name,
                                                                defualt_share=defualt_share)
                h.append(buy_close_report)

            sell_code_list = sub_signal['sell'].unique().tolist()
            t_sell_quote = temp_quote[temp_quote['contract_code'].isin(sell_code_list)]
            sell_delta = t_sell_quote['delta'].values[0]

            sell_share = round(buy_delta * defualt_share / sell_delta, 0)
            if sell_share <= 0 or sell_share is None:
                print(1)

            sell_report = cls.create_short_open_report(t_sell_quote, price_col='open', transac_name=transac_name,
                                                       defualt_share=sell_share)
            h.append(sell_report)

            if no_overnight:
                sell_close_report = cls.create_short_close_report(t_sell_quote, price_col='close',
                                                                  transac_name=transac_name,
                                                                  defualt_share=sell_share)
                h.append(sell_close_report)

        report = pd.concat(h).reset_index()
        return report


def op_quote(sql, node, underlying_endpos=2,
             ym_length=4,
             cp_length=3):
    raw = node(sql)

    raw['prefix'] = raw['contract_code'].apply(lambda x: x[:underlying_endpos])
    raw['ym'] = raw['contract_code'].apply(lambda x: x[underlying_endpos:underlying_endpos + ym_length])
    raw['cp'] = raw['contract_code'].apply(
        lambda x: x[underlying_endpos + ym_length:underlying_endpos + ym_length + cp_length].strip('-').lower())

    raw['strike'] = raw['contract_code'].apply(
        lambda x: int(x[underlying_endpos + ym_length + cp_length:].split('.')[0]))
    return raw


def parse_iv_data(ym_data):
    ym_data['moneyness'] = ym_data['k'] / ym_data['f']
    ym_data['log_days_to_maturity'] = np.log(1 + ym_data['t'])
    ym_data['sabr_bs_loss'] = ym_data['sabr_iv'] - ym_data['bs_iv']

    c_itm = (ym_data['moneyness'] < 1) & (ym_data['cp'] == 'c')
    c_otm = (ym_data['moneyness'] > 1) & (ym_data['cp'] == 'c')

    ym_data.loc[c_otm, 'otm'] = 1
    ym_data.loc[c_itm, 'itm'] = 1

    p_itm = (ym_data['moneyness'] > 1) & (ym_data['cp'] == 'p')
    p_otm = (ym_data['moneyness'] < 1) & (ym_data['cp'] == 'p')

    ym_data.loc[p_otm, 'otm'] = 1
    ym_data.loc[p_itm, 'itm'] = 1

    ym_data['otm'] = ym_data['otm'].fillna(0)
    ym_data['itm'] = ym_data['itm'].fillna(0)
    return ym_data


if __name__ == '__main__':
    from option_pricing_analysis.op_backtest.op_backtest import MockBackTest

    uri = 'clickhouse://default:Imsn0wfree@10.67.20.52:8123/system'

    node = BaseSingleFactorTableNode('clickhouse://data:data@47.104.186.157:8123/system')

    node_local = BaseSingleFactorTableNode(uri)

    start = '2020-12-30'
    end = '2024-12-31'
    cp = 'p'

    underlying_endpos = 2
    ym_length = 4
    cp_length = 3
    transac_name = 'op_arb_strategy_test1_put'

    wh = WindHelper()
    config = Configs()

    # 导入期权日行情数据
    _sub_sql = f"select contract_code,trade_date,open,high,low,close,pre_settlement,delta,volume,amount,open_int,chg_open_int from quote.view_quote_mo where trade_date>='{start}' and trade_date<='{end}'"
    # moqqm = MOQuoteQuickMatrix(node, start='2024-01-01', end='2024-12-31')

    raw = op_quote(_sub_sql, node, underlying_endpos=underlying_endpos, ym_length=ym_length, cp_length=cp_length)
    put_quote = raw[(raw['cp'] == cp)].set_index('trade_date')

    # 中证1000指数
    zz1000 = get_market_index_close_data_via_ak(None, ['000852.SH'], ['中证1000'], start=start,
                                                end=datetime.datetime.today().strftime("%Y-%m-%d"), period='D',
                                                comm_fut_code_dict={'NH0100.NHF': 'NHCI'})

    # 导入iv

    ym_data = node_local(f"select *,k/f as moneyness from quote.temp_res_bs_sabr_iv where 1 and cp = '{cp}'   ")
    ym_data = parse_iv_data(ym_data)

    lastdel_multi = DerivativeSummary.get_info_last_delivery_multi(ym_data['contract_code'].unique().tolist(),
                                                                   wh).reset_index()
    lastdel_multi['ym'] = lastdel_multi['委托合约'].apply(lambda x: x[underlying_endpos:underlying_endpos + ym_length])
    ym2exec = dict(lastdel_multi[['ym', 'EXE_DATE']].drop_duplicates().values)

    # 偏离点策略_put
    signal2 = StrategyVolArb.opt_selected_func_to_signal(ym_data, top_n=2)

    report = StrategyVolArb.signal_2_transactions(signal2, put_quote, defualt_share=10, transac_name=transac_name,
                                                  no_overnight=True)

    upload_report = report.rename(columns=StrategyVolArb.rename_cn_en())

    node_local('truncate table mock_op_backtest.trial_1_put')
    node_local.insert_df(upload_report, 'mock_op_backtest', 'trial_1_put')
    node_local('optimize table mock_op_backtest.trial_1_put final')

    PR = MockBackTest(
        uri, transac_name, 'trial_1_put', db='mock_op_backtest',
        contract_2_person_rule={'MO\d{4}-[CP]-[0-9]+.CFE': 'll', }
    )
    PR.auto(wh, f'trail_1_{transac_name}.xlsx',

            start_with='2022-01-04', trade_type_mark={"卖开": 1, "卖平": -1,
                                                      "买开": 1, "买平": -1,
                                                      "买平今": -1, })

    pass
