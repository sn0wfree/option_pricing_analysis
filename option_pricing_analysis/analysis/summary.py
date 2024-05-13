import datetime
import os
from glob import glob

import pandas as pd
from ClickSQL import BaseSingleFactorTableNode

from option_pricing_analysis.analysis.option_analysis_monitor import WindHelper, ReportAnalyst, Configs, Tools


def chunk(obj, chunks=2000):
    if hasattr(obj, '__len__'):
        length = len(obj)
        for i in range(0, length, chunks):
            yield obj[i:i + chunks]


def _map_dict_get(keys, dictitems):
    for key in keys:
        val = dictitems.get(key, None)
        if isinstance(val, pd.DataFrame):
            return val
    else:
        raise KeyError(f"{keys} not in dict!")


def _map_dict_get_funcs(keys, dictitems, func):
    return func(_map_dict_get(keys, dictitems))


##输出
def create_summary_output_file_path(output_path='./'):
    output_path = './' if output_path is None else output_path
    today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')
    output_name_format = f'输出指标统计@{today_str}.xlsx'
    _path = os.path.join(output_path, output_name_format)
    return _path

def check_ls(data_dict, keyword):
    return {key: df for key, df in data_dict.items() if keyword in key and isinstance(df, pd.DataFrame)}

class DerivativeSummary(ReportAnalyst):
    @staticmethod
    def _cal_year_result(col, pnl, nv):
        res = pnl.resample('Y').last()
        res.index = res.index.year
        y1 = res.head(1)
        res2 = pd.concat([y1, res.fillna(0).diff(1).dropna()], axis=0)
        res2.index = list(map(lambda x: str(x) + '年收益', res2.index))
        for amt_k, amt_v in res2.to_dict().items():
            yield amt_k, amt_v
        # 收益率
        nv_year = nv.resample('Y').last()
        nv_year.index = nv_year.index.year
        y1 = (nv_year.head(1) - 1) / 1
        nv_ret2 = pd.concat([y1, nv_year.fillna(1).pct_change(1).dropna()], axis=0)
        nv_ret2.index = list(map(lambda x: str(x) + '年收益率', nv_ret2.index))
        for ret_k, ret_v in nv_ret2.to_dict().items():
            yield ret_k, ret_v

    @classmethod
    def cal_year_result(cls, df, pnl_col, nv_col):
        for k, v in cls._cal_year_result('cal_year_result', df[pnl_col], df[nv_col]):
            yield k, v

    @classmethod
    def cal_periods_result_v2(cls, df, pnl_col, nv_col, ):
        periods_results = {}
        # 根据列名分别计算

        periods_results['累计收益'] = df[pnl_col].iloc[-1]
        periods_results['当日收益'] = df[pnl_col].fillna(0).diff(periods=1).iloc[-1]
        periods_results['近一周收益'] = df[pnl_col].diff(periods=5).iloc[-1]
        periods_results['近一月收益'] = df[pnl_col].diff(periods=20).iloc[-1]

        periods_results['累计收益率'] = df[nv_col].iloc[-1] - 1
        periods_results['当日收益率'] = df[nv_col].fillna(1).pct_change(periods=1).iloc[-1]
        periods_results['近一周收益率'] = df[nv_col].pct_change(periods=5).iloc[-1]
        periods_results['近一月收益率'] = df[nv_col].pct_change(periods=20).iloc[-1]

        temp_year_result = dict(cls.cal_year_result(df, pnl_col, nv_col))

        periods_results.update(temp_year_result)

        return periods_results

    @classmethod
    def process_ph(cls, person_holder, person_list=['wj', 'gr'], return_cum=False, add_year_return=True):
        if isinstance(person_list, str) and person_list == 'ALL':
            person_list = list(person_holder.keys())

        all_person_dfs = []
        target_col = ['累计收益', '累计收益率', '当日收益', '当日收益率',
                      '近一周收益', '近一周收益率', '近一月收益', '近一月收益率']
        year_cols = []
        for df in person_holder.values():
            for year in set(df.index.year.tolist()):
                year_cols.append(str(year) + '年收益')
                year_cols.append(str(year) + '年收益率')

        year_cols = sorted(set(year_cols))
        # years = list(zip(*[set(df.index.year) ]))

        base_cols = ['累计净值', '累计净损益(右轴)', ]

        cum_index = []

        for prsn, df in person_holder.items():
            if prsn in person_list:
                filtered_col = base_cols + sorted(df.columns.tolist()[8:])
                # filtered_col = [item for item in df.columns if "损益" in item or "净值" in item]
                person_dict = {}  # pd.DataFrame(columns=target_col)
                for nv_col, pnl_col in chunk(filtered_col, 2):
                    metrics_info = cls.cal_periods_result_v2(df, pnl_col, nv_col)
                    name = nv_col[:2]
                    person_dict[(prsn, name)] = metrics_info
                    if name == '累计':
                        cum_index.append((prsn, name))
                remaining_cols = target_col + year_cols if add_year_return else target_col
                all_person_dfs.append(pd.DataFrame(person_dict).T.reindex(columns=remaining_cols))

        full_summary_df = pd.concat(all_person_dfs, axis=0)
        if return_cum:
            m = full_summary_df.index.isin(cum_index)
            simple_summary_df = full_summary_df[m]
            if len(simple_summary_df.index._levels) != 1:
                simple_summary_df.index = simple_summary_df.index._levels[0]
            return full_summary_df, simple_summary_df
        else:

            return full_summary_df

    @staticmethod
    def _check_contract_condition_closed_value(df):
        return df[df.diff(1).tail(1) != 0].tail(1).dropna(axis=1)

    @staticmethod
    def _check_contract_condition_all_zero(df):
        return df[df.isnull().all()].tail(1).dropna(axis=1)

    @staticmethod
    def _check_contract_condition_resid_shares(df):
        return df[df.tail(1) != 0].tail(1).dropna(axis=1)

    @classmethod
    def check_contract_conditions_v2(cls, sub_dict, reduce=False):
        holding_contracts_cols = []

        cum_closed_value = ['衍生品多头累计平仓价值', '衍生品空头累计平仓价值']
        cum_pnl = ['衍生品多头累计净损益', '衍生品空头累计净损益']
        share = ['衍生品多头剩余份数', '衍生品空头剩余份数']
        executed = ['衍生品多头累计行权收益', '衍生品空头累计行权收益']

        closed_value_mask = _map_dict_get_funcs(cum_closed_value, sub_dict, cls._check_contract_condition_closed_value)
        holding_contracts_cols.extend(closed_value_mask.columns.tolist())

        cum_pnl_mask = _map_dict_get_funcs(cum_pnl, sub_dict, cls._check_contract_condition_closed_value)
        holding_contracts_cols.extend(cum_pnl_mask.columns.tolist())

        share_mask = _map_dict_get_funcs(share, sub_dict, cls._check_contract_condition_resid_shares)
        holding_contracts_cols.extend(share_mask.columns.tolist())

        executed_mask = _map_dict_get_funcs(executed, sub_dict, cls._check_contract_condition_resid_shares)
        executed_code = set(executed_mask.columns.tolist())
        # holding_contracts_cols.extend(share_mask.columns.tolist())

        # share_mask = _map_dict_get_funcs(share, sub_dict, check_contract_condition_resid_shares)
        holding_contracts_cols = set(holding_contracts_cols) - executed_code if reduce else holding_contracts_cols

        return set(holding_contracts_cols)

    @classmethod
    def cal_today_result(cls, holding_value_df, cum_closed_value_df,
                         cum_pnl_df, nv_df_by_cum_pnl_1, cum_cost_df, share_df, cum_exe_df, checked_cols, lastdel_multi,
                         ls, dt):
        # 过滤其他日期
        holding_value_df_rd = holding_value_df[holding_value_df.index <= dt]
        cum_closed_value_df_rd = cum_closed_value_df[cum_closed_value_df.index <= dt]
        cum_pnl_df_rd = cum_pnl_df[cum_pnl_df.index <= dt]

        nv_df_by_cum_pnl_1_rd = nv_df_by_cum_pnl_1[nv_df_by_cum_pnl_1.index <= dt]
        cum_cost_df_rd = cum_cost_df[cum_cost_df.index <= dt]
        share_df_rd = share_df[share_df.index <= dt]

        cum_exe_df_rd = cum_exe_df[cum_exe_df.index <= dt]

        # holding_value_df, cum_closed_value_df,
        # cum_pnl_df, cum_cost_df, share_df

        if ls == '空头':
            print(1)
        today_result = {}
        for col in checked_cols:
            periods_results = {}
            cm = lastdel_multi.loc[col, 'CONTRACTMULTIPLIER']

            today_profit_loss = cum_pnl_df_rd[col].fillna(0).diff(1).iloc[-1]

            # 当日累计收益率涨跌幅
            today_open_cost = cum_cost_df_rd[col].iloc[-1]  # 累计开仓成本
            # prev_open_cost = abs(
            #     cum_cost_df[col].iloc[-2] if cum_cost_df[col].shape[0] > 1 else today_open_cost)  # 前一日累计开仓成本，如有
            # 当日累计收益率涨跌幅
            today_total_return_change = nv_df_by_cum_pnl_1_rd[col].fillna(1).pct_change(periods=1).iloc[-1]
            # else:
            #     today_total_return_change = np.nan  # 如果前一日累计开仓成本为零
            # 持仓合约数
            num_contracts = share_df_rd[col].iloc[-1] / cm
            # 持仓名义市值
            nominal_value = abs(holding_value_df_rd[col].iloc[-1])
            # 平仓价值
            closing_value = abs(cum_closed_value_df_rd[col].iloc[-1])
            # 行权价值
            exe_value = cum_exe_df_rd[col].iloc[-1]

            # 累计净损益
            cum_profit_loss = cum_pnl_df_rd[col].iloc[-1]
            # 累计净损益（%）
            if today_open_cost != 0:
                cum_profit_loss_perc = cum_profit_loss / today_open_cost
            else:
                cum_profit_loss_perc = float('nan')  # 如果累计开仓成本为零

            periods_results['累计收益'] = cum_pnl_df_rd[col].iloc[-1]
            periods_results['当日收益'] = cum_pnl_df_rd[col].fillna(0).diff(periods=1).iloc[-1]
            periods_results['近一周收益'] = cum_pnl_df_rd[col].diff(periods=5).iloc[-1]
            periods_results['近一月收益'] = cum_pnl_df_rd[col].diff(periods=20).iloc[-1]

            # nv = nv_df_by_cum_pnl_1[col]

            periods_results['累计收益率'] = nv_df_by_cum_pnl_1_rd[col].iloc[-1] - 1
            periods_results['当日收益率'] = nv_df_by_cum_pnl_1_rd[col].fillna(1).pct_change(periods=1).iloc[-1]
            periods_results['近一周收益率'] = nv_df_by_cum_pnl_1_rd[col].pct_change(periods=5).iloc[-1]
            periods_results['近一月收益率'] = nv_df_by_cum_pnl_1_rd[col].pct_change(periods=20).iloc[-1]

            year_result = dict(cls._cal_year_result(col, cum_pnl_df_rd[col], nv_df_by_cum_pnl_1_rd[col]))

            periods_results.update(year_result)
            if col == 'AG2408.SHF':
                print(1)
                # year_result = dict(cls._cal_year_result(col, cum_pnl_df[col], nv_df_by_cum_pnl_1[col]))

            today_result[col] = {
                '持仓方向': ls,
                '持仓手数': num_contracts,
                '当日收益': today_profit_loss,
                '当日累计收益率涨跌幅': today_total_return_change,
                '持仓名义市值': nominal_value,
                '平仓价值': closing_value,
                '行权价值': exe_value,
                '累计收益(元)': cum_profit_loss,
                '累计收益率（%）': cum_profit_loss_perc,
                **periods_results,
            }
        indicators_today = pd.DataFrame(today_result).T.sort_index()
        # 存在该合约没有被平掉且到期了，实际上被行权了
        last_mask = lastdel_multi['EXE_DATE'] >= holding_value_df_rd.index.max()

        contract_code_list = indicators_today.index
        lastdel_multi_reducd = lastdel_multi[(lastdel_multi.index.isin(contract_code_list)) & last_mask]
        mask = indicators_today.index.isin(lastdel_multi_reducd.index)
        indicators_today_reduce = indicators_today[mask]

        return indicators_today_reduce

    @classmethod
    def holding_contract_info(cls, info_dict, lastdel_multi, c2p, dt):
        cum_closed_value = ['衍生品多头累计平仓价值', '衍生品空头累计平仓价值']
        cum_pnl = ['衍生品多头累计净损益', '衍生品空头累计净损益']
        share = ['衍生品多头剩余份数', '衍生品空头剩余份数']
        executed = ['衍生品多头累计行权收益', '衍生品空头累计行权收益']
        holding_value = ['衍生品多头持仓价值', '衍生品空头持仓价值']
        cum_cost = ['衍生品多头累计开仓成本', '衍生品空头累计开仓成本']

        holding_summary_info = []

        for sub_dict in info_dict.maps:
            # 拆分空头和多头分开处理
            ls = list(sub_dict.keys())[0][3:5]
            holding_contracts_cols = cls.check_contract_conditions_v2(sub_dict)

            # df1 = {key: value for key, value in sub_dict.items() if '持仓价值' in key}  # 包含持仓价值
            holding_value_df = _map_dict_get(holding_value, sub_dict)

            # df2 = {key: value for key, value in sub_dict.items() if '累计平仓价值' in key}  # 包含累计平仓价值
            cum_closed_value_df = _map_dict_get(cum_closed_value, sub_dict)

            # df3 = {key: value for key, value in sub_dict.items() if '累计净损益' in key}  # 包含累计净损益
            cum_pnl_df = _map_dict_get(cum_pnl, sub_dict)

            # df4 = {key: value for key, value in sub_dict.items() if '累计开仓成本' in key}  # 包含累计开仓成本
            cum_cost_df = _map_dict_get(cum_cost, sub_dict)

            # 包含行权收益
            cum_exe_df = _map_dict_get(executed, sub_dict)

            pnl_mask = ~cum_pnl_df.isna()

            nv_df_by_cum_pnl_1 = cum_pnl_df / cum_cost_df[pnl_mask] + 1

            # df5 = {key: value for key, value in sub_dict.items() if '剩余份数' in key}  # 剩余合约数
            share_df = _map_dict_get(share, sub_dict)

            indicators_df = cls.cal_today_result(holding_value_df, cum_closed_value_df,
                                                 cum_pnl_df, nv_df_by_cum_pnl_1, cum_cost_df, share_df, cum_exe_df,
                                                 holding_contracts_cols, lastdel_multi, ls, dt)

            m_index = list(map(lambda x: (c2p.get(x, None), x), indicators_df.index))

            indicators_df.index = m_index
            holding_summary_info.append(indicators_df)

        return holding_summary_info

    def output_summary(self, person_holder, info_dict, lastdel_multi, base_store_path=None, return_data=False,
                       dt=datetime.datetime.today(), ):

        contracts = self.reduced_contracts()
        person_contract_summary = self.process_ph(person_holder)
        # columns = ['contract', 'person', 'symbol', 'commodity']
        c, p, s, comm = list(
            zip(*list(self.contract_link_person(contracts, contract_2_person_rule=self.contract_2_person_rule))))

        c2p = dict(zip(c, p))

        holding_contracts_summary = self.holding_contract_info(info_dict, lastdel_multi, c2p, dt)

        holding_contracts_summary_merged = pd.concat(holding_contracts_summary + [person_contract_summary],
                                                     axis=0).sort_index()

        store_path = create_summary_output_file_path(output_path=base_store_path)

        with pd.ExcelWriter(store_path) as f:
            person_contract_summary.to_excel(f, sheet_name='整体统计输出')
            holding_contracts_summary_merged.to_excel(f, sheet_name='整体统计输出2')
            holding_contracts_summary[0].to_excel(f, sheet_name='衍生品多头截面持仓合约统计')
            holding_contracts_summary[1].to_excel(f, sheet_name='衍生品空头截面持仓合约统计')
        if return_data:
            return person_contract_summary, holding_contracts_summary

    @staticmethod
    def process_person_by_year(person_holder, person_list):

        for person, df in person_holder.items():
            if person in person_list:
                current_pnl = df.diff(1)['累计净损益(右轴)'].to_frame(person)
                c_pnl = current_pnl.tail(1)
                c_pnl.index = ['当日损益']

                res = df.resample('Y').last()['累计净损益(右轴)'].to_frame(person)
                res.index = res.index.year
                y1 = res.head(1)
                cum = res.tail(1)
                cum.index = ['累计']

                res_out = res.diff(1).dropna()

                merged_person1 = pd.concat([c_pnl, y1, res_out, cum, ], axis=0)
                yield merged_person1

    @staticmethod
    def merge_prsn_contr_smmry_df_with_contr_info(holding_contracts_summary, person_contract_summary_all):

        yield person_contract_summary_all.reset_index().rename(
            columns={'level_0': 'person', 'level_1': 'contract'})

        for summary in holding_contracts_summary:
            # re_index_summary = summary.reset_index()
            person, contract = list(zip(*summary.index))
            summary['person'] = person
            summary['contract'] = contract
            yield summary

    @staticmethod
    def clean_order_for_summary_df(holding_contracts_summary_merged):

        for _person, _df in holding_contracts_summary_merged.groupby('person'):
            contract_list = _df['contract'].tolist()
            contract_list.pop(contract_list.index('累计'))

            sorted_contract_order = ['累计'] + sorted(contract_list)

            for order, c in enumerate(sorted_contract_order):
                m = _df['contract'] == c
                _df.loc[m, '_order'] = order
            _df = _df.sort_values(by='_order')

            idx_list = tuple(zip([_person] * (len(contract_list) + 1), sorted_contract_order))
            _df.index = pd.MultiIndex.from_tuples(idx_list)
            yield _df

    @staticmethod
    def latest_contract_ls(contract_summary_dict):
        res_ls = []
        for key, df in contract_summary_dict.items():
            contract_name, direction = key[:2], key[2:4]
            last_cum_res = df['累计净损益(右轴)'].iloc[-1]
            res_ls.append([contract_name, direction, last_cum_res])
        contract_last_cum = pd.DataFrame(res_ls, columns=['合约名称', '交易方向', '累计净损益']).pivot_table(index='合约名称',columns='交易方向',values='累计净损益')
        return contract_last_cum

    def output_v2(self, info_dict, lastdel_multi, output_config={'汇总': ['wj', 'gr', 'll'],
                                                                 '期货多头': ['wj', 'gr'],
                                                                 '期货对冲': ['IM'],
                                                                 '分品种': 'ALL',
                                                                 },

                  dt=datetime.datetime.today(),
                  trade_type_mark={"卖开": 1, "卖平": -1,
                                   "买开": 1, "买平": -1,
                                   "买平今": -1, },

                  contract_col='委托合约',
                  share_col='手数',
                  price_col='成交均价',
                  trade_type='trade_type',

                  method='FIFO'
                  ):

        person_holder, merged_summary_dict, contract_summary_dict = self.group_by_summary(
            info_dict, return_data=True, store_2_excel=False)

        target_cols = ['持仓方向', '持仓手数', '平均开仓成本', '现价', '当日收益', '当日收益率',
                       '持仓名义市值', '平仓价值', '行权价值',
                       '累计收益','累计收益率','近一周收益', '近一周收益率',
                       '近一月收益', '近一月收益率', ]

        year_cols = []
        for df in person_holder.values():
            for year in set(df.index.year.tolist()):
                year_cols.append(str(year) + '年收益')
                year_cols.append(str(year) + '年收益率')

        # 汇总：分人分年度：
        person_by_year_summary = pd.concat(
            self.process_person_by_year(person_holder, person_list=output_config['汇总']), axis=1)
        person_by_year_summary['累计盈亏'] = person_by_year_summary.sum(axis=1)
        person_by_year_summary.index.name = '统计维度'

        #分合约多空 要输出contract_by_ls
        contract_by_ls_summary = self.latest_contract_ls(contract_summary_dict)

        # 期货多头：分人分年度分品种
        # person_cum_sub 要输出的

        person_contract_summary_all, person_cum_sub_all = self.process_ph(person_holder,
                                                                          person_list=output_config['汇总'],
                                                                          return_cum=True)

        # person_contract_summary = person_contract_summary_all.loc[(output_config['期货多头'], slice(None)), :]
        person_cum_sub = person_cum_sub_all.loc[output_config['期货多头'], :]
        person_cum_sub.index.name = '交易员'

        # person_contract_summary, person_cum_sub = self.process_ph(person_holder, person_list=output_config['期货多头'],
        #                                                           return_cum=True)

        # 期货多头：分品种
        _, commodity_cum_sub = self.process_ph(merged_summary_dict, person_list=output_config['分品种'],
                                               return_cum=True, add_year_return=True)
        # 去除多空输出
        commodity_cum_sub['symbol'] = [comm[:-4] for comm in commodity_cum_sub.index]
        mask = ~(commodity_cum_sub == 0).all(axis=1)  # remove all zero rows
        comm_cum_sub = commodity_cum_sub[mask].set_index('symbol')  # 要输出的
        comm_cum_sub.index.name = '品种'

        # 分人分合约统计
        contracts = self.reduced_contracts()
        c, p, s, comm = list(
            zip(*list(self.contract_link_person(contracts, contract_2_person_rule=self.contract_2_person_rule))))
        c2p = dict(zip(c, p))

        holding_contracts_summary = self.holding_contract_info(info_dict, lastdel_multi, c2p, dt)
        tempc = list(
            self.merge_prsn_contr_smmry_df_with_contr_info(holding_contracts_summary, person_contract_summary_all))

        hld_contracts_smy_mrgd = pd.concat(
            self.clean_order_for_summary_df(pd.concat(tempc).reset_index(drop=True)))

        res_avg_price_rd_df = self.create_current_cost_price(lastdel_multi,
                                                             trade_type_mark=trade_type_mark,
                                                             contract_col=contract_col,
                                                             share_col=share_col,
                                                             price_col=price_col,
                                                             trade_type=trade_type,
                                                             dt_col='报单日期时间',
                                                             method=method)

        hld_contracts_smy_mrgd2 = pd.merge(hld_contracts_smy_mrgd,
                                           res_avg_price_rd_df[
                                               ['contract_code', '平均开仓成本', 'CONTRACTMULTIPLIER']],
                                           left_on=['contract'], right_on=['contract_code'], how='left')
        hld_contracts_smy_mrgd2.index = hld_contracts_smy_mrgd.index

        hld_contracts_smy_mrgd2['现价'] = hld_contracts_smy_mrgd2['持仓名义市值'] / (
                hld_contracts_smy_mrgd2['持仓手数'] * hld_contracts_smy_mrgd2[
            'CONTRACTMULTIPLIER'])

        hld_contracts_smy_mrgd2['现价'] = hld_contracts_smy_mrgd2['现价'].abs()

        holding_contracts_summary_merged_sorted = hld_contracts_smy_mrgd2[
            target_cols + sorted(set(year_cols))]
        holding_contracts_summary_merged_sorted.index.names = ['交易员', '统计维度']

        return person_by_year_summary, person_cum_sub, comm_cum_sub, holding_contracts_summary_merged_sorted,contract_by_ls_summary

    def auto_run(self, output_config,
                 quote_start_with='2022-06-04',
                 trade_type_mark={"卖开": 1, "卖平": -1,
                                  "买开": 1, "买平": -1,
                                  "买平今": -1, }):

        # config = Configs(conf_file=conf_file)

        today = datetime.datetime.today()
        # today_str = pd.to_datetime(today).strftime('%Y%m%d')

        contracts = self.reduced_contracts()
        wh = WindHelper()
        # ---------------------------

        quote = self.get_quote_and_info(contracts, wh, start_with=quote_start_with)

        lastdel_multi = self.get_info_last_delivery_multi(contracts, wh)

        info_dict = self.parse_transactions_with_quote_v2(quote, lastdel_multi,
                                                          trade_type_mark=trade_type_mark)

        person_holder_dict, merged_summary_dict, contract_summary_dict = self.group_by_summary(info_dict,
                                                                                               return_data=True,
                                                                                               store_2_excel=False)

        person_by_year_summary, person_cum_sub, commodity_cum_sub, holding_summary_merged_sorted,contract_by_ls_summary = self.output_v2(
            info_dict, lastdel_multi, output_config, dt=today, trade_type_mark=trade_type_mark)

        store_path = self.create_daily_summary_file_path(output_path='./', version='v3')

        with pd.ExcelWriter(store_path) as f:
            person_by_year_summary.to_excel(f, 'person_by_year_summary')
            person_cum_sub.to_excel(f, 'person_cum_sub')
            commodity_cum_sub.to_excel(f, 'commodity_cum_sub')
            holding_summary_merged_sorted.to_excel(f, 'holding_summary_merged_sorted')
            contract_by_ls_summary.to_excel(f, 'contract_by_ls_summary')

            for name, data in sorted(person_holder_dict.copy().items(), key=lambda d: d[0][:2], reverse=True):
                data.index = data.index.strftime('%Y-%m-%d')
                data.to_excel(f, name)
                Tools.create_draw_from_opened_excel(f, data.shape[0], target_sheet=name)
                print(f"{name} output!")

            ## 分人分项目分年度计算盈亏
            # for name, data in sorted(person_ls_summary_dict.copy().items(), key=lambda d: d[0][:2], reverse=True):
            #     data.index = data.index.strftime('%Y-%m-%d')
            #     data.to_excel(f, name)
            #     Tools.create_draw_from_opened_excel(f, data.shape[0], target_sheet=name)
            #     print(f"{name} output!")

            ## 分合约计算盈亏
            for name, data in sorted(merged_summary_dict.copy().items(), key=lambda d: d[0][:2], reverse=True):
                data.index = data.index.strftime('%Y-%m-%d')
                data.to_excel(f, name)
                Tools.create_draw_from_opened_excel(f, data.shape[0], target_sheet=name)
                print(f"{name} output!")

            for name, data in sorted(contract_summary_dict.copy().items(), key=lambda d: d[0][:2], reverse=True):
                data.index = data.index.strftime('%Y-%m-%d')
                data.to_excel(f, name)
                Tools.create_draw_from_opened_excel(f, data.shape[0], target_sheet=name)
                print(f"{name} output!")

            info_dict['衍生品多头持仓价值'].reindex(columns=contracts).to_excel(f, '衍生品多头持仓价值截面')
            info_dict['衍生品多头累计开仓成本'].reindex(columns=contracts).to_excel(f, '衍生品多头累计开仓成本')
            info_dict['衍生品多头累计平仓价值'].reindex(columns=contracts).to_excel(f, '衍生品多头累计平仓价值')
            info_dict['衍生品多头累计行权收益'].reindex(columns=contracts).to_excel(f, '衍生品多头累计行权收益')
            info_dict['衍生品多头累计净损益'].reindex(columns=contracts).to_excel(f, '衍生品多头累计净损益')
            info_dict['衍生品多头剩余份数'].reindex(columns=contracts).to_excel(f, '衍生品多头剩余合约数')

            info_dict['衍生品空头持仓价值'].reindex(columns=contracts).to_excel(f, '衍生品空头持仓价值截面')
            info_dict['衍生品空头累计开仓成本'].reindex(columns=contracts).to_excel(f, '衍生品空头累计开仓成本')
            info_dict['衍生品空头累计平仓价值'].reindex(columns=contracts).to_excel(f, '衍生品空头累计平仓价值')
            info_dict['衍生品空头累计行权收益'].reindex(columns=contracts).to_excel(f, '衍生品空头累计行权收益')
            info_dict['衍生品空头累计净损益'].reindex(columns=contracts).to_excel(f, '衍生品空头累计净损益')
            info_dict['衍生品空头剩余份数'].reindex(columns=contracts).to_excel(f, '衍生品空头剩余合约数')


if __name__ == '__main__':


    config = Configs()

    PR = DerivativeSummary(
        report_file_path=config['report_file_path'],
        contract_2_person_rule=config['contract_2_person_rule'])

    PR.auto_run(config['output_config'], quote_start_with='2022-06-04', trade_type_mark={"卖开": 1, "卖平": -1,
                                                                                         "买开": 1, "买平": -1,
                                                                                         "买平今": -1, })

    from upload import UploadDailyInfo
    file_name = max(list(glob('日度衍生品交易收益率统计及汇总@*v3.xlsx')))

    # result_dict = pd.read_excel(file_name, sheet_name=None)
    # summary = ['person_by_year_summary', 'person_cum_sub', 'commodity_cum_sub', 'holding_summary_merged_sorted', ]
    # sql_dict = config['sql_dict']
    # traders = config['output_config']['汇总']

    node = BaseSingleFactorTableNode(config['src'])
    UDI = UploadDailyInfo(file_name)
    UDI.upload_all(node, mappings_link=config['mappings_link'], sheet_key_word='输出',
                   traders=config['output_config']['汇总'], db=None, sql_dict=config['sql_dict'], reduce=False)

    pass

#   1、浮动盈亏+已实现盈亏（拆分累计净损益）
#   2、保证金占用
#   3、品种收益柱形图，多空、持仓占比饼图
