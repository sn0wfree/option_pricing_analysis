import datetime
import os

import pandas as pd

from option_pricing_analysis.analysis.option_analysis_monitor import WindHelper, ReportAnalyst, Configs


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


class DerivativeSummary(ReportAnalyst):
    @staticmethod
    def _cal_year_result(pnl, nv):
        res = pnl.resample('Y').last()
        res.index = res.index.year
        y1 = res.head(1)
        res2 = pd.concat([y1, res.diff(1).dropna()], axis=0)
        res2.index = list(map(lambda x: str(x) + '年收益', res2.index))
        for k, v in res2.to_dict().items():
            yield k, v
        # 收益率
        nv_ret = nv.resample('Y').last()
        nv_ret.index = nv_ret.index.year
        y1 = nv_ret.head(1) - 1
        nv_ret2 = pd.concat([y1, nv_ret.pct_change(1).dropna()], axis=0)
        nv_ret2.index = list(map(lambda x: str(x) + '年收益率', nv_ret2.index))
        for k, v in nv_ret2.to_dict().items():
            yield k, v

    @staticmethod
    def cal_year_result(df, pnl_col, nv_col):
        # years = range(start_year, datetime.datetime.today().year + 1)
        # 收益
        res = df[pnl_col].resample('Y').last()
        res.index = res.index.year
        y1 = res.head(1)
        res2 = pd.concat([y1, res.diff(1).dropna()], axis=0)
        res2.index = list(map(lambda x: str(x) + '年收益', res2.index))
        for k, v in res2.to_dict().items():
            yield k, v
        # 收益率
        nv_ret = df[nv_col].resample('Y').last()
        nv_ret.index = nv_ret.index.year
        y1 = nv_ret.head(1) - 1
        nv_ret2 = pd.concat([y1, nv_ret.pct_change(1).dropna()], axis=0)
        nv_ret2.index = list(map(lambda x: str(x) + '年收益率', nv_ret2.index))
        for k, v in nv_ret2.to_dict().items():
            yield k, v

    @classmethod
    def cal_periods_result_v2(cls, df, pnl_col, nv_col, ):
        periods_results = {}
        # 根据列名分别计算

        periods_results['累计收益'] = df[pnl_col].iloc[-1]
        periods_results['当日收益'] = df[pnl_col].diff(periods=1).iloc[-1]
        periods_results['近一周收益'] = df[pnl_col].diff(periods=5).iloc[-1]
        periods_results['近一月收益'] = df[pnl_col].diff(periods=20).iloc[-1]

        periods_results['累计收益率'] = df[nv_col].iloc[-1] - 1
        periods_results['当日收益率'] = df[nv_col].pct_change(periods=1).iloc[-1]
        periods_results['近一周收益率'] = df[nv_col].pct_change(periods=5).iloc[-1]
        periods_results['近一月收益率'] = df[nv_col].pct_change(periods=20).iloc[-1]

        periods_results.update(dict(cls.cal_year_result(df, pnl_col, nv_col)))

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
                         cum_pnl_df, cum_cost_df, share_df, checked_cols, lastdel_multi, ls):
        # holding_value_df, cum_closed_value_df,
        # cum_pnl_df, cum_cost_df, share_df

        today_result = {}
        for col in checked_cols:
            periods_results = {}
            cm = lastdel_multi.loc[col, 'CONTRACTMULTIPLIER']

            today_profit_loss = cum_pnl_df[col].iloc[-1] - cum_pnl_df[col].iloc[-2]

            # 当日累计收益率涨跌幅
            today_open_cost = abs(cum_cost_df[col].iloc[-1])  # 累计开仓成本
            prev_open_cost = abs(
                cum_cost_df[col].iloc[-2] if cum_cost_df[col].shape[0] > 1 else today_open_cost)  # 前一日累计开仓成本，如有
            if prev_open_cost != 0:
                today_total_return_change = (1 + cum_pnl_df[col].iloc[-1] / today_open_cost) / (
                        1 + cum_pnl_df[col].iloc[-2] / prev_open_cost) - 1
            else:
                today_total_return_change = float('nan')  # 如果前一日累计开仓成本为零
            # 持仓合约数
            num_contracts = share_df[col].iloc[-1] / cm
            # 持仓名义市值
            nominal_value = abs(holding_value_df[col].iloc[-1])
            # 平仓价值
            closing_value = abs(cum_closed_value_df[col].iloc[-1])
            # 累计净损益
            cum_profit_loss = cum_pnl_df[col].iloc[-1]
            # 累计净损益（%）
            if today_open_cost != 0:
                cum_profit_loss_perc = cum_profit_loss / today_open_cost
            else:
                cum_profit_loss_perc = float('nan')  # 如果累计开仓成本为零

            nv = cum_pnl_df[col] + 1

            periods_results['累计收益'] = cum_pnl_df[col].iloc[-1]
            periods_results['当日收益'] = cum_pnl_df[col].diff(periods=1).iloc[-1]
            periods_results['近一周收益'] = cum_pnl_df[col].diff(periods=5).iloc[-1]
            periods_results['近一月收益'] = cum_pnl_df[col].diff(periods=20).iloc[-1]

            periods_results['累计收益率'] = nv.iloc[-1] - 1
            periods_results['当日收益率'] = nv.pct_change(periods=1).iloc[-1]
            periods_results['近一周收益率'] = nv.pct_change(periods=5).iloc[-1]
            periods_results['近一月收益率'] = nv.pct_change(periods=20).iloc[-1]

            periods_results.update(dict(cls._cal_year_result(cum_pnl_df[col], nv)))

            today_result[col] = {
                '持仓方向': ls,
                '持仓手数': num_contracts,
                '当日收益': today_profit_loss,
                '当日累计收益率涨跌幅': today_total_return_change,
                '持仓名义市值': nominal_value,
                '平仓价值': closing_value,
                '累计收益': cum_profit_loss,
                '累计收益率（%）': cum_profit_loss_perc,
                **periods_results,
            }
        indicators_today = pd.DataFrame(today_result).T.sort_index()
        mask = (indicators_today['持仓名义市值'] != 0) | (indicators_today['平仓价值'] != 0)
        return indicators_today[mask]

    @classmethod
    def holding_contract_info(cls, info_dict, lastdel_multi, c2p):
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

            # df5 = {key: value for key, value in sub_dict.items() if '剩余份数' in key}  # 剩余合约数
            share_df = _map_dict_get(share, sub_dict)

            indicators_df = cls.cal_today_result(holding_value_df, cum_closed_value_df,
                                                 cum_pnl_df, cum_cost_df, share_df,
                                                 holding_contracts_cols, lastdel_multi, ls)

            indicators_df.index = list(map(lambda x: (c2p.get(x, None), x), indicators_df.index))
            holding_summary_info.append(indicators_df)

        return holding_summary_info

    def output_summary(self, person_holder, info_dict, lastdel_multi, base_store_path=None, return_data=False):
        person_contract_summary = self.process_ph(person_holder)
        # columns = ['contract', 'person', 'symbol', 'commodity']
        c, p, s, comm = list(
            zip(*list(self.contract_link_person(contracts, contract_2_person_rule=self.contract_2_person_rule))))

        c2p = dict(zip(c, p))

        holding_contracts_summary = self.holding_contract_info(info_dict, lastdel_multi, c2p)

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
                merged_person1 = pd.concat([c_pnl, y1, res_out, cum], axis=0)
                yield merged_person1

    def output_v2(self, info_dict, lastdel_multi, output_config={'汇总': ['wj', 'gr', 'll'],
                                                                 '期货多头': ['wj', 'gr'],
                                                                 '期货对冲': ['IM'],
                                                                 '分品种': 'ALL',
                                                                 }):
        person_holder, merged_summary_dict, contract_summary_dict = PR.group_by_summary(
            info_dict, return_data=True, store_2_excel=False)
        # 汇总：分人分年度：
        person_by_year_summary = pd.concat(
            self.process_person_by_year(person_holder, person_list=output_config['汇总']), axis=1)
        person_by_year_summary['累计盈亏'] = person_by_year_summary.sum(axis=1)

        # 期货多头：分人分年度
        # person_cum_sub 要输出的
        person_contract_summary, person_cum_sub = self.process_ph(person_holder, person_list=output_config['期货多头'],
                                                                  return_cum=True)
        person_contract_summary_all, person_cum_sub_all = self.process_ph(person_holder,
                                                                          person_list=output_config['汇总'],
                                                                          return_cum=True)
        # 缺一个汇总的

        # 期货多头：分品种

        _, commodity_cum_sub = self.process_ph(merged_summary_dict, person_list=output_config['分品种'],
                                               return_cum=True, add_year_return=True)

        commodity_cum_sub['symbol'] = [comm[:-4] for comm in commodity_cum_sub.index]

        mask = ~(commodity_cum_sub == 0).all(axis=1)  # remove all zero rows

        comm_cum_sub = commodity_cum_sub[mask].set_index('symbol')  # 要输出的
        comm_cum_sub.index.name = '品种'

        # 分人分品种统计

        c, p, s, comm = list(
            zip(*list(self.contract_link_person(contracts, contract_2_person_rule=self.contract_2_person_rule))))

        c2p = dict(zip(c, p))

        holding_contracts_summary = self.holding_contract_info(info_dict, lastdel_multi, c2p)

        holding_contracts_summary_merged = pd.concat(holding_contracts_summary + [person_contract_summary_all],
                                                     axis=0).sort_index()

        c1, c2 = list(zip(*holding_contracts_summary_merged.index))

        holding_contracts_summary_merged['person'] = c1
        holding_contracts_summary_merged['contract'] = c2

        holding_contracts_summary_merged.index = pd.MultiIndex.from_tuples(holding_contracts_summary_merged.index)

        for _person, _df in holding_contracts_summary_merged.groupby('person'):
            contract_list = _df['contract'].tolist()

            contract_list.pop(contract_list.index('累计'))
            additional_index = pd.MultiIndex.from_tuples(

                )

            idx_list = zip([_person] * (len(contract_list) + 1), ['累计'] + contract_list)

            print(1)

        return person_by_year_summary, person_cum_sub, commodity_cum_sub,holding_contracts_summary_merged


def cal_periods_result(df, col):
    periods_results = {}
    # 根据列名分别计算
    if "损益" in col:
        periods_results['累计收益'] = df[col].iloc[-1]
        periods_results['当日收益'] = df[col].diff(periods=1).iloc[-1]
        periods_results['近一周收益'] = df[col].diff(periods=5).iloc[-1]
        periods_results['近一月收益'] = df[col].diff(periods=20).iloc[-1]
    if "净值" in col:
        periods_results['累计收益率'] = df[col].iloc[-1] - 1
        periods_results['当日收益率'] = df[col].pct_change(periods=1).iloc[-1]
        periods_results['近一周收益率'] = df[col].pct_change(periods=5).iloc[-1]
        periods_results['近一月收益率'] = df[col].pct_change(periods=20).iloc[-1]

    return periods_results


##单个合约
def check_contract_conditions(info_dict, dict_index):
    target_dict = info_dict.maps[dict_index]
    holding_contracts_cols = set()
    for df_name, df in target_dict.items():
        for column in df.columns:
            if df[column].isnull().all():
                continue
            # 检查是否是累计净损益或累计平仓价值的DataFrame
            ## todo: df中不存在累计净损益，第一个条件无效
            if '累计净损益' in df_name or '累计平仓价值' in df_name:
                # 执行最新值-前值是否为0的判断
                if df[column].iloc[-1] - df[column].iloc[-2] != 0:
                    holding_contracts_cols.add(column)
            # 检查是否是剩余合约数的DataFrame
            elif '剩余份数' in df_name:
                # 执行最新值是否为0的判断
                if df[column].iloc[-1] != 0:
                    holding_contracts_cols.add(column)
    return list(holding_contracts_cols)


##输出
def create_summary_output_file_path(output_path='./'):
    output_path = './' if output_path is None else output_path
    today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')
    output_name_format = f'输出指标统计@{today_str}.xlsx'
    _path = os.path.join(output_path, output_name_format)
    return _path


if __name__ == '__main__':
    config = Configs()

    today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')

    wh = WindHelper()

    PR = DerivativeSummary(
        report_file_path=config['report_file_path'],
        contract_2_person_rule=config['contract_2_person_rule']
    )

    contracts = PR.reduced_contracts()

    quote = PR.get_quote_and_info(contracts, wh, start_with='2022-06-04')

    lastdel_multi = PR.get_info_last_delivery_multi(contracts, wh)

    info_dict = PR.parse_transactions_with_quote_v2(quote, lastdel_multi,
                                                    trade_type_mark={"卖开": 1, "卖平": -1,
                                                                     "买开": 1, "买平": -1,
                                                                     "买平今": -1, }

                                                    )

    person_holder, merged_summary_dict, contract_summary_dict = PR.group_by_summary(
        info_dict, return_data=True, store_2_excel=True)

    output_config = {'汇总': ['wj', 'gr', 'll'],
                     '期货多头': ['wj', 'gr'],
                     '期货对冲': ['IM'],
                     '分品种': 'ALL',
                     }

    person_by_year_summary, person_cum_sub, commodity_cum_sub, holding_contracts_summary_merged = PR.output_v2(info_dict, lastdel_multi, output_config)

    print(1)
    pass
