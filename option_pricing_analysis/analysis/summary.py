import datetime
import os

import pandas as pd

from option_pricing_analysis.analysis.option_analysis_monitor import WindHelper, ReportAnalyst


def chunk(obj, chunks=2000):
    if hasattr(obj, '__len__'):
        length = len(obj)
        for i in range(0, length, chunks):
            yield obj[i:i + chunks]


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


def cal_periods_result_v2(df, pnl_col, nv_col):
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

    return periods_results


def process_ph(person_holder):
    all_person_dfs = []
    target_col = ['累计收益', '累计收益率', '当日收益', '当日收益率',
                  '近一周收益', '近一周收益率', '近一月收益', '近一月收益率']

    base_cols = ['累计净值', '累计净损益(右轴)', ]

    for prsn, df in person_holder.items():

        filtered_col = base_cols + sorted(df.columns.tolist()[8:])
        # filtered_col = [item for item in df.columns if "损益" in item or "净值" in item]
        person_dict = {}  # pd.DataFrame(columns=target_col)
        for nv_col, pnl_col in chunk(filtered_col, 2):
            metrics_info = cal_periods_result_v2(df, pnl_col, nv_col)
            name = nv_col[:2]
            person_dict[(prsn, name)] = metrics_info
            # for metric in metrics_info:
            #     if metric in person_df.columns:
            # person_df.at[(pnl_col, nv_col), metric] = metrics_info[metric]
        all_person_dfs.append(pd.DataFrame(person_dict).T[target_col])
    full_summary_df = pd.concat(all_person_dfs, axis=0)

    # for key in person_holder.keys():
    #     if '累计' in full_summary_df.index:
    #         full_summary_df = full_summary_df.rename(index={'累计': key})

    return full_summary_df


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


# def check_contract_condition_executed(df):
#     return df[df.tail(1) != 0].tail(1).dropna(axis=1)


def check_contract_condition_closed_value(df):
    return df[df.diff(1).tail(1) != 0].tail(1).dropna(axis=1)


def check_contract_condition_all_zero(df):
    return df[df.isnull().all()].tail(1).dropna(axis=1)


def check_contract_condition_resid_shares(df):
    return df[df.tail(1) != 0].tail(1).dropna(axis=1)


def _map_dict_get(keys, dictitems):
    for key in keys:
        val = dictitems.get(key, None)
        if isinstance(val, pd.DataFrame):
            return val
    else:
        raise KeyError(f"{keys} not in dict!")


1


def _map_dict_get_funcs(keys, dictitems, func):
    return func(_map_dict_get(keys, dictitems))


def check_contract_conditions_v2(sub_dict, reduce=False):
    holding_contracts_cols = []

    cum_closed_value = ['衍生品多头累计平仓价值', '衍生品空头累计平仓价值']
    cum_pnl = ['衍生品多头累计净损益', '衍生品空头累计净损益']
    share = ['衍生品多头剩余份数', '衍生品空头剩余份数']
    executed = ['衍生品多头累计行权收益', '衍生品空头累计行权收益']

    closed_value_mask = _map_dict_get_funcs(cum_closed_value, sub_dict, check_contract_condition_closed_value)
    holding_contracts_cols.extend(closed_value_mask.columns.tolist())

    cum_pnl_mask = _map_dict_get_funcs(cum_pnl, sub_dict, check_contract_condition_closed_value)
    holding_contracts_cols.extend(cum_pnl_mask.columns.tolist())

    share_mask = _map_dict_get_funcs(share, sub_dict, check_contract_condition_resid_shares)
    holding_contracts_cols.extend(share_mask.columns.tolist())

    executed_mask = _map_dict_get_funcs(executed, sub_dict, check_contract_condition_resid_shares)
    executed_code = set(executed_mask.columns.tolist())
    # holding_contracts_cols.extend(share_mask.columns.tolist())

    # share_mask = _map_dict_get_funcs(share, sub_dict, check_contract_condition_resid_shares)
    holding_contracts_cols = set(holding_contracts_cols) - executed_code if reduce else holding_contracts_cols

    return set(holding_contracts_cols)


def cal_today_result(holding_value_df, cum_closed_value_df,
                     cum_pnl_df, cum_cost_df, share_df, checked_cols, lastdel_multi, ls):
    # holding_value_df, cum_closed_value_df,
    # cum_pnl_df, cum_cost_df, share_df

    today_result = {}
    for col in checked_cols:
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

        today_result[col] = {
            '持仓方向': ls,
            '持仓手数': num_contracts,
            '当日损益': today_profit_loss,
            '当日累计收益率涨跌幅': today_total_return_change,
            '持仓名义市值': nominal_value,
            '平仓价值': closing_value,
            '累计净损益': cum_profit_loss,
            '累计净损益（%）': cum_profit_loss_perc
        }
    indicators_today = pd.DataFrame(today_result).T.sort_index()
    mask = (indicators_today['持仓名义市值'] != 0) | (indicators_today['平仓价值'] != 0)
    return indicators_today[mask]


def c2p_p2c(p2c):
    for k, vs in p2c.items():
        for v in vs:
            yield v, k


def holding_contract_info(info_dict, lastdel_multi, c2p):
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
        holding_contracts_cols = check_contract_conditions_v2(sub_dict)

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

        indicators_df = cal_today_result(holding_value_df, cum_closed_value_df,
                                         cum_pnl_df, cum_cost_df, share_df,
                                         holding_contracts_cols, lastdel_multi, ls)

        indicators_df.index = list(map(lambda x: (c2p.get(x[:2], None), x), indicators_df.index))
        holding_summary_info.append(indicators_df)

    return holding_summary_info


##输出
def create_summary_output_file_path(output_path='./'):
    output_path = './' if output_path is None else output_path
    today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')
    output_name_format = f'输出指标统计@{today_str}.xlsx'
    _path = os.path.join(output_path, output_name_format)
    return _path


def output_summary(person_holder, info_dict, lastdel_multi, base_store_path=None, return_data=False):
    person_contract_summary = process_ph(person_holder)

    p2c = {pp: set(map(lambda x: x[:2], df.columns.tolist()[8:])) for pp, df in person_holder.items()}

    c2p = dict(c2p_p2c(p2c))

    holding_contracts_summary = holding_contract_info(info_dict, lastdel_multi, c2p)

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


if __name__ == '__main__':
    import yaml

    path = os.path.join(os.path.split(__file__)[0], 'config.yml')

    with open(path, 'r', encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')

    wh = WindHelper()

    PR = ReportAnalyst(
        report_file_path=config['report_file_path'],
        contract_2_person_rule={'MO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'HO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'IO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'IH\d{4}.CFE': 'wj',
                                'IF\d{4}.CFE': 'wj',
                                'IM\d{4}.CFE': 'll',
                                'AG\d{4}.SHF': 'gr',
                                'AU\d{4}.SHF': 'wj',
                                'CU\d{4}.SHF': 'wj',
                                'CU\d{4}[CP]\d{5}.SHF': 'wj',
                                'AL\d{4}.SHF': 'gr'}

    )

    contracts = PR.reduced_contracts()

    quote = PR.get_quote_and_info(contracts, wh, start_with='2022-09-04')

    lastdel_multi = PR.get_info_last_delivery_multi(contracts, wh)

    info_dict = PR.parse_transactions_with_quote_v2(quote, lastdel_multi,
                                                    trade_type_mark={"卖开": 1, "卖平": -1,
                                                                     "买开": 1, "买平": -1,
                                                                     "买平今": -1, }

                                                    )

    person_holder, person_ls_summary_dict, merged_summary_dict, contract_summary_dict, info_dict = PR.group_by_summary(
        info_dict, return_data=True, store_2_excel=False)

    # PR.summary_person_info(person_summary_dict, merged_summary_dict, info_dict, lastdel_multi, quote, )

    person_summary, holding_summary = output_summary(person_holder, info_dict, lastdel_multi, return_data=True)

    print(1)
    pass
