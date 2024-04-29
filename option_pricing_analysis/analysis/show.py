# coding=utf-8
import os
import re
from glob import glob

import pandas as pd
import streamlit as st


def scan_summary_file_path(output_path='./', version='v3'):
    if output_path is None:
        output_path = './'

    output_name_format = f'日度衍生品交易收益率统计及汇总@*{version}.xlsx'

    _path = os.path.join(output_path, output_name_format)
    options = glob(_path)
    print(options, output_path)
    new = max(options)

    return new, options


def get_dt_from_file_name(path):
    filename = os.path.split(path)[-1]
    date_match = re.search(r'\d{8}', filename)

    if date_match:
        extracted_date = date_match.group()
        # print("提取的日期是:", extracted_date)
        return extracted_date
    else:
        # print("没有找到匹配的日期。")
        raise ValueError('没有找到匹配的日期')


def load_excel(path, multi_target_sheets=['holding_summary_merged_sorted']):
    with pd.ExcelFile(path) as f:
        sheet_names = f.sheet_names
    for sheet_name in sheet_names:
        if sheet_name in multi_target_sheets:
            index_col = [0, 1]
        else:
            index_col = None

        yield sheet_name, st.cache_data(pd.read_excel)(path, sheet_name=sheet_name, index_col=index_col)


class MainBody(object):
    @st.experimental_fragment()
    @staticmethod
    def today_summary(st, data_dict: dict, year=2024):
        col1, col2, col3 = st.columns(3)
        # 当日损益
        col1.metric("当日损益", data_dict['当日损益'], data_dict['当日损益'], delta_color='inverse')
        # 累计盈亏
        col2.metric("累计盈亏", data_dict['累计盈亏'], data_dict['当日损益'], delta_color='inverse')
        # 24年盈亏
        col3.metric("累计收益率", "{:.2%}".format(data_dict['累计收益率']),"None",  delta_color='inverse')

    @st.experimental_fragment()
    @staticmethod
    def by_person_by_year(st, person_by_year_summary):
        # 分人分年度统计
        st.header('分人分年度汇总统计')
        st.dataframe(person_by_year_summary, width=None, height=None, use_container_width=True)

    @st.experimental_fragment()
    @staticmethod
    def by_person_summary(st, person_cum_sub):
        # 分人分年度统计
        st.header('分交易员汇总统计绩效')
        st.dataframe(person_cum_sub, width=None, height=None, use_container_width=True)

    @st.experimental_fragment()
    @staticmethod
    def holding_summary(st, holding_summary_merged_sorted):
        # 分人分年度统计
        st.header('分当前持仓汇总统计绩效')
        st.dataframe(holding_summary_merged_sorted, width=None, height=None, use_container_width=True)

    @st.experimental_fragment()
    @staticmethod
    def commodity_summary(st, commodity_cum_sub):
        # 分人分年度统计
        st.header('分合约汇总统计绩效')
        st.dataframe(commodity_cum_sub, width=None, height=None, use_container_width=True)

    @st.experimental_fragment()
    @staticmethod
    def single_commodity_show(st, commodities, data_dict):
        # 分人分年度统计
        st.header('品种统计绩效')
        comm_elem = st.selectbox(
            "选择品种",
            sorted(commodities),
            index=0,
            placeholder="Select commodity...", )

        comm_elem_df = data_dict[comm_elem + '多空输出']
        st.dataframe(comm_elem_df, use_container_width=True)


if __name__ == '__main__':
    newest_path, store_path_list = scan_summary_file_path(output_path='./', version='v3')
    #     dt = get_dt_from_file_name(newest_path)
    avaiable_dt = list(map(get_dt_from_file_name, store_path_list))

    st.set_page_config(layout="wide", )

    # 创建一个标题
    st.title('日度衍生品交易收益率统计及汇总-Beta')

    with st.sidebar:
        st.write(
            '日度衍生品交易收益率统计及汇总-Settings')
        option = st.selectbox(
            "选择统计日期",
            avaiable_dt,
            index=0,
            placeholder="Select datetime method...",
        )
    #     st.write(avaiable_dt.index(option))
    selected_path = store_path_list[avaiable_dt.index(option)]
    #     st.write(avaiable_dt.index(option))
    if os.path.exists(selected_path):
        data_dict = dict(load_excel(selected_path, multi_target_sheets=['holding_summary_merged_sorted']))
    else:
        raise ValueError(f'data have not create：{newest_path}')

    commodities = [k[:2] for k in data_dict.keys() if '多空输出' in k]

    # 概览

    person_by_year_summary = data_dict['person_by_year_summary'].set_index('统计维度')
    person_cum_sub = data_dict['person_cum_sub'].set_index('交易员')
    commodity_cum_sub = data_dict['commodity_cum_sub'].set_index('品种')
    holding_summary_merged_sorted = data_dict['holding_summary_merged_sorted']  # .set_index(['交易员','统计维度'])

    # 缺一个汇总成本的
    pnl, cost = [], []
    for person, df in data_dict.items():
        if person in person_by_year_summary.columns:
            pnl.append(df['累计净损益(右轴)'])
            cost.append(df['累计开仓成本'])

    pnl2 = pd.concat(pnl, axis=1).sum(axis=1)
    cost2 = pd.concat(cost, axis=1).sum(axis=1)
    cum_ret2 = (pnl2 / cost2).to_frame('累计收益率')

    cum_pnl = person_by_year_summary.loc['累计', '累计盈亏']
    cum_pnl_str = (cum_pnl / 10000).round(2).astype(str) + '万元'

    conf_fdict = {'累计盈亏': cum_pnl_str,
                  '当日损益': person_by_year_summary.loc['当日损益', '累计盈亏'].round(2).astype(str) + '元',
                  '累计收益率': cum_ret2.tail(1).values[0]
                  }

    MainBody.today_summary(st, conf_fdict)

    # 分人分年度统计
    MainBody.by_person_by_year(st, person_by_year_summary)

    # 分交易员汇总统计绩效
    MainBody.by_person_summary(st, person_cum_sub)

    # 分当前持仓汇总统计绩效
    MainBody.holding_summary(st, holding_summary_merged_sorted)
    #     st.header('分当前持仓汇总统计绩效')
    #     st.dataframe(holding_summary_merged_sorted,use_container_width=True)
    # 分合约汇总统计绩效
    MainBody.commodity_summary(st, commodity_cum_sub)
    #     st.header('分合约汇总统计绩效')
    #     st.dataframe(commodity_cum_sub.style,use_container_width=True)

    # 品种统计绩效
    MainBody.single_commodity_show(st, commodities, data_dict)

    #     st.write(commodities)
    #     comm_elem = st.selectbox(
    #            "选择品种",
    #            commodities,
    #            index=0,
    #            placeholder="Select commodity...",)

    #     comm_elem_df = data_dict[comm_elem+'多空输出']
    #     st.dataframe(comm_elem_df,use_container_width=True)

    #     altair_chart = (alt.Chart(comm_elem_df))

    #     st.altair_chart(altair_chart, use_container_width=False, theme="streamlit")

    pass
