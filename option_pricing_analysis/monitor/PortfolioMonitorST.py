# coding=utf-8
import datetime
import functools
import io
import json
import os
import warnings
from glob import glob

import pandas as pd
import plotly.graph_objs as go
import requests
import streamlit as st
import yaml

warnings.filterwarnings('ignore')
import time
from CodersWheel.QuickTool.timer import timer
from fastapi.responses import Response
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决‘-’bug

st.set_page_config(
    page_title="期权监控平台",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={})


class APITools(object):
    @classmethod
    def post(cls, url, data: pd.DataFrame):

        file_obj = json.dumps({'Data': data.to_dict('records')})

        headers = {'Content-Type': 'application/json'}

        x = requests.post(url, headers=headers, data=file_obj)

        return x.text

    @classmethod
    def query(cls, url: str):
        x = requests.get(url)
        dat = cls.parse_df(x.content)
        return dat

    @staticmethod
    def query_decorator(func):
        def _deco(*args, **kwargs):
            url: str = func(*args, **kwargs)

            return APITools.query(url)

        return _deco

    @staticmethod
    def lru_cache(timeout: int, *lru_args, **lru_kwargs):
        def wrapper_cache(func):
            func = functools.lru_cache(*lru_args, **lru_kwargs)(func)
            func.delta = timeout
            func.expiration = time.monotonic() + func.delta

            @functools.wraps(func)
            def wrapped_func(*args, **kwargs):
                if time.monotonic() >= func.expiration:
                    func.cache_clear()
                    func.expiration = time.monotonic() + func.delta
                return func(*args, **kwargs)

            wrapped_func.cache_info = func.cache_info
            wrapped_func.cache_clear = func.cache_clear
            return wrapped_func

        return wrapper_cache

    @staticmethod
    def parse_df(content):
        # 将文件内容转换为 BytesIO 对象
        buffer = io.BytesIO(content)
        return pd.read_parquet(buffer)

    @staticmethod
    def send_df(df):
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        file_obj = buffer.getvalue()
        return file_obj

    @staticmethod
    def send_df_decrotor(func):
        def _deco(*args, **kwargs):

            nav = func(*args, **kwargs)
            if nav.empty:
                return None
            else:
                parquet_data = APITools.send_df(nav)
                return Response(content=parquet_data)

        return _deco


class QuoteHolder(object):
    @staticmethod
    @timer
    @APITools.query_decorator
    def get_op_quote_via_cffex(symbol='mo', end_month="ALL", base_host_port='47.104.186.157:3100'):
        url = f'http://{base_host_port}/ak/op/{symbol}/{end_month}'
        # 获取行情数据
        # t_trading_board = APITools.query(url)

        return url

    @staticmethod
    @timer
    @APITools.query_decorator
    def get_op_not_board_quote_via_cffex(symbol='mo', end_month="ALL", base_host_port='47.104.186.157:3100'):
        url = f'http://{base_host_port}/ak/op_not_board/{symbol}/{end_month}'
        # 获取行情数据
        # t_trading_board = query(url)

        return url

    @staticmethod
    @timer
    # @APITools.query_decorator
    def get_idx_minute_quote_via_ak(symbol='000852', base_host_port='47.104.186.157:3100'):
        url = f'http://{base_host_port}/ak/idx/{symbol}/1'
        dat = APITools.query(url)
        dat['代码'] = symbol

        return dat


class PositionHolder(object):

    @staticmethod
    def code_type_detect(code, pattern_rule_dict={'Future': r'^[A-Z]{2}\d{4}\.\w{3}$',
                                                  'Option': r'^[A-Z]+[0-9]+-[CP]-[0-9]+\.\w+$', }
                         ):
        """
        Detects the type of a given financial code based on the provided pattern rules.

        Args:
        code (str): The financial code to be detected.
        pattern_rule_dict (dict): The dictionary containing pattern rules for different code types.
                                Default value contains rules for 'Future' and 'Option' codes.

        Returns:
        str: The type of the financial code ('Future', 'Option', or 'Unknown').
        """
        import re
        for name, pattern_str in pattern_rule_dict.items():
            if re.compile(pattern_str).match(code):
                return name
            elif re.compile(pattern_str.replace('+-', '+').replace(']-', ']')).match(code):
                return name
        else:
            return 'Unknown'

    @staticmethod
    def option_type_detect(code, pattern_rule_dict={'Option-Put': r'^[A-Z]+[0-9]+-[P]-[0-9]+\.\w+$',
                                                    'Option-Call': r'^[A-Z]+[0-9]+-[C]-[0-9]+\.\w+$', }
                           ):
        """
        Detects the type of a given financial code based on the provided pattern rules.

        Args:
        code (str): The financial code to be detected.
        pattern_rule_dict (dict): The dictionary containing pattern rules for different code types.
                                Default value contains rules for 'Future' and 'Option' codes.

        Returns:
        str: The type of the financial code ('Future', 'Option', or 'Unknown').
        """
        import re
        for name, pattern_str in pattern_rule_dict.items():
            if re.compile(pattern_str).match(code):
                return name
            elif re.compile(pattern_str.replace('+-', '+').replace(']-', ']')).match(code):
                return name
        else:
            return 'Unknown'

    @classmethod
    def get_current_position(cls, pos_file_name, base_host_port='47.104.186.157:3100'):
        # 假设的持仓数据

        url = f'http://{base_host_port}/holding/{pos_file_name}'
        positions_df = APITools.query(url)

        if positions_df.empty:
            return pd.DataFrame(columns=['Code', 'Quantity', 'Cost', 'CodeType'])
        else:

            h = []
            for contract, v in positions_df.groupby('Code'):
                code_type = cls.code_type_detect(contract)
                if code_type == 'Option':
                    code_type = cls.option_type_detect(contract)

                v['CodeType'] = code_type
                h.append(v)

            return pd.concat(h)

    @classmethod
    def post_current_position(cls, pos_file_name, pos_df, base_host_port='47.104.186.157:3100'):
        # 假设的持仓数据

        url = f'http://{base_host_port}/holding/{pos_file_name}'
        res = APITools.post(url, pos_df)

        # if positions_df.empty:
        #     return pd.DataFrame(columns=['Code', 'Quantity', 'Cost', 'CodeType'])
        # else:
        #     h = []
        #     for contract, v in positions_df.groupby('Code'):
        #         code_type = cls.code_type_detect(contract)
        #         if code_type == 'Option':
        #             code_type = cls.option_type_detect(contract)
        #
        #         v['CodeType'] = code_type
        #         h.append(v)

        return res


class Calculator(object):
    @staticmethod
    def calculate_greeks(edited_pos):
        # 这里应该是计算Greeks的逻辑
        return pd.DataFrame([{'Delta': 0.5, 'Gamma': 0.05, 'Theta': -0.05, 'Vega': 0.1}])


# @st.experimental_fragment()
def draw_line(st, nav_df, name, fill=None, mode='lines', color='#ff0000', title=''):
    # 创建图形
    fig = go.Figure()
    # 添加线条
    fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[name].dropna(), fill=fill, mode='lines', line=dict(color=color)))

    fig.update_layout(
        title_text=title,  # 设置图表标题
    )
    # 调整坐标轴范围
    #     fig.update_xaxes(range=[0, 6])  # 设置x轴范围
    fig.update_yaxes(range=[nav_df[name].min() - 0.1, nav_df[name].max() + 0.1])  # 设置y轴范围
    # 启用自动缩放
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)
    # 在 Streamlit 中显示图表
    st.plotly_chart(fig, autoscale=True)


class Show(object):
    @staticmethod
    def create_side_settings(st, config_path='./'):
        with st.sidebar:
            st.write('日度期权监控-Settings')

            # 参数文件

            all_configs = glob(os.path.join(config_path, 'op_config*.yml'))
            all_config_file_dict = dict(zip(map(lambda x: os.path.split(x)[-1], all_configs), all_configs))

            file_name = st.selectbox("行情API接口参数文件", all_config_file_dict.keys(), index=0, )

            full_path = all_config_file_dict.get(file_name)

            with open(full_path, 'r', encoding="utf-8") as f:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)
            st.markdown('参数细节')
            st.write(config)

            return config

    @staticmethod
    def rename_df(call, direct='看涨'):

        cols = {'position': f'{direct}持仓量',
                'volume': f'{direct}成交量',
                'lastprice': f'{direct}最新成交价',
                'updown': f'{direct}涨跌',
                'pct': f'{direct}涨跌幅',
                'bprice': f'{direct}买一价格',
                'bamount': f'{direct}买一手数',
                'sprice': f'{direct}卖一价格',
                'samount': f'{direct}卖一手数',

                'k': '行权价', 'instrument': f'{direct}合约'}
        if direct == '看涨':
            col_order = ['instrument', 'updown', 'pct', 'volume', 'position', 'lastprice', 'end_month', 'k']
        else:
            col_order = ['k', 'end_month', 'lastprice', 'position', 'volume', 'pct', 'updown', 'instrument', ]
        return call[col_order].rename(columns=cols)

    @classmethod
    @st.experimental_fragment(run_every="3s")
    def market_part(cls, st, config):
        st.markdown('## 行情显示')

        c11, c12 = st.columns([2, 1])

        idx_minute = QuoteHolder.get_idx_minute_quote_via_ak(base_host_port=config['quote_ak_api']).set_index('时间')
        idx_minute.index = pd.to_datetime(idx_minute.index)
        # show today only
        mask = idx_minute.index >= datetime.datetime.today().strftime('%Y-%m-%d')
        idx_quote = idx_minute[mask]

        draw_line(c12, idx_quote.drop_duplicates(), '收盘')

        # ----------------
        t_trading_board = QuoteHolder.get_op_quote_via_cffex(symbol='mo', end_month="ALL",
                                                             base_host_port=config['quote_ak_api'])

        # call = merged_op_quote[merged_op_quote['direct'] == 'C']
        # renamed_call = cls.rename_df(call, direct='看涨')
        # put = merged_op_quote[merged_op_quote['direct'] == 'P']
        # renamed_put = cls.rename_df(put, direct='看跌')
        # t_trading_board = pd.merge(renamed_call, renamed_put, left_on=['行权价', 'end_month'],
        #                            right_on=['行权价', 'end_month'])
        # st.write(t_trading_board.columns.tolist())

        ym_idx = t_trading_board['看涨合约'].apply(lambda x: x[2:6])

        ym_list = ym_idx.unique().tolist()

        for ym, tab in zip(ym_list, c11.tabs(ym_list)):
            with tab:
                ym_mask = ym_idx == ym
                st.dataframe(t_trading_board[ym_mask], use_container_width=True, hide_index=True)

        return idx_quote, t_trading_board

    @staticmethod
    @st.experimental_fragment(run_every="3s")
    def holding_part(st, config, t_trading_board):

        ## 持仓显示

        c21, c22 = st.columns([2, 1])

        c21.markdown('## 持仓显示')

        pos_file_path = config['pos_file_path']

        # Initialization
        if 'Holding' not in st.session_state:
            pos = PositionHolder.get_current_position(pos_file_path)
            pos['Cost'] = pos['Cost'].astype(float)
            st.session_state['Holding'] = pos
        else:
            pos = st.session_state['Holding']

        call_cols = {"看涨合约": 'Code',
                     # "看涨涨跌",
                     "看涨涨跌幅": '涨跌幅',
                     # "看涨成交量",
                     # "看涨持仓量",
                     "看涨最新成交价": "最新成交价",
                     '行权价': '行权价'}

        put_cols = {"看跌合约": 'Code',
                    # "看跌涨跌",
                    "看跌涨跌幅": '涨跌幅',
                    # "看跌成交量",
                    # "看跌持仓量",
                    "看跌最新成交价": "最新成交价",
                    '行权价': '行权价'}

        call_df = t_trading_board[call_cols.keys()].rename(columns=call_cols)
        call_df['Code'] = call_df['Code'].apply(lambda x: x if x.upper().endswith('.CFE') else x.upper() + '.CFE')

        put_df = t_trading_board[put_cols.keys()].rename(columns=put_cols)
        put_df['Code'] = put_df['Code'].apply(
            lambda x: x if x.upper().endswith('.CFE') else x.upper() + '.CFE')

        h = []

        if pos.empty:
            st.write('pos empty')
            pos_df = pd.DataFrame([[put_df['Code'].head(1).values[0], 0, 0, 'Option-Put']],
                                  columns=['Code', 'Quantity', 'Cost', 'CodeType'])
        else:
            st.write('pos not empty')
            pos_df = pos

        for code_type, d in pos_df.groupby('CodeType'):
            if code_type == 'Option-Put':
                d_sub = pd.merge(d, put_df, left_on=['Code'], right_on=['Code'], how='left')
            elif code_type == 'Option-Call':
                d_sub = pd.merge(d, call_df, left_on=['Code'], right_on=['Code'], how='left')
            else:
                # todo 期货行情！！！

                pass
            h.append(d_sub)
        pos_df2 = pd.concat(h)

        # pos_df2.info()

        edited_pos = c21.data_editor(pos_df2, use_container_width=True, hide_index=True,
                                     column_order=['Code', 'Quantity', 'Cost', '最新成交价', '行权价', '涨跌幅', ],
                                     num_rows="dynamic")

        st.session_state['Holding'] = edited_pos[['Code', 'Quantity', 'Cost', 'CodeType']]

        edited_pos['time'] = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        stored_cols = ['Code', 'Quantity', 'Cost', 'time']
        st.write('edited_pos')
        st.write(edited_pos[stored_cols])

        PositionHolder.post_current_position(pos_file_path, edited_pos[stored_cols],
                                             base_host_port='47.104.186.157:3100')

        #
        st.write('Stored')
        pos2 = PositionHolder.get_current_position(pos_file_path)
        st.write(pos2)

        # with sqlite3.connect(pos_file_path) as conn:
        #     edited_pos[stored_cols].to_sql('Holding', conn, if_exists='append', index=False)

        # st.write(pos2)

        # c21.dataframe(pos)

        greeks = Calculator.calculate_greeks(edited_pos)
        c22.markdown('## Greeks')  ## greek显示
        c22.dataframe(greeks)

        return pos, greeks

    @classmethod
    def main(cls, st):
        config = cls.create_side_settings(st, config_path='./')
        # 行情
        idx_quote, t_trading_board = cls.market_part(st, config)

        pos, greeks = cls.holding_part(st, config, t_trading_board)

        ## 推荐组合显示
        st.markdown('## 推荐组合')
        # 暂时不支持


if __name__ == '__main__':
    # currt_hour = datetime.datetime.today().hour
    # trading_period = (currt_hour >= 9 and currt_hour <= 11) or (currt_hour >= 13 and currt_hour <= 15)
    # use_cache = ~trading_period

    col1, col2 = st.columns(2)
    col1.title('期权实时监控')

    # if trading_period:
    #     # 5000毫秒
    #     ticktime = datetime.datetime.now()
    #     #         tradedate = rtm_data['tradedate'].unique().tolist()[0]
    #     col2.write(f"更新时间：{ticktime}")
    #     count = st_autorefresh(interval=5000*2, limit=10000, key="fizzbuzzcounter")
    #
    # else:
    #     count = 0

    Show.main(st)

    pass
