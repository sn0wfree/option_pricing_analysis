# coding=utf-8
import copy
import datetime
import os
import re
from collections import Callable, ChainMap, deque
from collections.abc import Iterable
from glob import glob

import numpy as np
import pandas as pd
import yaml
from CodersWheel.QuickTool.detect_file_path import detect_file_full_path
from CodersWheel.QuickTool.file_cache import file_cache


class WindHelper(object):
    """
        A helper class for interacting with the Wind financial database.
    """

    def __init__(self):
        from WindPy import w
        w.start()
        self.w = w

    @file_cache(enable_cache=True, granularity='d')
    def get_data(self, code, start, end, fields):
        """
        Generic method to fetch data from the Wind database.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :param fields: The fields to retrieve (e.g., 'close', 'high').
        :return: DataFrame with the requested data.
        """
        error, data_frame = self.w.wsd(code, fields, start, end, "", usedf=True)
        if error != 0:
            raise ValueError(f"Wind API error: {error}")
        return data_frame

    def get_close_prices(self, code, start, end):
        """
        Retrieve closing prices for a given code and date range.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :return: DataFrame with closing prices.
        """
        return self.get_data(code, start, end, 'close')

    def wind_wsd_high(self, code: str, start: str, end: str):
        return self.get_data(code, start, end, "high")

    def wind_wsd_low(self, code: str, start: str, end: str):
        return self.get_data(code, start, end, "low")

    def wind_wsd_open(self, code: str, start: str, end: str):
        """

        :param code:
        :param start:
        :param end:
        :return:
        """
        return self.get_data(code, start, end, "open")

    def wind_wsd_volume(self, code: str, start: str, end: str):
        """

        :param code:
        :param start:
        :param end:
        :return:
        """
        return self.get_data(code, start, end, "volume")

    def wind_wsd_quote(self, code: str, start: str, end: str, required_cols=('open', 'high', 'low', 'close', 'volume')):
        """

        :param code:
        :param start:
        :param end:
        :param required_cols:
        :return:
        """

        res_generator = (self.get_data(code, start, end, col) for col in required_cols)
        df = pd.concat(res_generator, axis=1)
        df['symbol'] = code

        return df

    def wind_wsd_quote_reduce(self, code: str, start: str, end: str, required_cols=('close', 'volume')):
        df = self.wind_wsd_quote(code, start, end, required_cols=required_cols)
        return df

    @file_cache(enable_cache=True, granularity='d')
    def get_future_info_last_delivery_date_underlying(self, code_list=None,
                                                      date_fields=['EXE_ENDDATE', 'LASTDELIVERY_DATE', 'FTDATE_NEW',
                                                                   'STARTDATE'],
                                                      multiplier_fields=['CONTRACTMULTIPLIER'],
                                                      underlying_code=['UNDERLYINGWINDCODE'],
                                                      futures_margin=['MARGIN']
                                                      ):
        """
        Retrieve future information including last delivery dates and contract multipliers.

        :param code_list: List of future contract codes.
        :return: DataFrame with future information.
        """
        if code_list is None:
            code_list = ["IF2312.CFE"]

        cols_str = ",".join(date_fields + multiplier_fields + underlying_code + futures_margin).lower()
        # "ftdate_new,startdate,lastdelivery_date,exe_enddate,contractmultiplier"

        err, last_deliv_and_multi = self.w.wss(','.join(code_list), cols_str, usedf=True)

        if err != 0:
            raise ValueError(f"Wind API error: {err}")

        for field in date_fields:
            last_deliv_and_multi[field] = last_deliv_and_multi[field].replace('1899-12-30', None)
            # last_deliv_and_multi[field] = pd.to_datetime(last_deliv_and_multi[field], errors='coerce')

        last_deliv_and_multi['EXE_DATE'] = last_deliv_and_multi['EXE_ENDDATE'].combine_first(
            last_deliv_and_multi['LASTDELIVERY_DATE'])
        last_deliv_and_multi['START_DATE'] = last_deliv_and_multi['STARTDATE'].combine_first(
            last_deliv_and_multi['FTDATE_NEW'])

        return last_deliv_and_multi


class DequeMatcher(deque):
    def __init__(self, *args, k=None,
                 contract_col='contract_code',
                 share_col='手数',
                 price_col='成交均价',
                 trade_type='trade_type',
                 dt_col='日期', **kwargs):
        super(DequeMatcher, self).__init__(*args, **kwargs)
        if k is not None:
            if isinstance(k, Iterable):
                self.extend(list(k))
            else:
                pass
        self.__share_col__ = share_col
        self.__price_col__ = price_col
        self.__contract_col__ = contract_col
        self.__dt_col__ = dt_col
        self.__trade_type__ = trade_type

    # @property
    def get_total_shares(self):
        total = 0
        for s in self:
            total += s[self.__share_col__]
        return total

    # @property
    def get_avg_price(self):
        total_share = self.get_total_shares()
        if total_share == 0:
            return None
        amt = np.sum([s[self.__price_col__] * s[self.__share_col__] for s in self])

        return amt / total_share

    @classmethod
    def multi_contract_recus_match_v2(cls, transcations_multi,

                                      contract_col='contract_code',
                                      share_col='手数',
                                      price_col='成交均价',
                                      trade_type='trade_type',
                                      dt_col='报单日期', method='FIFO'
                                      ):

        for contract, sub_transca in transcations_multi.sort_values(dt_col).groupby(contract_col):
            if contract == 'MO2405-C-5200.CFE':
                print(1)
            # long
            long_mask = sub_transca['买卖开平'].isin(['买开', '卖平'])
            res = cls.single_contract_recurs_match_v2(sub_transca[long_mask], contract_col=contract_col,
                                                      share_col=share_col,
                                                      price_col=price_col,
                                                      trade_type=trade_type,
                                                      dt_col=dt_col, method=method)

            yield contract, '多头', res

            # long
            short_mask = sub_transca['买卖开平'].isin(['卖开', '买平'])
            res = cls.single_contract_recurs_match_v2(sub_transca[short_mask], contract_col=contract_col,
                                                      share_col=share_col,
                                                      price_col=price_col,
                                                      trade_type=trade_type,
                                                      dt_col=dt_col, method=method)

            yield contract, '空头', res

    @classmethod
    def single_contract_recurs_match_v2(cls, transcation_1contract,
                                        contract_col='contract_code',
                                        share_col='手数',
                                        price_col='成交均价',
                                        trade_type='trade_type',
                                        dt_col='日期', method='FIFO'

                                        ):

        buy_records = transcation_1contract[transcation_1contract[trade_type] == 1]
        buy_deque = DequeMatcher(k=buy_records.to_dict('records'),
                                 contract_col=contract_col,
                                 share_col=share_col,
                                 price_col=price_col,
                                 trade_type=trade_type,
                                 dt_col=dt_col, )

        sell_records = transcation_1contract[transcation_1contract[trade_type] == -1]
        sell_deque = DequeMatcher(k=sell_records.to_dict('records'), contract_col=contract_col,
                                  share_col=share_col,
                                  price_col=price_col,
                                  trade_type=trade_type,
                                  dt_col=dt_col, )
        h = []
        res = cls.recur_match_core(buy_deque, sell_deque, h,
                                   contract_col=contract_col,
                                   share_col=share_col,
                                   price_col=price_col,
                                   trade_type=trade_type,
                                   dt_col=dt_col, method=method)
        return res

    @staticmethod
    def recur_match_core(buy_deque, sell_deque, h,
                         contract_col='contract_code',
                         share_col='手数',
                         price_col='成交均价',
                         trade_type='trade_type',
                         dt_col='日期', method='FIFO'):

        if method == 'FIFO':
            popfunc = lambda x: x.popleft()
            appendfunc = lambda deque, x: deque.appendleft(x)
        elif method == 'LIFO':
            popfunc = lambda x: x.pop()
            appendfunc = lambda deque, x: deque.append(x)
        else:
            raise ValueError('unknown method!')
        if len(sell_deque) == 0:  # 这是最后退出的节点
            return buy_deque.get_avg_price(), h
        else:
            buy_length = len(buy_deque)
            if buy_length == 0:
                raise ValueError('No apply data left!')

            buy_total_share = buy_deque.get_total_shares()
            sell_total_share = sell_deque.get_total_shares()

            if buy_total_share < sell_total_share:
                raise ValueError('buy side less than sell side!')

            elif np.round(buy_total_share) >= np.round(sell_total_share, 2):  # buy share 大于sell share

                buy1 = popfunc(buy_deque)  # 取一个
                sell1 = popfunc(sell_deque)  # 取一个

                buy_shares = buy1[share_col]
                sell_shares = sell1[share_col]

                if buy_shares == sell_shares:  # redeem 与 apply 相等
                    h.append((buy1, sell1))
                    return DequeMatcher.recur_match_core(buy_deque, sell_deque, h,
                                                         contract_col=contract_col,
                                                         share_col=share_col,
                                                         price_col=price_col,
                                                         trade_type=trade_type,
                                                         dt_col=dt_col)
                elif buy_shares > sell_shares:

                    remaining_num = buy_shares - sell_shares

                    buy_sub_1 = copy.deepcopy(buy1)
                    rd_buy_sub_1 = copy.deepcopy(buy1)

                    buy_sub_1[share_col] = remaining_num
                    rd_buy_sub_1[share_col] = sell_shares
                    # buy_deque.appendleft(buy_sub_1)  # 放回去

                    appendfunc(buy_deque, buy_sub_1)

                    h.append((rd_buy_sub_1, sell1))

                    return DequeMatcher.recur_match_core(buy_deque, sell_deque, h,
                                                         contract_col=contract_col,
                                                         share_col=share_col,
                                                         price_col=price_col,
                                                         trade_type=trade_type,
                                                         dt_col=dt_col)

                else:  # buy_shares < sell_shares

                    sell_sub_1 = copy.deepcopy(sell1)
                    rm_sell_sub_1 = copy.deepcopy(sell1)
                    # rd_buy_sub_1 = buy1.copy(deep=True)
                    rm_sell_sub_1[share_col] = buy_shares

                    remaining_num = sell_shares - buy_shares
                    sell_sub_1[share_col] = remaining_num

                    # sell_deque.appendleft(sell_sub_1)  # 放回去
                    h.append((buy1, rm_sell_sub_1))
                    appendfunc(sell_deque, sell_sub_1)

                    return DequeMatcher.recur_match_core(buy_deque, sell_deque, h,
                                                         contract_col=contract_col,
                                                         share_col=share_col,
                                                         price_col=price_col,
                                                         trade_type=trade_type,
                                                         dt_col=dt_col)


class Tools(object):
    @staticmethod
    def code_commodity_detect(code: str):
        return code[:2]

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
        # if code.startswith('CU') and code.endswith('00.SHF'):
        #     print(1)
        for name, pattern_str in pattern_rule_dict.items():
            if re.compile(pattern_str).match(code):
                return name
            elif re.compile(pattern_str.replace('+-', '+').replace(']-', ']')).match(code):
                return name
        else:
            return 'Unknown'

    @staticmethod
    def create_draw_from_opened_excel(f, data_length, target_sheet='多头输出'):
        new_wb = f.book
        ws = new_wb.sheetnames[target_sheet]
        #
        # data_length = summary_data.shape[0]

        # 创建一个日期格式对象
        date_format = new_wb.add_format({'num_format': 'yyyy-mm-dd'})

        # 设置列 A（包含日期）的格式为日期格式
        ws.set_column(f'A1:A{data_length + 1}', None, date_format)

        # 创建一个折线图
        line_chart1 = new_wb.add_chart({'type': 'line'})

        # 配置第一个系列数据（折线图），使用相同的类别
        # 成本
        line_chart1.add_series({
            'name': f'={target_sheet}!$C$1',
            'categories': f'={target_sheet}!$A$2:$A${data_length + 1}',
            'values': f'={target_sheet}!$C$2:$C${data_length + 1}',
            'line': {'color': '#4f80bd'},  # RGB 颜色值 (79, 128, 189)  蓝色
        })

        # 配置第二个系列数据（折线图），使用相同的类别
        # 价值
        line_chart1.add_series({
            'name': f'={target_sheet}!$G$1',
            'categories': f'={target_sheet}!$A$2:$A${data_length + 1}',
            'values': f'={target_sheet}!$G$2:$G${data_length + 1}',
            'line': {'color': '#b1ca7d'},  # RGB 颜色值 (177，202，125)  绿色
        })

        # 创建一个面积图
        area_chart = new_wb.add_chart({'type': 'area'})

        # 配置第一个系列数据（面积图）
        area_chart.add_series({
            'name': f'={target_sheet}!$F$1',
            'categories': f'={target_sheet}!$A$2:$A${data_length + 1}',
            'values': f'={target_sheet}!$F$2:$F${data_length + 1}',
            'fill': {'color': '#c0504d'},  # RGB 颜色值 (192，80，77)  红色
            'y2_axis': True,  # 将此系列绑定到次坐标轴
        })

        # 将折线图合并到面积图中
        line_chart1.combine(area_chart)
        line_chart1.set_legend({'position': 'top'})
        # line_chart1.set_y_axis({'num_format': '#,##0,," 百万"'})
        # line_chart1.set_y2_axis({'num_format': '#,##0,," 百万"'})

        # 设置 X 轴的标题
        # line_chart1.set_x_axis({'name': 'Month'})

        # 设置 Y 轴的标题
        line_chart1.set_y_axis({'name': '名义市值'})
        # 设置 Y2 轴的标题
        line_chart1.set_y2_axis({'name': '累计损益'})

        # 将图表插入到工作表中
        ws.insert_chart('E2', line_chart1, {'x_offset': 330, 'y_offset': 10, 'x_scale': 2, 'y_scale': 2})

        # 设置日期格式

        workbook = f.book
        worksheet = f.sheets[target_sheet]

        # 定义一个单元格格式：加粗并且字体为蓝色
        # cell_format = workbook.add_format({'bold': True, 'font_color': 'blue'})

        # 定义日期格式
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})

        # 应用格式到 A1 单元格
        # worksheet.write('A1', 'Modified Data', cell_format)

        # 也可以设置列的格式
        worksheet.set_column('A:A', 13, date_format)

        return f


class DerivativesProcessTools(object):
    @staticmethod
    def _cal_xxx_cost(transactions, derivatives_list, dt_list, dt_col='报单日期', contract_col='委托合约'):
        """

        :param transactions:
        :param derivatives_list:
        :param dt_list:
        :return:
        """
        open_matrix = transactions.pivot_table(index=dt_col, values='cost',
                                               columns=contract_col).reindex(index=dt_list)
        open_unit_matrix = transactions.pivot_table(index=dt_col, values='unit',
                                                    columns=contract_col).reindex(index=dt_list)

        cum_xxx_df = open_matrix.reindex(index=dt_list, columns=derivatives_list).fillna(0).cumsum()
        return cum_xxx_df, open_unit_matrix, open_matrix

    @classmethod
    def _cal_open_cost(cls, m1_buy2open_transactions, derivatives_list, dt_list, dt_col='报单日期',
                       contract_col='委托合约'):
        return cls._cal_xxx_cost(m1_buy2open_transactions, derivatives_list, dt_list, dt_col=dt_col,
                                 contract_col=contract_col)

    @classmethod
    def _cal_close_cost(cls, m1_sell2close_transactions, derivatives_list, dt_list, dt_col='报单日期',
                        contract_col='委托合约'):
        return cls._cal_xxx_cost(m1_sell2close_transactions, derivatives_list, dt_list, dt_col=dt_col,
                                 contract_col=contract_col)

    @staticmethod
    def _create_executed_profit(end_dt, derivatives, contract_type, holding_value, lastdel_multi, sub_quote, net_unit):
        """

        :param end_dt:
        :param derivatives:
        :param contract_type:
        :param holding_value:
        :param lastdel_multi:
        :param sub_quote:
        :param net_unit:
        :return:
        """
        # 期权需要行权收益，期货需要交割收益
        if holding_value.index.max() >= end_dt:  # 判断是否需要计算行权或者交割收益
            if contract_type == 'Option':  # 判断期权
                vlu_f_exe_prof = max(0, holding_value.loc[end_dt].values[0])

            elif contract_type == 'Future':  # 判断期货
                underlying_code = lastdel_multi.loc[derivatives, 'UNDERLYINGWINDCODE']
                if underlying_code is not None:
                    quote_close = sub_quote.loc[end_dt, [underlying_code]].values[0]
                    vlu_f_exe_prof = (net_unit.loc[end_dt] * quote_close).values[0]
                else:
                    vlu_f_exe_prof = max(0, holding_value.loc[end_dt].values[0])
            else:
                raise ValueError('unknown contract_type, only accept Option or Future')

            dt_list = sub_quote.index[sub_quote.index > end_dt]
            executed_profit = pd.DataFrame(index=dt_list)
            # 需要行权收益，期货没有行权收益

            # 取最后一期的数据最为行权最后的损益
            executed_profit[derivatives] = vlu_f_exe_prof

            # executed_profit = cls._create_executed_profit(contract_type, end_dt, derivatives,
            #                                               vlu_f_exe_prof, sub_quote.index)
        else:
            executed_profit = pd.DataFrame(columns=[derivatives])
        return executed_profit

    @classmethod
    def sub_parse_general_short(cls, end_dt, lastdel_multi, derivatives, sub_transactions, sub_quote, contract_type,
                                open_symbol='卖开', close_symbol='买平'):

        mask_close = sub_transactions['买卖开平'] == close_symbol
        mask_open = sub_transactions['买卖开平'] == open_symbol

        open_transactions = sub_transactions[mask_open]
        close_transactions = sub_transactions[mask_close]

        cum_cost_df, open_unit, buy2open = cls._cal_open_cost(open_transactions, [derivatives],
                                                              sub_quote.index)
        cum_buy_unit = open_unit.cumsum().ffill()

        # 计算份额状态
        if not close_transactions.empty:  # 如果存在卖出操作的话，计算卖出损益和剩余份数
            # 有平仓就会为负数

            cum_sold_value_df, sell2close_unit, sell2close = cls._cal_close_cost(close_transactions,
                                                                                 [derivatives],
                                                                                 sub_quote.index)

            cum_sellvalue = sell2close[sell2close.index <= end_dt].fillna(0).cumsum()
            cum_sellvalue_unit = sell2close_unit[sell2close_unit.index <= end_dt].fillna(0).cumsum()
            net_unit = (cum_buy_unit[derivatives] - cum_sellvalue_unit.fillna(0)[derivatives]).to_frame(derivatives)
        else:
            cum_sellvalue = pd.DataFrame(index=buy2open.index, columns=[derivatives])
            net_unit = cum_buy_unit[derivatives].to_frame(derivatives)
        try:
            holding_value = net_unit[derivatives] * sub_quote[derivatives]
        except Exception as e:
            raise e
        if not isinstance(holding_value, pd.DataFrame):
            holding_value = holding_value.to_frame(derivatives)

        cum_cost_df = buy2open.reindex(index=sub_quote.index, columns=[derivatives]).fillna(0).cumsum()[[derivatives]]
        # cum_sellvalue 设定为平仓价值，设定为负数，所以需要减（--）得正
        # todo 逻辑有点绕
        res_sub = holding_value - cum_cost_df - cum_sellvalue if not cum_sellvalue.dropna().empty else holding_value - cum_cost_df

        # 期权需要行权收益，期货需要交割收益
        executed_profit = cls._create_executed_profit(end_dt, derivatives, contract_type, holding_value,
                                                      lastdel_multi, sub_quote, net_unit)

        if open_symbol == '卖开':

            return executed_profit * -1, res_sub * -1, holding_value * -1, net_unit * -1, cum_sellvalue * -1, cum_cost_df * -1

        else:

            return executed_profit, res_sub, holding_value, net_unit, cum_sellvalue, cum_cost_df

    @classmethod
    def sub_parse_general_long(cls, end_dt, lastdel_multi, derivatives, sub_transactions, sub_quote, contract_type,
                               open_symbol='买开', close_symbol='卖平'):

        mask_close = sub_transactions['买卖开平'] == close_symbol
        mask_open = sub_transactions['买卖开平'] == open_symbol

        open_transactions = sub_transactions[mask_open]
        close_transactions = sub_transactions[mask_close]

        cum_cost_df, open_unit, open_amt = cls._cal_open_cost(open_transactions, [derivatives],
                                                              sub_quote.index)
        cum_buy_unit = open_unit.cumsum().ffill()

        # 计算份额状态
        if not close_transactions.empty:  # 如果存在卖出操作的话，计算卖出损益和剩余份数

            cum_sold_value_df, sell2close_unit, sell2close = cls._cal_close_cost(close_transactions,
                                                                                 [derivatives],
                                                                                 sub_quote.index)

            cum_sellvalue = sell2close[sell2close.index <= end_dt].fillna(0).cumsum()
            cum_sellvalue_unit = sell2close_unit[sell2close_unit.index <= end_dt].fillna(0).cumsum()
            net_unit = (cum_buy_unit[derivatives] - cum_sellvalue_unit.fillna(0)[derivatives]).to_frame(derivatives)
        else:
            cum_sellvalue = pd.DataFrame(index=open_amt.index, columns=[derivatives])
            net_unit = cum_buy_unit[derivatives].to_frame(derivatives)
        try:
            holding_value = net_unit[derivatives] * sub_quote[derivatives]
        except Exception as e:
            raise e
        if not isinstance(holding_value, pd.DataFrame):
            holding_value = holding_value.to_frame(derivatives)

        cum_cost_df = open_amt.reindex(index=sub_quote.index, columns=[derivatives]).fillna(0).cumsum()[[derivatives]]

        # cum_sellvalue 设定为平仓价值，设定为负数，所以需要减（--）得正
        res_sub = holding_value - cum_cost_df - cum_sellvalue if not cum_sellvalue.dropna().empty else holding_value - cum_cost_df

        # 期权需要行权收益，期货需要交割收益
        executed_profit = cls._create_executed_profit(end_dt, derivatives,
                                                      contract_type, holding_value,
                                                      lastdel_multi, sub_quote, net_unit)

        return executed_profit, res_sub, holding_value, net_unit, cum_sellvalue, cum_cost_df

    @classmethod
    def sub_parse_general(cls, end_dt, lastdel_multi, derivatives, sub_transactions, sub_quote, contract_type,
                          open_symbol='买开', close_symbol='卖平'):

        mask_close = sub_transactions['买卖开平'] == close_symbol
        mask_open = sub_transactions['买卖开平'] == open_symbol

        open_transactions = sub_transactions[mask_open]
        close_transactions = sub_transactions[mask_close]

        cum_cost_df, open_unit, buy2open = cls._cal_open_cost(open_transactions, [derivatives],
                                                              sub_quote.index)
        cum_buy_unit = open_unit.cumsum().ffill()

        # 计算份额状态
        if not close_transactions.empty:  # 如果存在卖出操作的话，计算卖出损益和剩余份数

            cum_sold_value_df, sell2close_unit, sell2close = cls._cal_close_cost(close_transactions,
                                                                                 [derivatives],
                                                                                 sub_quote.index)

            cum_sellvalue = sell2close[sell2close.index <= end_dt].fillna(0).cumsum()
            cum_sellvalue_unit = sell2close_unit[sell2close_unit.index <= end_dt].fillna(0).cumsum()
            net_unit = (cum_buy_unit[derivatives] - cum_sellvalue_unit.fillna(0)[derivatives]).to_frame(derivatives)
        else:
            cum_sellvalue = pd.DataFrame(index=buy2open.index, columns=[derivatives])
            net_unit = cum_buy_unit[derivatives].to_frame(derivatives)
        try:
            holding_value = net_unit[derivatives] * sub_quote[derivatives]
        except Exception as e:
            raise e
        if not isinstance(holding_value, pd.DataFrame):
            holding_value = holding_value.to_frame(derivatives)

        cum_cost_df = buy2open.reindex(index=sub_quote.index, columns=[derivatives]).fillna(0).cumsum()[[derivatives]]

        res_sub = holding_value - cum_cost_df - cum_sellvalue if not cum_sellvalue.dropna().empty else holding_value - cum_cost_df

        # 期权需要行权收益，期货需要交割收益
        executed_profit = cls._create_executed_profit(end_dt, derivatives, contract_type, holding_value,
                                                      lastdel_multi, sub_quote, net_unit)

        if open_symbol == '卖开':

            return executed_profit * -1, res_sub * -1, holding_value * -1, net_unit * -1, cum_sellvalue * -1, cum_cost_df * -1

        else:

            return executed_profit, res_sub, holding_value, net_unit, cum_sellvalue, cum_cost_df

    @classmethod
    def parse_bo2sc_so2pc(cls, lastdel_multi, derivatives: str, transactions, quote, return_dict=True):
        """

        :param lastdel_multi:
        :param derivatives:
        :param transactions:
        :param quote:
        :param return_dict:
        :return:
        """

        end_dt = lastdel_multi.loc[derivatives, 'EXE_DATE']
        underlying_code = lastdel_multi.loc[derivatives, 'UNDERLYINGWINDCODE']

        sub_quote = quote[[derivatives, underlying_code]] if underlying_code is not None else quote[[derivatives]]

        contract_type = Tools.code_type_detect(derivatives)

        m1 = transactions['委托合约'].isin([derivatives])

        sub_transactions = transactions[m1]

        #  long
        if not sub_transactions[sub_transactions['买卖开平'].isin(['买开', '卖平'])].empty:

            long_sub_transac = sub_transactions[sub_transactions['买卖开平'].isin(['买开', '卖平'])]

            long_executed_profit, long_res_sub, long_holding_value, long_net_unit, long_cum_sellvalue, long_cum_cost_df = cls.sub_parse_general_long(
                end_dt, lastdel_multi, derivatives, long_sub_transac,
                sub_quote, contract_type, open_symbol='买开', close_symbol='卖平')
        else:
            long_executed_profit, long_res_sub, long_holding_value, long_net_unit, long_cum_sellvalue, long_cum_cost_df = pd.DataFrame(
                columns=[derivatives]), pd.DataFrame(columns=[derivatives]), pd.DataFrame(
                columns=[derivatives]), pd.DataFrame(columns=[derivatives]), pd.DataFrame(
                columns=[derivatives]), pd.DataFrame(columns=[derivatives])
        # short
        if not sub_transactions[sub_transactions['买卖开平'].isin(['卖开', '买平'])].empty:

            short_sub_transac = sub_transactions[sub_transactions['买卖开平'].isin(['卖开', '买平'])]

            short_executed_profit, short_res_sub, short_holding_value, short_net_unit, short_cum_sellvalue, short_cum_cost_df = cls.sub_parse_general_short(
                end_dt, lastdel_multi, derivatives, short_sub_transac,
                sub_quote, contract_type, open_symbol='卖开', close_symbol='买平')
        else:
            short_executed_profit, short_res_sub, short_holding_value, short_net_unit, short_cum_sellvalue, short_cum_cost_df = pd.DataFrame(
                columns=[derivatives]), pd.DataFrame(columns=[derivatives]), pd.DataFrame(
                columns=[derivatives]), pd.DataFrame(columns=[derivatives]), pd.DataFrame(
                columns=[derivatives]), pd.DataFrame(columns=[derivatives])

        # , , , , , , , , , , ,
        # long_executed_holder,long_result_holder,long_holding_value_holder,long_unit_holder,long_value_holder,long_cum_cost_df_holder,
        # short_executed_holder,short_result_holder,short_holding_value_holder,short_unit_holder,short_value_holder,short_cum_cost_df_holder

        if return_dict:

            result_holder = {derivatives: (
                long_executed_profit, long_res_sub, long_holding_value, long_net_unit, long_cum_sellvalue,
                long_cum_cost_df,
                short_executed_profit, short_res_sub, short_holding_value, short_net_unit, short_cum_sellvalue,
                short_cum_cost_df)}

            return result_holder
        else:
            return (
                long_executed_profit, long_res_sub, long_holding_value, long_net_unit, long_cum_sellvalue,
                long_cum_cost_df,
                short_executed_profit, short_res_sub, short_holding_value, short_net_unit, short_cum_sellvalue,
                short_cum_cost_df)


class DerivativesItem(DerivativesProcessTools):
    __slot__ = ('_single_transaction',)

    def __init__(self, data_dict: (dict, pd.Series)):
        if isinstance(data_dict, dict) or isinstance(data_dict, pd.Series):
            self._single_transaction = data_dict
        else:
            raise ValueError('data_dict only accept dict or Series')

    @property
    def contract_type(self):
        return Tools.code_type_detect(self.contract)

    @property
    def CONTRACTMULTIPLIER(self):
        return self._single_transaction['CONTRACTMULTIPLIER']

    @property
    def EXE_DATE(self):
        return self._single_transaction['EXE_DATE']  # 到期日

    @property
    def contract(self):
        return self._single_transaction['委托合约']  # 合约名称

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
    def date(self):
        return self._single_transaction['报单日期']


class DerivativesItemHolder(object):
    __slot__ = ('_holder', 'transacted_contracts')

    def __init__(self, transactions: pd.DataFrame):
        self._holder = [DerivativesItem(series) for row, series in transactions.iterrows()]

    def __add__(self, other: (dict, pd.Series, DerivativesItem)):

        if isinstance(other, pd.Series):

            self._holder.append(DerivativesItem(other))
        elif isinstance(other, DerivativesItem):
            self._holder.append(other)
        else:
            raise TypeError('only derivative transaction or DerivativesItem can add!')

    def __len__(self):
        return len(self._holder)

    @property
    def transacted_contracts(self):
        return sorted(set(map(lambda x: x.contract, self._holder)))


class ProcessReportLoadingTools(Tools):

    @staticmethod
    def _load_single_daily_report_(daily_file_path: str):
        """
        导入每日委托单报告
        :param daily_file_path:
        :return:
        """
        report = pd.read_csv(daily_file_path, encoding='GBK')
        match = re.search(r'report(\d{8})\.csv', os.path.split(daily_file_path)[-1])
        dt = int(match.group(1))  # pd.to_datetime(match.group(1), format='%Y%m%d')
        report['报单日期'] = dt
        return report

    @classmethod
    def _load_multi_daily_reports_(cls, report_daily_folder: str):
        """
        遍历多个日度委托单报告
        :param report_daily_folder:
        :return:
        """
        for daily_file_path in glob(report_daily_folder):
            yield cls._load_single_daily_report_(daily_file_path)

    @classmethod
    def _load_multi_period_reports_(cls, report_period_folder: str):
        """
        导入多个一段时间的报告
        :param report_period_folder:
        :return:
        """
        for period_file_path in glob(report_period_folder):
            yield pd.read_excel(period_file_path, sheet_name='report')

    @classmethod
    def load_report(cls,
                    report_file_path: str = 'C:\\Users\\linlu\\Documents\\GitHub\\pf_analysis\\pf_analysis\\optionanalysis\\report_file',
                    daily_report='report*.csv',
                    period_report='report*-*.xlsx',
                    target_cols=['委托合约', '买卖', '开平', '手数', '成交均价', '报单时间', '报单日期'],
                    buysell_cols_replace={'买\u3000': '买', '\u3000卖': '卖'},
                    openclose_cols_replace={'开仓': '开', '平仓': '平', '平昨': '平', '平今': '平'},
                    rule: dict = {'MO\d{4}-[CP]-[0-9]+': 'CFE',
                                  'HO\d{4}-[CP]-[0-9]+': 'CFE',
                                  'IO\d{4}-[CP]-[0-9]+': 'CFE',
                                  'IH\d{4}': 'CFE',
                                  'IF\d{4}': 'CFE',
                                  'IM\d{4}': 'CFE',
                                  'AG\d{4}': 'SHF',
                                  'AU\d{4}': 'SHF'}
                    ):

        if isinstance(report_file_path, str):
            if not os.path.isdir(report_file_path):
                raise ValueError(f'{report_file_path} is not folder')

            # load daily report
            # report_daily = os.path.join(report_file_path, daily_report)
            daily_list = list(cls._load_multi_daily_reports_(os.path.join(report_file_path, daily_report)))
            daily = pd.concat(daily_list) if len(daily_list) != 0 else None

            # load period report
            # report_period = os.path.join(report_file_path, period_report)
            period_all = list(cls._load_multi_period_reports_(os.path.join(report_file_path, period_report)))
            report_period_all = pd.concat(period_all) if len(period_all) != 0 else None
        elif isinstance(report_file_path, pd.DataFrame):
            report_period_all = report_file_path
            daily = None
        else:
            raise ValueError('report_file_path only accept path or df!')

        # merge daily and period
        if daily is not None:
            merged_report = pd.concat([report_period_all, daily]) if period_all is not None else daily
        else:
            merged_report = report_period_all

        _parsed_report = cls._simple_parse_(merged_report,
                                            buysell_cols_replace=buysell_cols_replace,
                                            openclose_cols_replace=openclose_cols_replace, )
        # 委托合约处理空值并且全部大写
        _parsed_report[target_cols[0]] = _parsed_report[target_cols[0]].apply(lambda x: x.strip()).str.upper()

        contracts = _parsed_report[target_cols[0]].unique().tolist()

        contracts_translated_dict = cls.translated_contract_full_name_process(contracts, rule=rule)
        # 补全合约全名
        _parsed_report[target_cols[0]] = _parsed_report[target_cols[0]].replace(contracts_translated_dict)

        _parsed_report['报单时间'] = _parsed_report['报单时间'].fillna('00:00:00')
        dt = _parsed_report['报单日期'].astype(str) + ' ' + _parsed_report['报单时间']
        _parsed_report['报单日期时间'] = pd.to_datetime(dt, format='%Y%m%d %H:%M:%S')

        return _parsed_report

    @staticmethod
    def _loading_data_func_(dt, report_file_path: (str, pd.DataFrame)):
        """
        loaded data
        :param dt:
        :param report_file_path:
        :return:
        """

        if isinstance(report_file_path, str) and report_file_path.endswith('csv'):
            _report = pd.read_csv(report_file_path, encoding='GBK')
            _report['报单日期'] = pd.to_datetime(str(dt), format='%Y%m%d') if isinstance(dt, str) else dt
        elif isinstance(report_file_path, pd.DataFrame):
            _report = report_file_path
        else:
            raise ValueError(f'report_file_path only accept str or pd.DataFrame; but got {type(report_file_path)}')
        return _report

    @staticmethod
    def _completed_contract_full_name(contract, rule: dict = {'MO\d{4}-[CP]-[0-9]+': 'CFE',
                                                              'HO\d{4}-[CP]-[0-9]+': 'CFE',
                                                              'IO\d{4}-[CP]-[0-9]+': 'CFE',
                                                              'IH\d{4}': 'CFE', 'IF\d{4}': 'CFE',
                                                              'IM\d{4}': 'CFE', 'AG\d{4}': 'SHF', 'AU\d{4}': 'SHF'}
                                      ):

        prefix = contract[:2]

        for pattern, suffix_name in rule.items():
            if pattern.startswith(prefix):
                if re.compile(pattern + '$').match(contract):
                    return contract + '.' + suffix_name

        else:
            raise ValueError(f'found {contract} is not in rule struction!')

    @classmethod
    def translated_contract_full_name_process(cls, contracts, rule: dict = {'MO\d{4}-[CP]-[0-9]+': 'CFE',
                                                                            'HO\d{4}-[CP]-[0-9]+': 'CFE',
                                                                            'IO\d{4}-[CP]-[0-9]+': 'CFE',
                                                                            'IH\d{4}': 'CFE', 'IF\d{4}': 'CFE',
                                                                            'IM\d{4}': 'CFE', 'AG\d{4}': 'SHF',
                                                                            'AU\d{4}': 'SHF'}, ):

        suffix = tuple(set(rule.values()))

        # filtered_contracts = (x for x in contracts ) # 有后缀了就不替换了
        h = {contract: cls._completed_contract_full_name(contract, rule) if not contract.endswith(suffix) else contract
             for contract in contracts}

        return h

    @staticmethod
    def _alter_non_traded_transcripts(_report: pd.DataFrame):
        """
        修正部分成交的手数为新的挂单成交手数
        :param _report:
        :return:
        """

        # 全部成交

        all_traded_mask = (_report['挂单状态'] == '全部成交') & (_report['未成交'] == 0)
        all_settle_down = _report[all_traded_mask]
        # 已撤单-部分成交
        partial_traded_mask = (_report['挂单状态'].isin(('已撤单', '部分成交'))) & (
                _report['手数'] != _report['未成交'])
        partial_settle_down = _report[partial_traded_mask].copy(deep=True)

        partial_settle_down['手数'] = partial_settle_down['手数'] - partial_settle_down['未成交']
        partial_settle_down['未成交'] = 0
        partial_settle_down['挂单状态'] = '全部成交'

        _parsed_report = pd.concat([all_settle_down, partial_settle_down])
        _parsed_report['成交均价'] = _parsed_report['成交均价'].astype(float)

        return _parsed_report

    @staticmethod
    def _parse_cols_symbol(_report: pd.DataFrame,
                           buysell_cols_replace={'买\u3000': '买', '\u3000卖': '卖'},
                           openclose_cols_replace={'开仓': '开', '平仓': '平', '平昨': '平', '平今': '平'}
                           ):
        """
        处理部分列的名称

        :param _report:
        :param buysell_cols_replace:
        :param openclose_cols_replace:
        :return:
        """

        ## process 买卖
        _report['买卖'] = _report['买卖'].replace(buysell_cols_replace)
        ## process 开平
        _report['开平'] = _report['开平'].replace(openclose_cols_replace)

        # _report['成交均价'] = _report['成交均价'].astype(float)
        return _report

    @classmethod
    def _simple_parse_(cls, _report: pd.DataFrame,
                       buysell_cols_replace={'买\u3000': '买', '\u3000卖': '卖'},
                       openclose_cols_replace={'开仓': '开', '平仓': '平', '平昨': '平', '平今': '平'},
                       ):
        _report = cls._parse_cols_symbol(_report,
                                         buysell_cols_replace=buysell_cols_replace,
                                         openclose_cols_replace=openclose_cols_replace,
                                         )
        _parsed_report = cls._alter_non_traded_transcripts(_report)
        return _parsed_report

    @classmethod
    def parse(cls, _report: pd.DataFrame,
              buysell_cols_replace={'买\u3000': '买', '\u3000卖': '卖'},
              openclose_cols_replace={'开仓': '开', '平仓': '平', '平昨': '平', '平今': '平'},
              name_process_rule={'MO\d{4}-[CP]-[0-9]+': 'CFE',
                                 'HO\d{4}-[CP]-[0-9]+': 'CFE',
                                 'IO\d{4}-[CP]-[0-9]+': 'CFE',
                                 'IH\d{4}': 'CFE', 'IF\d{4}': 'CFE',
                                 'IM\d{4}': 'CFE', 'AG\d{4}': 'SHF',
                                 'AU\d{4}': 'SHF'}
              ):
        # process 买卖
        _parsed_report = cls._simple_parse_(_report,
                                            buysell_cols_replace=buysell_cols_replace,
                                            openclose_cols_replace=openclose_cols_replace, )

        contracts = _parsed_report['委托合约'].unique().tolist()

        contracts_translated_dict = cls.translated_contract_full_name_process(contracts, rule=name_process_rule)

        _parsed_report['委托合约'] = _parsed_report['委托合约'].replace(contracts_translated_dict)

        _parsed_report['报单时间'] = _parsed_report['报单时间'].fillna('00:00:00')
        dt = _parsed_report['报单日期'].astype(str) + ' ' + _parsed_report['报单时间']
        _parsed_report['报单日期时间'] = pd.to_datetime(dt, format='%Y%m%d %H:%M:%S')

        return _parsed_report


class ProcessReportDataTools(object):
    @staticmethod
    def get_info_last_delivery_multi(contracts, wind_helper: object):
        # contracts = self._transactions['委托合约'].unique().tolist() if contracts is None else contracts
        lastdeliv_and_multi = wind_helper.get_future_info_last_delivery_date_underlying(contracts)
        lastdeliv_and_multi.index.name = '委托合约'
        return lastdeliv_and_multi[['START_DATE', 'EXE_DATE', 'CONTRACTMULTIPLIER', 'UNDERLYINGWINDCODE', 'MARGIN']]

    @staticmethod
    def get_quote_iter(lastdeliv_and_multi: pd.DataFrame, wind_helper, today=None, min_start_dt='2024-01-01', ):
        today = datetime.datetime.today() if today is None else today

        min_start_dt = pd.to_datetime(min(min_start_dt, today.strftime("%Y-%m-%d")))
        got = []

        # 获取合约行情
        for contract, params in lastdeliv_and_multi.iterrows():
            start = min(min_start_dt, params['START_DATE']).strftime("%Y-%m-%d")
            end = min(today, params['EXE_DATE']).strftime("%Y-%m-%d")
            q = wind_helper.wind_wsd_quote_reduce(contract, start, end, required_cols=('close',)).dropna()[
                'CLOSE'].to_frame(contract)
            q.index.name = 'date'
            yield q
            got.append(contract)

        # 获取underlyling 行情
        for underlying in lastdeliv_and_multi['UNDERLYINGWINDCODE'].dropna().unique():
            if underlying not in got:
                q = \
                    wind_helper.wind_wsd_quote_reduce(underlying, min_start_dt, today,
                                                      required_cols=('close',)).dropna()[
                        'CLOSE'].to_frame(underlying)
                q.index.name = 'date'
                yield q

    @classmethod
    def get_quote(cls, lastdeliv_and_multi: pd.DataFrame, wind_helper, today=None, min_start_dt='2024-01-01', ):

        h = cls.get_quote_iter(lastdeliv_and_multi, wind_helper, today=today, min_start_dt=min_start_dt)

        quote = pd.concat(h, axis=1).sort_index()
        quote.index = pd.to_datetime(quote.index)

        return quote

    @classmethod
    def get_quote_and_info(cls, contracts, wind_helper, today=datetime.datetime.today(), start_with='1999-01-01'):
        lastdeliv_and_multi = cls.get_info_last_delivery_multi(contracts, wind_helper)
        quote = cls.get_quote(lastdeliv_and_multi, wind_helper, today=today, min_start_dt=start_with, )

        # reduce useless quote data
        quote = quote[quote.index >= start_with]

        return quote


class ProcessReportSingle(ProcessReportLoadingTools, ProcessReportDataTools):  # , ProcessReportCalculationTools
    def __init__(self, dt, report_file_path: str = 'report.csv',
                 target_cols=['委托合约', '买卖', '开平', '手数', '成交均价', '报单时间', '报单日期', '报单日期时间'],

                 name_process_rule={'MO\d{4}-[CP]-[0-9]+': 'CFE',
                                    'HO\d{4}-[CP]-[0-9]+': 'CFE',
                                    'IO\d{4}-[CP]-[0-9]+': 'CFE',
                                    'IH\d{4}': 'CFE',
                                    'IF\d{4}': 'CFE',
                                    'IM\d{4}': 'CFE',
                                    'AG\d{4}': 'SHF',
                                    'AU\d{4}': 'SHF'},
                 buysell_cols_replace={'买\u3000': '买', '\u3000卖': '卖'},
                 openclose_cols_replace={'开仓': '开', '平仓': '平', '平昨': '平', '平今': '平'},
                 ):

        _report = self._loading_data_func_(dt, report_file_path=report_file_path)
        _parsed_report = self.parse(_report, buysell_cols_replace=buysell_cols_replace,
                                    openclose_cols_replace=openclose_cols_replace,
                                    name_process_rule=name_process_rule)

        self._transactions = _transactions = _parsed_report[target_cols]

    def reduced_contracts(self, filter_func=None):
        """
        create contracts list and if pass filter_func will operator func else use code type
        :param filter_func:
        :return:
        """
        if filter_func is None:
            filter_func = lambda x: True
        elif isinstance(filter_func, str):
            if filter_func.lower() == 'option':
                filter_func = lambda x: Tools.code_type_detect(x) == 'Option'
            elif filter_func.lower() == 'future':
                filter_func = lambda x: Tools.code_type_detect(x) == 'Future'
            else:
                raise ValueError('filter_func only accept callable object!')

        elif isinstance(filter_func, Callable):
            pass
        else:
            raise ValueError('filter_func only accept callable object!')

        contracts = self._transactions['委托合约'].unique().tolist()

        option_list = list(filter(filter_func, contracts))

        return option_list

    @property
    def reduced_transactions(self):
        transactions = self._transactions
        transactions['amt_100'] = transactions['手数'] * transactions['成交均价']
        out = transactions.groupby(['委托合约', '买卖', '开平', '报单日期'])[['手数', 'amt_100']].sum().reset_index()
        out['成交均价'] = out['amt_100'] / out['手数']
        output_cols = ['委托合约', '买卖', '开平', '报单日期', '手数', '成交均价']
        return out[output_cols]

    @staticmethod
    def prepare_transactions(transactions, trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, }):
        transactions['买卖开平'] = transactions['买卖'] + transactions['开平']
        transactions['trade_type'] = transactions['买卖开平'].replace(trade_type_mark)
        transactions['unit'] = transactions['手数'] * transactions['CONTRACTMULTIPLIER']  # 添加合约乘数
        transactions['cost'] = transactions['unit'] * transactions['成交均价'] * transactions['trade_type']
        transactions['报单日期'] = pd.to_datetime(transactions['报单日期'], format='%Y%m%d')
        # ['委托合约', '买卖', '开平', '报单日期', '手数', '成交均价', 'START_DATE', 'EXE_DATE', 'CONTRACTMULTIPLIER']
        return transactions


class ProcessReport(ProcessReportSingle):
    def __init__(self,
                 report_file_path: str = 'C:\\Users\\linlu\\Documents\\GitHub\\pf_analysis\\pf_analysis\\optionanalysis\\report_file',
                 daily_report='report*.csv',
                 period_report='report*-*.xlsx',
                 target_cols=['委托合约', '买卖', '开平', '手数', '成交均价', '报单时间', '报单日期', '报单日期时间'],
                 buysell_cols_replace={'买\u3000': '买', '\u3000卖': '卖'},
                 openclose_cols_replace={'开仓': '开', '平仓': '平', '平昨': '平', '平今': '平'},
                 contract_2_person_rule={'MO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                         'HO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                         'IO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                         'IH\d{4}.CFE': 'wj',
                                         'IF\d{4}.CFE': 'wj',
                                         'IM\d{4}.CFE': 'll',
                                         'AG\d{4}.SHF': 'gr',
                                         'AU\d{4}.SHF': 'wj',
                                         'CU\d{4}.SHF': 'wj',
                                         'AL\d{4}.SHF': 'gr'}, **kwargs):
        rule = dict([k.split('.') for k in contract_2_person_rule.keys()])

        _parsed_report = self.load_report(report_file_path,
                                          daily_report=daily_report,
                                          period_report=period_report,
                                          target_cols=target_cols,
                                          buysell_cols_replace=buysell_cols_replace,
                                          openclose_cols_replace=openclose_cols_replace,
                                          rule=rule)

        super().__init__(None, _parsed_report, name_process_rule=rule)
        self.contract_2_person_rule = contract_2_person_rule

    def create_transactions(self, last_deliv_and_multi,
                            trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, },
                            reduced=False, return_df=False, ):
        transactions = self.reduced_transactions if reduced else self._transactions

        # transactions = transactions[['报单日期', '报单时间']].apply(lambda x, y: x + ' ' + y)

        merged_transactions = pd.merge(transactions, last_deliv_and_multi.reset_index(),
                                       left_on='委托合约', right_on='委托合约')

        transactions = self.prepare_transactions(merged_transactions, trade_type_mark=trade_type_mark)

        return transactions if return_df else DerivativesItemHolder(transactions)

    def create_current_cost_price(self, lastdel_multi,
                                  trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, },
                                  contract_col='委托合约',
                                  share_col='手数',
                                  price_col='成交均价',
                                  trade_type='trade_type',
                                  dt_col='报单日期时间', method='FIFO'):
        # 处理交易对，计算开仓成本
        transcations_multi = self.create_transactions(lastdel_multi, reduced=False, return_df=True,
                                                      trade_type_mark=trade_type_mark)  # .sort_values('报单日期')

        avg_transcripts_dict = dict()
        avg_price = []
        for k, direct, v in DequeMatcher.multi_contract_recus_match_v2(transcations_multi,
                                                                       contract_col=contract_col,
                                                                       share_col=share_col,
                                                                       price_col=price_col,
                                                                       trade_type=trade_type,
                                                                       dt_col=dt_col, method=method):
            avg_transcripts_dict[(k, direct)] = v
            avg_price.append([k, direct, v[0]])

        avg_price_df = pd.DataFrame(filter(lambda x: x[-1] is not None, avg_price),
                                    columns=['contract_code', '持仓方向', '平均开仓成本'])
        res_avg_price_df = pd.merge(avg_price_df, lastdel_multi[['EXE_DATE', 'CONTRACTMULTIPLIER','MARGIN']].reset_index(),
                                    left_on=['contract_code'], right_on=['委托合约'], how='left').drop_duplicates()
        res_avg_price_rd_df = res_avg_price_df[res_avg_price_df['EXE_DATE'] >= datetime.datetime.today()]
        # contract_list = res_avg_price_rd_df['contract_code'].unique()
        # contract_mask = res_avg_price_rd_df['contract_code'].isin(contract_list)

        return res_avg_price_rd_df

    @staticmethod
    def conct_and_rd_all_zero_rows_and_parse_more(long_cum_cost_df):
        mask = ~(long_cum_cost_df.fillna(0) == 0).all(axis=1)
        mask = mask.shift(-1).fillna(True)
        df = long_cum_cost_df[mask]
        # df = df.index.dt.date
        return df

    @classmethod
    def _merged_function(cls, long_holding_value_holder, long_unit_holder, long_cum_cost_df_holder, long_value_holder,
                         long_executed_holder, long_result_holder, quote, direct='long'):
        """

        :param long_holding_value_holder:  当前持仓价值-即残值
        :param long_unit_holder:   当前持仓份数
        :param long_cum_cost_df_holder: 当前累计开仓成本
        :param long_value_holder:  累计平仓价值
        :param long_executed_holder: 累计行权收益
        :param long_result_holder: 累计净损益
        :param quote:
        :param direct:
        :return:
        """
        if direct.lower() not in ('long', 'short'):
            raise ValueError(f'unknown direct parameters: {direct}')

        direct_name = '多头' if direct.lower() == 'long' else '空头'

        # long or short
        long_holding_df = pd.concat(long_holding_value_holder, axis=1).reindex(index=quote.index).fillna(0)  # 期权残值
        long_holding_df = cls.conct_and_rd_all_zero_rows_and_parse_more(long_holding_df)

        long_unit_df = pd.concat(long_unit_holder, axis=1).reindex(index=quote.index).fillna(0)  # 多头持仓份数
        long_unit_df = cls.conct_and_rd_all_zero_rows_and_parse_more(long_unit_df)

        long_cum_cost_df = pd.concat(long_cum_cost_df_holder, axis=1).reindex(index=quote.index).fillna(0)  # 累计开仓成本
        long_cum_cost_df = cls.conct_and_rd_all_zero_rows_and_parse_more(long_cum_cost_df)

        long_value_df = pd.concat(long_value_holder, axis=1).reindex(index=quote.index).ffill().fillna(0)  # '累计平仓价值'
        long_value_df = cls.conct_and_rd_all_zero_rows_and_parse_more(long_value_df)

        long_executed_df = pd.concat(long_executed_holder, axis=1).reindex(index=quote.index).fillna(0)  # '累计行权收益'
        long_executed_df = cls.conct_and_rd_all_zero_rows_and_parse_more(long_executed_df)

        long_res = pd.concat(long_result_holder, axis=1).reindex(index=quote.index).ffill()  # 累计净损益
        long_res = cls.conct_and_rd_all_zero_rows_and_parse_more(long_res)

        temp_holder = (long_holding_df, long_unit_df, long_cum_cost_df, long_value_df, long_executed_df, long_res)

        info_data_key = (f'衍生品{direct_name}持仓价值',  #: long_holding_df,
                         f'衍生品{direct_name}剩余份数',  #: long_unit_df,
                         f'衍生品{direct_name}累计开仓成本',  #: long_cum_cost_df,
                         f'衍生品{direct_name}累计平仓价值',  #: long_value_df,
                         f'衍生品{direct_name}累计行权收益',  #: long_executed_df,
                         f'衍生品{direct_name}累计净损益',  #: long_res,
                         )

        info_dict = dict(zip(info_data_key, temp_holder))

        return info_dict

    def parse_transactions_with_quote_v2(self, quote, lastdel_multi,
                                         trade_type_mark={"卖开": 1, "卖平": -1, "买开": 1, "买平": -1, "买平今": -1, },
                                         ):
        transactions = self.create_transactions(lastdel_multi, reduced=True, return_df=True,
                                                trade_type_mark=trade_type_mark)

        result_holder = [
            DerivativesItem.parse_bo2sc_so2pc(lastdel_multi, contract, sub_transaction, quote, return_dict=False) for
            contract, sub_transaction in
            transactions.groupby('委托合约')
        ]

        l1, l2, l3, l4, l5, l6, s1, s2, s3, s4, s5, s6 = list(zip(*result_holder))

        # long_executed_holder : l1
        # long_result_holder  : l2
        # long_holding_value_holder: l3
        # long_unit_holder :l4
        # long_value_holder :l5
        # long_cum_cost_df_holder:l6

        # long
        long_info_dict = self._merged_function(l3, l4, l6, l5, l1, l2, quote, direct='long')
        # short
        short_info_dict = self._merged_function(s3, s4, s6, s5, s1, s2, quote, direct='short')

        return ChainMap(long_info_dict, short_info_dict)


class SummaryFunctions(object):
    @staticmethod
    def _summary_function(holding_df, cum_cost_df, value_df, executed_df, res, symbol='期权'):
        sum_realized_value = value_df.sum(axis=1).to_frame('累计平仓价值') * -1
        sum_cost = cum_cost_df.sum(axis=1).to_frame('累计开仓成本')
        cross_resid_value = holding_df.sum(axis=1).to_frame(symbol + '残值')
        sum_executed = executed_df.sum(axis=1).to_frame('行权收益')
        sum_res = res.sum(axis=1).to_frame('累计净损益(右轴)')

        summary_info = pd.concat([sum_realized_value, sum_cost, cross_resid_value, sum_executed, sum_res],
                                 axis=1).fillna(0)

        summary_info[symbol + '累计价值（残值+行权收益+平仓收益）'] = summary_info['累计平仓价值'] + summary_info[
            symbol + '残值'] + summary_info['行权收益']
        summary_info['累计持仓收益率'] = (summary_info['累计净损益(右轴)'] / summary_info['累计开仓成本'].abs()).fillna(
            0)
        summary_info['累计净值'] = summary_info['累计持仓收益率'] + 1

        summary_info = ProcessReport.conct_and_rd_all_zero_rows_and_parse_more(summary_info)

        return summary_info

    @classmethod
    def create_summary_info(cls, item_list, result_dict, symbol='期权'):

        # option_list = list(filter(lambda x: cls.code_type_detect(x) == 'Option', transactions['委托合约'].unique()))

        # contracts = PR.reduced_contracts(filter_func='Future')

        long_summary_info = cls._summary_function(result_dict['衍生品多头持仓价值'].reindex(columns=item_list),
                                                  result_dict['衍生品多头累计开仓成本'].reindex(columns=item_list),
                                                  result_dict['衍生品多头累计平仓价值'].reindex(columns=item_list),
                                                  result_dict['衍生品多头累计行权收益'].reindex(columns=item_list),
                                                  result_dict['衍生品多头累计净损益'].reindex(columns=item_list),
                                                  symbol=symbol)

        short_summary_info = cls._summary_function(result_dict['衍生品空头持仓价值'].reindex(columns=item_list),
                                                   result_dict['衍生品空头累计开仓成本'].reindex(columns=item_list),
                                                   result_dict['衍生品空头累计平仓价值'].reindex(columns=item_list),
                                                   result_dict['衍生品空头累计行权收益'].reindex(columns=item_list),
                                                   result_dict['衍生品空头累计净损益'].reindex(columns=item_list),
                                                   symbol=symbol)

        short_summary_info = short_summary_info.fillna(0)

        return long_summary_info, short_summary_info

    # @classmethod
    # def general_summary_and_writing_process(cls, special_contracts, result_dict,
    #                                         today_str=None,
    #                                         symbol='期货',
    #                                         output_path=None,
    #                                         store=True):
    #
    #     if today_str is None:
    #         today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')
    #     else:
    #         today_str = today_str
    #
    #     long_summary_info, short_summary_info = cls.create_summary_info(special_contracts, result_dict,
    #                                                                     symbol=symbol)
    #
    #     if output_path is not None and isinstance(output_path, str) and os.path.exists(output_path) and store:
    #
    #         store_path = os.path.join(output_path, f'{symbol}日收益率统计及汇总@{today_str}v2.xlsx')
    #
    #         with pd.ExcelWriter(store_path) as f:
    #             long_summary_info.to_excel(f, '多头输出')
    #             result_dict['衍生品多头持仓价值'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                           '衍生品多头持仓价值截面')
    #             result_dict['衍生品多头累计开仓成本'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                               '衍生品多头累计开仓成本')
    #             result_dict['衍生品多头累计平仓价值'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                               '衍生品多头累计平仓价值')
    #             result_dict['衍生品多头累计行权收益'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                               '衍生品多头累计行权收益')
    #             result_dict['衍生品多头累计净损益'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                             '衍生品多头累计净损益')
    #             result_dict['衍生品多头剩余份数'].reindex(columns=special_contracts).to_excel(f, '衍生品多头剩余合约数')
    #
    #             short_summary_info.to_excel(f, '空头输出')
    #             result_dict['衍生品空头持仓价值'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                           '衍生品空头持仓价值截面')
    #             result_dict['衍生品空头累计开仓成本'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                               '衍生品空头累计开仓成本')
    #             result_dict['衍生品空头累计平仓价值'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                               '衍生品空头累计平仓价值')
    #             result_dict['衍生品空头累计行权收益'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                               '衍生品空头累计行权收益')
    #             result_dict['衍生品空头累计净损益'].reindex(columns=special_contracts).to_excel(f,
    #                                                                                             '衍生品空头累计净损益')
    #             result_dict['衍生品空头剩余份数'].reindex(columns=special_contracts).to_excel(f, '衍生品空头剩余合约数')
    #     else:
    #         if not os.path.exists(output_path):
    #             warnings.warn('unknown output_path parameter, will skip output process')
    #
    #     return long_summary_info, short_summary_info

    # def future_summary(self, info_dict, output_path=None,
    #                    today_str=pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')):
    #
    #     contracts = PR.reduced_contracts(filter_func='Future')
    #
    #     long_summary_info, short_summary_info = self.general_summary_and_writing_process(contracts, info_dict,
    #                                                                                      today_str,
    #                                                                                      symbol='期货',
    #                                                                                      output_path=output_path)
    #
    #     return long_summary_info, short_summary_info
    #
    # def option_summary(self, info_dict, output_path=None,
    #                    today_str=pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')):
    #
    #     contracts = PR.reduced_contracts(filter_func='Option')
    #
    #     long_summary_info, short_summary_info = self.general_summary_and_writing_process(contracts, info_dict,
    #                                                                                      today_str,
    #                                                                                      symbol='期权',
    #                                                                                      output_path=output_path)
    #
    #     return long_summary_info, short_summary_info
    #
    # def general_summary(self, special_contracts, info_dict, output_path=None,
    #                     today_str=pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')):
    #
    #     return self.general_summary_and_writing_process(special_contracts, info_dict,
    #                                                     today_str=today_str,
    #                                                     symbol='',
    #                                                     output_path=output_path)

    @staticmethod
    def contract_link_commodity(contracts):
        for contract in contracts:
            commodity_type = Tools.code_commodity_detect(contract)
            con_type = Tools.code_type_detect(contract)
            yield contract, con_type, commodity_type

    @staticmethod
    def contract_link_person(contracts, contract_2_person_rule={'MO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                                                'HO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                                                'IO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                                                'IH\d{4}.CFE': 'wj',
                                                                'IF\d{4}.CFE': 'wj',
                                                                'IM\d{4}.CFE': 'll',
                                                                'AG\d{4}.SHF': 'gr',
                                                                'AU\d{4}.SHF': 'wj',
                                                                'AL\d{4}.SHF': 'gr'}):

        for contract in contracts:
            con_type = Tools.code_type_detect(contract)
            commodity_type = Tools.code_commodity_detect(contract)

            # result = filter(None,map(lambda x: re.compile(x).match(contract),contract_2_person_rule.keys() ))
            alternative = []
            for pattern, person in contract_2_person_rule.items():

                if re.compile(pattern).match(contract):
                    code_pattern = contract == pattern
                    alternative.append((code_pattern, person))

            if len(alternative) == 1:
                person = alternative[0][-1]
                yield contract, person, con_type, commodity_type
            elif len(alternative) > 1:
                for code_pattern, person in alternative:
                    if code_pattern:
                        yield contract, person, con_type, commodity_type
                else:
                    raise ValueError(f'got multi-regex({len(alternative)}) rules on {contract}:{alternative}')
            else:
                yield contract, None, con_type, commodity_type

    @staticmethod
    def create_daily_summary_file_path(output_path='./', version='v2'):
        if output_path is None:
            output_path = './'

        today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')

        output_name_format = f'日度衍生品交易收益率统计及汇总@{today_str}{version}.xlsx'

        _path = os.path.join(output_path, output_name_format)

        return _path

    @staticmethod
    def long_short_merge(commodity, long, short):

        common_cols = ['累计平仓价值', '累计开仓成本', f'{commodity}残值',
                       '行权收益', '累计净损益(右轴)', f'{commodity}累计价值（残值+行权收益+平仓收益）',
                       '累计持仓收益率']

        long_mask = long is None
        short_mask = short is None

        h = []
        for col in common_cols[:-1]:
            mulp = -1 if col != '累计净损益(右轴)' else 1
            if long_mask and short_mask:
                c = None
            elif long_mask and not short_mask:
                c = short[col].fillna(0).to_frame(col) * mulp
            elif short_mask and not long_mask:
                c = long[col].fillna(0).to_frame(col)
            else:
                c = pd.concat([long[col], short[col] * mulp], axis=1).fillna(0).sum(axis=1).to_frame(col)

            h.append(c)

        summary_ls_merged = pd.concat(h, axis=1).sort_index().fillna(0)

        summary_ls_merged = ProcessReport.conct_and_rd_all_zero_rows_and_parse_more(summary_ls_merged)

        summary_ls_merged['累计持仓收益率'] = (
                summary_ls_merged['累计净损益(右轴)'] / summary_ls_merged['累计开仓成本']).fillna(0, limit=1)
        summary_ls_merged['累计净值'] = summary_ls_merged['累计持仓收益率'] + 1
        return summary_ls_merged


class ReportAnalyst(ProcessReport, SummaryFunctions):

    # def grouper(self, df_dict, link_df, groupby='person'):
    #     all_ls_output = all(map(lambda x: '多空输出' in x, df_dict.keys()))
    #
    #     pass

    def groupby_person_summary(self, info_dict, person_link_df, groupby='person'):
        if groupby not in person_link_df.columns:
            raise ValueError(f'group by value({groupby}) on in link_df!')

        # person_ls_summary_dict = {}
        person_holder = {}
        for person, dfdd in person_link_df.groupby(groupby):
            commodity_contracts = dfdd['contract'].unique().tolist()
            x_long_summary_info, x_short_summary_info = self.create_summary_info(commodity_contracts,
                                                                                 info_dict,
                                                                                 symbol=person)
            summary_ls_merged = self.long_short_merge(person, x_long_summary_info, x_short_summary_info)
            # person_ls_summary_dict[person + '多空输出'] = summary_ls_merged

            res = summary_ls_merged.copy(deep=True)

            res['累计持仓收益率'] = (res['累计净损益(右轴)'] / res['累计开仓成本'].abs()).fillna(0)
            res['累计净值'] = res['累计持仓收益率'] + 1

            for commodity, ddff in dfdd.groupby('commodity'):
                commodity_contracts2 = ddff['contract'].unique().tolist()

                sub_x_long_summary_info, sub_x_short_summary_info = self.create_summary_info(commodity_contracts2,
                                                                                             info_dict,
                                                                                             symbol=person)
                required_cols = ['累计净损益(右轴)', '累计开仓成本']
                reduced_comm_res = self.long_short_merge(person, sub_x_long_summary_info, sub_x_short_summary_info)[
                    required_cols]

                temp = (reduced_comm_res['累计净损益(右轴)'] / reduced_comm_res['累计开仓成本'].abs()).fillna(0)
                res[f'{commodity[:2]}净值'] = temp.copy(deep=True).fillna(0) + 1
                # res[f'{commodity[:2]}净值'] = res[f'{commodity[:2]}净值'].fillna(1)
                res[f'{commodity[:2]}损益'] = reduced_comm_res['累计净损益(右轴)'].copy(deep=True).fillna(0)
                # res[f'{commodity[:2]}损益'] = res[f'{commodity[:2]}损益']
            person_holder[person] = res
        return person_holder  # , person_ls_summary_dict

    def groupby_commodity_summary(self, info_dict, commodity_contract_df, groupby='commodity'):
        if groupby not in commodity_contract_df.columns:
            raise ValueError(f'group by value({groupby}) on in link_df!')
        contract_summary_dict = {}
        merged_summary_dict = {}
        for commodity, dfdd in commodity_contract_df.groupby(groupby):
            commodity_contracts = dfdd['contract'].unique().tolist()
            x_long_summary_info, x_short_summary_info = self.create_summary_info(commodity_contracts,
                                                                                 info_dict,
                                                                                 symbol=commodity)
            if not x_long_summary_info.empty and not (x_long_summary_info == 0).all().all():

                contract_summary_dict[f'{commodity}多头输出'] = x_long_summary_info
                # x_long_summary_info.to_excel(f, f'{commodity}多头输出')
            else:
                x_long_summary_info = None

            if not x_short_summary_info.empty and not (x_short_summary_info == 0).all().all():
                x_short_summary_info = self.conct_and_rd_all_zero_rows_and_parse_more(x_short_summary_info)
                contract_summary_dict[f'{commodity}空头输出'] = x_short_summary_info
                # x_short_summary_info.to_excel(f, f'{commodity}空头输出')
            else:
                x_short_summary_info = None

            summary_ls_merged = self.long_short_merge(commodity, x_long_summary_info, x_short_summary_info)
            merged_summary_dict[commodity + '多空输出'] = summary_ls_merged
        return contract_summary_dict, merged_summary_dict

    @staticmethod
    def store_2_excel(store_path, person_holder_dict, merged_summary_dict, contract_summary_dict, info_dict, contracts):
        with pd.ExcelWriter(store_path) as f:

            # 分人

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

    def group_by_summary(self, info_dict, base_store_path=None, return_data=False, store_2_excel=True):

        contracts = self.reduced_contracts()
        person_link_df = pd.DataFrame(
            list(self.contract_link_person(contracts, contract_2_person_rule=self.contract_2_person_rule)),
            columns=['contract', 'person', 'symbol', 'commodity'])

        commodity_contract_df = person_link_df[['contract', 'symbol', 'commodity']].drop_duplicates()

        contract_summary_dict, merged_summary_dict = self.groupby_commodity_summary(info_dict, commodity_contract_df,
                                                                                    groupby='commodity')

        person_holder_dict = self.groupby_person_summary(info_dict,
                                                         person_link_df, groupby='person')

        # 分人

        ## 分人分项目分年度计算盈亏
        if store_2_excel:
            store_path = self.create_daily_summary_file_path(output_path=base_store_path)
            self.store_2_excel(store_path, person_holder_dict, merged_summary_dict, contract_summary_dict, info_dict,
                               contracts)

        if return_data:
            return person_holder_dict, merged_summary_dict, contract_summary_dict

    def summary_person_info(self, person_summary_dict: dict, merged_summary_dict: dict, ):
        person_holder = {}
        contracts = PR.reduced_contracts()
        person_link_df = pd.DataFrame(
            list(self.contract_link_person(contracts, contract_2_person_rule=self.contract_2_person_rule)),
            columns=['contract', 'person', 'symbol', 'commodity'])

        for person, ob in person_link_df.groupby('person'):
            # required_person_cols = ['累计净损益(右轴)', f'{person[:2]}累计价值（残值+行权收益+平仓收益）',
            #                         '累计开仓成本']

            result = person_summary_dict[person[:2] + '多空输出']

            res = result
            res['累计持仓收益率'] = (res['累计净损益(右轴)'] / res['累计开仓成本'].abs()).fillna(0)
            res['累计净值'] = res['累计持仓收益率'] + 1

            for commodity in ob['commodity'].unique():
                # required_comm_cols = ['累计净损益(右轴)', f'{symbol[:2]}累计价值（残值+行权收益+平仓收益）',
                #                       '累计开仓成本']

                comm_res = merged_summary_dict[commodity + '多空输出']

                comm_res[f'{commodity[:2]}持仓收益率'] = (
                        comm_res['累计净损益(右轴)'] / comm_res['累计开仓成本'].abs()).fillna(0)

                res[f'{commodity[:2]}净值'] = comm_res[f'{commodity[:2]}持仓收益率'].fillna(0) + 1
                res[f'{commodity[:2]}净值'] = res[f'{commodity[:2]}净值'].fillna(1)
                res[f'{commodity[:2]}损益'] = comm_res['累计净损益(右轴)']
                res[f'{commodity[:2]}损益'] = res[f'{commodity[:2]}损益'].fillna(0)
            person_holder[person] = res
        return person_holder

        ## 需要一个期权对冲，期货对冲
        ## 分个人收益统计,分品种净值化和损益,留下2天的持仓状态


class Configs(object):
    def __init__(self, conf_file='config.yml'):
        full_path = detect_file_full_path(conf_file=conf_file)
        with open(full_path, 'r', encoding="utf-8") as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

    def __getitem__(self, item):
        return self.config[item]


if __name__ == '__main__':
    today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')

    wh = WindHelper()

    PR = ReportAnalyst(
        report_file_path='C:\\Users\\linlu\\Documents\\GitHub\\pf_analysis\\pf_analysis\\optionanalysis\\report_file',
        contract_2_person_rule={'MO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'HO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'IO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'IM\d{4}.CFE': 'll',

                                'IH\d{4}.CFE': 'wj',
                                'IF\d{4}.CFE': 'wj',
                                'AU\d{4}.SHF': 'wj',
                                'CU\d{4}.SHF': 'wj',
                                'CU\d{4}[CP]\d{5}.SHF': 'wj',
                                'AL2406.SHF': 'wj',

                                'AG\d{4}.SHF': 'gr',
                                'AL2303.SHF': 'gr',
                                'SN2406.SHF': 'gr',

                                }

    )

    contracts = PR.reduced_contracts()

    quote = PR.get_quote_and_info(contracts, wh, start_with='2022-09-04')

    lastdel_multi = PR.get_info_last_delivery_multi(contracts, wh)

    info_dict = PR.parse_transactions_with_quote_v2(quote, lastdel_multi,
                                                    trade_type_mark={"卖开": 1, "卖平": -1,
                                                                     "买开": 1, "买平": -1,
                                                                     "买平今": -1, }

                                                    )

    person_holder, merged_summary_dict, contract_summary_dict = PR.group_by_summary(
        info_dict, return_data=True)

    # PR.summary_person_info(person_summary_dict, merged_summary_dict, info_dict, lastdel_multi, quote, )

    print(1)
    pass
