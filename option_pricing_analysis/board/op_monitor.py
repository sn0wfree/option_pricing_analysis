# coding=utf-8

import warnings

from CodersWheel.QuickTool.file_cache import file_cache

warnings.filterwarnings('ignore', category=FutureWarning)
import akshare as ak

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import time
from collections import deque


class WindHelper(object):
    """
        A helper class for interacting with the Wind financial database.
    """

    def __init__(self):
        from WindPy import w
        w.start()
        self.w = w

    def get_wsq_iter(self, code="IO2409-C-3200.CFE,IO2409-C-3300.CFE",
                     cols="rt_date,rt_open,rt_high,rt_low,rt_last,rt_latest,rt_vol,rt_amt,rt_delta,rt_gamma,rt_vega,rt_theta,rt_rho,rt_imp_volatility,rt_ustock_price,rt_ustock_chg,rt_vol_pcr",
                     max_count=1000000,
                     ):

        count = 0
        res_holder = deque(maxlen=1)

        s = self.get_wsq(res_holder, code=code, cols=cols)

        try:
            while count <= max_count:

                res = res_holder[-1]
                yield res
                count += 1
            else:
                self.w.cancelRequest(s.RequestID)
                print('closed')

        except KeyboardInterrupt as e:
            self.w.cancelRequest(s.RequestID)

        finally:
            self.w.cancelRequest(s.RequestID)

    def get_wsq(self, res_holder, code="IO2409-C-3200.CFE,IO2409-C-3300.CFE",
                cols="rt_date,rt_time,rt_pre_close,rt_open,rt_high,rt_low,rt_last,rt_last_amt,rt_last_vol,rt_latest,rt_vol,rt_amt,rt_chg,rt_pct_chg,rt_mkt_vol,rt_high_limit,rt_low_limit,rt_pre_oi,rt_oi,rt_oi_chg,rt_oi_change,rt_delta,rt_gamma,rt_vega,rt_theta,rt_rho,rt_imp_volatility,rt_ustock_price,rt_ustock_chg,rt_vol_pcr"):
        error, init_data = self.w.wsq(code, cols, '', usedf=True)

        res_holder.append(init_data)

        def recall_func(realtime_obj):
            global res_holder
            try:
                # 获取基本数据，后面进行分解
                # res_rd = pd.DataFrame(realtime_obj.Data, columns=realtime_obj.Codes, index=realtime_obj.Fields).T
                sec_code = realtime_obj.Codes
                fields_list = realtime_obj.Fields
                rt_data_list = realtime_obj.Data

                rt_data = pd.DataFrame(rt_data_list, index=sec_code, columns=fields_list).T
                print(rt_data)

                last = res_holder[-1]
                last.loc[sec_code, fields_list] = rt_data
                res_holder.append(last)
            except:
                pass

        s = self.w.wsq(code, cols, func=recall_func)
        return s

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


class AKHelper(object):
    """
        A helper class for interacting with the Wind financial database.
    """

    def __init__(self):
        self._ak = ak

class OptionPortfolio(object):
    pass

class OptBundle(object):

    @staticmethod
    def delta_exposure_constraint(weights, target_delta_exposure, mask_selected):
        return np.sum(weights * mask_selected['f'] * mask_selected['Delta']) - target_delta_exposure

    @staticmethod
    def delta(weights, mask_selected):
        return np.sum(weights * mask_selected['Delta'] * mask_selected['f'] * 100)

    @classmethod
    def delta_constraint(cls, weights, mask_selected, target_delta_exposure):
        return cls.delta(weights, mask_selected) - target_delta_exposure

    @staticmethod
    def gamma_constraint(weights, mask_selected):
        return np.sum(weights * mask_selected['Gamma'] * 100)

        # 定义目标函数和约束条件

    @classmethod
    def objective(cls, weights, mask_selected, target_delta_exposure):
        # 价格上涨时Delta平稳变化
        delta_change = cls.delta_constraint(weights, mask_selected, target_delta_exposure) ** 2

        # 价格下跌时Gamma增加
        gamma_change = -cls.gamma_constraint(weights) ** 2

        return np.sum(weights * mask_selected['fee'] * 100) + delta_change + gamma_change

    @classmethod
    def run_opt(cls, initial_weights, mask_selected, target_delta_exposure, method='SLSQP'):
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda w: cls.delta_constraint(w, mask_selected, target_delta_exposure)},
                       {'type': 'ineq', 'fun': lambda w: cls.gamma_constraint(w, mask_selected)},
                       {'type': 'eq', 'fun': lambda w: cls.delta_exposure_constraint(w, target_delta_exposure)}
                       ]

        objective = lambda w: cls.objective(w, mask_selected, target_delta_exposure)

        # 边界条件
        bounds = [(-1000, 1000)] * len(mask_selected)

        # 执行优化
        result = minimize(objective, initial_weights, constraints=constraints, bounds=bounds, method=method)

        # 打印结果
        print(result)
        print('损失函数：', objective(result.x))
        print("优化的权重:", result.x)
        print("Delta中性:", cls.delta(result.x), mask_selected)
        print("Gamma中性:", cls.gamma_constraint(result.x, mask_selected))
        return result


if __name__ == '__main__':
    import datetime

    target_delta_exposure = -100 * 10000
    wh = WindHelper()
    wh.w.cancelRequest(0)

    s = wh.get_wsq_iter(code="MO2409-P-4500.CFE,MO2409-P-4400.CFE", )

    for ss in s:
        data = ss.copy(deep=True)

        data['RT_DATE'] = data['RT_DATE'].astype(int)

        data['RT_TIME'] = datetime.datetime.now().strftime('%H:%M:%S')

        data.T.to_excel('data.xlsx')

        print(1)

        time.sleep(3)
