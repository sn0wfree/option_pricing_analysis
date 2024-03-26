# coding=utf-8
# -*- coding:utf-8 -*-


# ----------
import requests
import time

from option_pricing_analysis.data import OptionDataCollector
from option_pricing_analysis.data.etf50 import d3

# import requests_cache
# cache_duration = 300
# requests_cache.install_cache(cache_name='APICentercache', backend='memory', expire_after=cache_duration)
__label__ = 'OF5.Extension.Input.2'


# ----------

# ------------


class DataCenter(object):

    # def __new__(cls, DataCenter):
    #     return DataCenter.__new__(cls)

    # Basic Data Request Function Area Start

    def checkNonClosedDay(self, date):
        apiurl = 'http://m.dsjcj.cc/tornado/common?a=check_holiday&date='
        #   """ http://m.dsjcj.cc/tornado/common?a=check_holiday&date=2018-4-25

        #       用这个接口判断是否节假日，取data的值，如果为0为不休市，如果为1则为休市
        #       不休市返回：{"data": 0, "errmsg": "", "error": 0}
        # 休市返回：{"data": 1, "errmsg": "", "error": 0}"""
        response = requests.get(apiurl + date)
        if response.status_code == 200:
            import ast
            return int(ast.literal_eval(response.text)['data'])
        else:
            """raise ValueError('info: response code :%d ;%s' %
                             (response.status_code,response.text))"""
            return 0

    def LoadRiskFreeRate(self):
        # type: () -> object
        # need change part
        return 0.029

    def Load50ETFRealTimeDataViaAPI(self, Underlying=510050, viaapi=False):
        # need change part
        if viaapi:
            response = requests.get('http://hq.sinajs.cn/list=sh510050')
            if response.status_code == 200:
                pass
            else:
                raise ValueError('response.status_code : %d' %
                                 response.status_code)
            text = response.text
            return text
        else:
            import pickle

            new50etfdf = pickle.loads(d3)

            return ({'open': 2.5, 'close': 2.6}, new50etfdf)

    def LoadOptionRealTimeDataViaAPI(self, Underlying=510050, viaapi=False):
        # need change part

        cc = OptionDataCollector.Collector()
        ss = cc.beforeStart()
        columns = [u'BuyVolume', u'BuyPrice', u'LastestPrice',
                   u'SellPrice', u'SellVol', u'HoldingVolume',
                   u'Change', u'StrikePrice', u'YesterdayClosePrice',
                   u'OpenPrice', u'HighPriceinTheory', u'LowPriceinTheory',
                   u'SellPriceT5', u'SellVolumeT5', u'SellPriceT4',
                   u'SellVolumeT4', u'SellPriceT3', u'SellVolumeT3', u'SellPriceT2',
                   u'SellVolumeT2', u'SellPriceT1', u'SellVolumeT1', u'BuyPriceT1',
                   u'BuyVolumeT1', u'BuyPriceT2', u'BuyVolumeT2', u'BuyPriceT3',
                   u'BuyVolumeT3', u'BuyPriceT4', u'BuyVolumeT4', u'BuyPriceT5',
                   u'BuyVolumeT5', u'QuotatesTime', u'MainContractCode', u'StatusCode',
                   u'UnderlyingSecCode', u'UnderlyingSec', u'OptionContShortName',
                   u'Amplitude', u'High', u'Low', u'TradingVolume', u'Amount',
                   u'RequestTimeStamp', u'OptionType', u'YM', u'Unknown Var']
        realtimeoption = cc.regularGetRTOQ(
            ss, time.time(), Optimizer=True)[columns]
        return realtimeoption

    def LoadGreekRealTimeDataViaAPI(self, Underlying=510050):

        cc = OptionDataCollector.Collector()
        ss = cc.beforeStart()
        columns = ['OptionContShortName', 'TradingVolume', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV', 'High', 'Low',
                   'TrasactionCode', 'StrikePrice', 'LastestPrice', 'TheoreticalValue', 'Unknown Var',
                   'RequestTimeStamp']
        realtimegreek = cc.regularGetGI(
            ss, time.time(), Optimizer=True)[columns]
        return realtimegreek
    # ----------------------------


if __name__ == '__main__':
    pass

