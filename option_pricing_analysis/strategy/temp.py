# coding=utf-8


import pandas as pd
import numpy as np
import talib

class FactorFunc(object):

    @staticmethod
    def SMA(data,timeperiod=5):
        # 计算简单移动平均线（SMA）
        data[f'SMA{timeperiod}'] = talib.SMA(data['close'], timeperiod=timeperiod)
        return data

    @staticmethod
    def EMA(data,timeperiod=5):
        # 计算简单移动平均线（SMA）
        data[f'EMA{timeperiod}'] = talib.EMA(data['close'], timeperiod=timeperiod)
        return data

    @staticmethod
    def RSI(data,timeperiod=5):
        # 计算相对强弱指数（RSI）
        data[f'RSI{timeperiod}'] = talib.RSI(data['close'], timeperiod=timeperiod)
        return data

    @staticmethod
    def StochasticOscillator(data,fastk_period=14, slowk_period=3, slowk_matype=0,
                                                   slowd_period=3, slowd_matype=0):
        # 计算随机震荡指标（Stochastic Oscillator）
        data['slowk'], data['slowd'] = talib.STOCH(data['high'], data['low'], data['close'],
                                                   fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype,
                                                   slowd_period=slowd_period, slowd_matype=slowd_matype)
        return data

    @staticmethod
    def BollingerBands(data,timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        # 计算布林带（Bollinger Bands）
        data['upperband'], data['middleband'], data['lowerband'] = talib.BBANDS(data['close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

        return data

    @staticmethod
    def ATR(data,timeperiod=14):
        # 计算平均真实波幅（ATR）
        data[f'ATR{timeperiod}'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
        return data

    @staticmethod
    def MACD(data,fastperiod=12, slowperiod=26, signalperiod=9):
        # 计算MACD
        data['MACD'], data['MACDsignal'], data['MACDhist'] = talib.MACD(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return data

    @staticmethod
    def Ichimoku(data):
        # 计算Ichimoku云图
        data['tenkan_sen'] = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
        data['kijun_sen'] = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        data['senkou_span_b'] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)
        data['chikou_span'] = data['close'].shift(-26)
        return data

    @staticmethod
    def OBV(data):
        # 计算OBV（On-Balance Volume）
        data['OBV'] = talib.OBV(data['close'], data['volume'])
        return data

    @staticmethod
    def VMA(data,timeperiod=20):
        # 计算OBV（On-Balance Volume）
        data[f'VMA{timeperiod}'] = data['volume'].rolling(window=timeperiod).mean()
        return data

if __name__ == '__main__':
    pass
