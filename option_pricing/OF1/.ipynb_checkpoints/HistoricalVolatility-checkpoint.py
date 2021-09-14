# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

import datetime
import scipy.stats as sps

try:
    from ..APICenter import API
    from ..DataCenter import DataCenter
except ValueError:
    import sys
    sys.path.append('../')
    from APICenter import API
    from DataCenter import DataCenter as DATA

Api = API()

Data = DataCenter()


# load function in current path


# CalVolatility
class HistoricalVolatility():
    """下面以计算股票的历史波动率为例加以说明。
    1、从市场上获得标的股票在固定时间间隔(如每天、每周或每月等)上的价格。
    2、对于每个时间段，求出该时间段末的股价与该时段初的股价之比的自然对数。
    3、求出这些对数值的标准差，再乘以一年中包含的时段数量的平方根(如，选取时间间隔为每天，则若扣除闭市，每年中有250个交易日，应乘以根号250)，得到的即为历史波动率。
    针对权证来说，标的证券的波动率是影响其价值的重要因素之一。在其他参数不变的情况下，标的证券价格波动越大，权证的价值也越大。常见的波动率表述有两种：历史波动率和隐含波动率。他们在含义和计算过程的区别如下。
    历史波动率是使用历史的股价数据计算得到的波动率数值。"""

    def __init__(self):
        pass

    def getrecentXDayHV(self, df50etf, recent=50, Underlying='50ETF', LOGrate=True, diffs='PrevClose', Annualised=True):

        df50etfhead = df50etf.head(recent)

        if diffs == 'OpenClose':
            prev = df50etfhead.open
            nex = df50etfhead.close
        elif diffs == 'PrevClose':
            prev = df50etfhead.prev_close
            nex = df50etfhead.close
        if LOGrate:
            changePercentage = np.log(nex / prev)
        else:
            changePercentage = (nex - prev) / prev
        if Annualised:
            return changePercentage.std() * np.sqrt(250)
        else:
            return changePercentage.std()


if __name__ == '__main__':
    # use Sina API and YM =1806
    RTdf = Data.LoadOptionRealTimeDataViaAPI().sort_values('RequestTimeStamp')
    RTGreek = Data.LoadGreekRealTimeDataViaAPI().sort_values('RequestTimeStamp')
    RTGreek.RequestTimeStamp = np.round(RTGreek.RequestTimeStamp)
    RTdf.RequestTimeStamp = np.round(RTdf.RequestTimeStamp)
    CombinedOptiondf = pd.merge(RTGreek, RTdf,
                                on=['OptionContShortName', 'RequestTimeStamp', 'TradingVolume', 'Unknown Var', 'High',
                                    'Low', 'LastestPrice', 'StrikePrice'])
    RiskFreeRate = 0.04
    DividendYield = 0
    columns = ['OptionContShortName', 'BuyPrice', 'SellPrice', 'StrikePrice', 'YM', 'OptionType', 'Unknown Var',
               u'Delta', u'Gamma', u'Theta',
               u'Vega', u'IV', u'TheoreticalValue']

    #df50etf = CombinedOptiondf[CombinedOptiondf.YM == 1806.0]
    # Volatility = HistoricalVolatility().getrecentXDayHV(df50etf, 50)  #
    # type: option volatility
