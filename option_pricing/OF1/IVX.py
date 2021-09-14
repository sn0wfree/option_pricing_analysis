# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np


# import Custom module
try:
    from ..APICenter import API
    from ..DataCenter import DataCenter
except ValueError:
    import sys
    sys.path.append('../')
    from APICenter import API
    from DataCenter import DataCenter
__label__ = 'OF1.Model.Info.3'
Api = API()
Data = DataCenter()


def findnearone(CurrentUnderlyingPrice, dflist, outputtype='range', lengthofstrategy=1):
    """
    this function is to get near strike price which assign to current price
    :param CurrentUnderlyingPrice:
    :param dflist:
    :param outputtype:
    :param lengthofstrategy:
    :return:
    """

    if outputtype == 'range%':
        s, l = CurrentUnderlyingPrice * (1 - 0.05 * lengthofstrategy), CurrentUnderlyingPrice * (
            1 + 0.05 * lengthofstrategy)
        # print value
        dflist['lsqStrikePrice'] = np.square(dflist.StrikePrice - l)
        dflist['ssqStrikePrice'] = np.square(dflist.StrikePrice - s)
        large = dflist.sort_values('lsqStrikePrice').StrikePrice.values[0]
        small = dflist.sort_values('ssqStrikePrice').StrikePrice.values[0]
        return small, large
    elif outputtype == 'range':
        s, l = CurrentUnderlyingPrice - 0.05 * \
            lengthofstrategy, CurrentUnderlyingPrice + 0.05 * lengthofstrategy
        # print value
        dflist['lsqStrikePrice'] = np.square(dflist.StrikePrice - l)
        dflist['ssqStrikePrice'] = np.square(dflist.StrikePrice - s)

        large = dflist.sort_values('lsqStrikePrice').StrikePrice.values[0]
        small = dflist.sort_values('ssqStrikePrice').StrikePrice.values[0]
        return small, large
    elif outputtype == 'match':
        dflist['sqStrikePrice'] = np.square(
            dflist.StrikePrice - CurrentUnderlyingPrice)

        if lengthofstrategy != 1:
            listsl = dflist.sort_values('sqStrikePrice').StrikePrice.values[
                :lengthofstrategy]
            return listsl
        else:
            small = large = dflist.sort_values(
                'sqStrikePrice').StrikePrice.values[0]
            return small, large
    elif outputtype == 'nearest':
        lStrikePrice = dflist[dflist.StrikePrice >=
                              CurrentUnderlyingPrice].StrikePrice.values
        sStrikePrice = dflist[dflist.StrikePrice <=
                              CurrentUnderlyingPrice].StrikePrice.values
        # print lStrikePrice,sStrikePrice
        l = np.unique(lStrikePrice)
        s = np.unique(sStrikePrice)
        l.sort()
        s.sort()
        return s[-1 * lengthofstrategy - 1:], l[:lengthofstrategy + 1]
    else:
        raise ValueError('Unknown params: %s' % outputtype)


class IVXTable(object):

    def __init__(self, rtOPdf, current):
        self.Table = self.calIVXTable(rtOPdf, current)
        pass

    def getIVXX(self, X=30):
        """
        cal IVX30 days
        :param X: the duration of IVX
        :return: IVX with X day period
        """
        if X == 30:
            return self.CalIVX30()
        else:
            raise ValueError('Unsupported IVX : %d' % X)

    def calIVXTable(self, rtOPdf, current):

        temp = {int(YM): self.CalIVX(df, current)
                for YM, df in rtOPdf.groupby('YM')}

        return pd.DataFrame(temp).T

    def CalIVX(self, rtOPdf, current, selfcal=False, **fixedparas):
        """
        Cal IVX
        :param rtOPdf: real-time option dataframe
        :param current: current price
        :param selfcal: bool, for whether use self calculation or ext calculator
        :param fixedparas: None
        :return: dict : {'IVX_Call': 1,'IVX_Mean': 2,'IVX_Put':3}
        """
        di = {}

        for OT, df in rtOPdf.groupby('OptionType'):
            s, l = findnearone(
                current, df, outputtype='nearest', lengthofstrategy=3)
            pricelist = np.append(s, l)
            CallSC = df[df.StrikePrice.isin(pricelist)]
            if selfcal:
                # from ..OF1 import ImpliedVolatility as IV_func

                """
                r = fixedparas['RiskfreeRate']
                t = fixedparas['Time2Maturity']
                rtOPdf_cal = rtOPdf[rtOPdf.OptionContShortName.isin(
                    CallSC.OptionContShortName.values)]
                for row, df in rtOPdf_cal.iterrows():
                    cp=df['BuyPrice']
                    IV_func().ImpliedVolatility_OlD(s, k, r, t, cp, cp_sign, g)
                """
                raise ValueError('This function is not done yet')
            else:
                sliceddf = rtOPdf[rtOPdf.OptionContShortName.isin(
                    CallSC.OptionContShortName.values)][['IV', 'Vega']]
            sliceddf['VegaWeightedIV'] = sliceddf['IV'] * \
                sliceddf['Vega'] / sliceddf.Vega.sum()
            di['IVX_' + OT] = sliceddf.VegaWeightedIV.sum()
        di['IVX_Mean'] = np.mean(di.values())
        return di

    def CalIVX30(self):
        """
        cal IVX 30
        :return: df dict ; {'IVX_Call_30': 1,'IVX_Mean_30': 2,'IVX_Put_30':3}
        """

        df = self.Table.sort_index().head(2)

        df['Duration'] = [Api.DateAPI.CalMaturity(
            i) * 250 for i in df.index.values]
        d = np.abs(np.diff(np.sqrt(df.Duration.values)))
        dd = [np.abs(np.sqrt(i) - np.sqrt(30)) / d for i in df.Duration.values]
        df['weighted'] = dd[::-1]  # reverse the lsit
        dfdict = {name + '_30': sum(df[name] * df.weighted)
                  for name in ['IVX_Call', 'IVX_Put', 'IVX_Mean']}
        return dfdict


if __name__ == '__main__':
    rtOPdfam = Data.LoadOptionRealTimeDataViaAPI()
    rtOgreefam = Data.LoadGreekRealTimeDataViaAPI()
    rtOgreefam.RequestTimeStamp = np.round(rtOgreefam.RequestTimeStamp)
    rtOPdfam.RequestTimeStamp = np.round(rtOPdfam.RequestTimeStamp)
    close, current = Api.DataAPI.getUnderlyingLastCloseandCurrentUnderlyingPrice()

    CombinedOptiondf = pd.merge(rtOPdfam, rtOgreefam,
                                on=['OptionContShortName', 'RequestTimeStamp', 'Unknown Var', 'LastestPrice',
                                    'StrikePrice'])
    CombinedOptiondf = CombinedOptiondf[CombinedOptiondf['Unknown Var'] == 'M']
    # rtOPdf=rtOPdf[rtOPdf.YM==1806]
    CombinedOptiondf = CombinedOptiondf[
        ['IV', 'Vega', 'OptionType', 'StrikePrice', 'OptionContShortName', 'YM', 'RequestTimeStamp']]
    # ---------------
    print IVXTable(CombinedOptiondf, current).getIVXX(X=30)
