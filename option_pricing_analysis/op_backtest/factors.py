# coding=utf-8
import akshare as ak
import numpy as np
import pandas as pd
import talib
from CodersWheel.QuickTool.file_cache import file_cache
from scipy import stats


### 快速 + 慢速 移动平均线MA


class FactorOperatorSingle(object):
    @staticmethod
    def SMA(df_ohlc, timeperiod=21):
        return talib.SMA(df_ohlc['close'], timeperiod=timeperiod)

    @staticmethod
    def MACD(df_ohlc, fastperiod=12, slowperiod=26, signalperiod=9):
        macd, macdsignal, macdhist = talib.MACD(df_ohlc['close'], fastperiod=fastperiod, slowperiod=slowperiod,
                                                signalperiod=signalperiod)
        return macd, macdsignal, macdhist

    @staticmethod
    def RSI(df_ohlc, rsi_period=12, ):
        return talib.RSI(df_ohlc['close'], timeperiod=rsi_period)

    @staticmethod
    def BBANDS(df_ohlc, timeperiod=16, nbdevup=-4e+37, nbdevdn=-4e+37, matype=0, ):
        # matype =0 (Simple Moving Average)

        # Calculate Bollinger Bands using pandas_ta
        upperband, middleband, lowerband = talib.BBANDS(df_ohlc['close'], timeperiod=timeperiod, nbdevup=nbdevup,
                                                        nbdevdn=nbdevdn, matype=matype)
        return upperband, middleband, lowerband

    @staticmethod
    def ADX(df_ohlc, timeperiod=16, ):
        return talib.ADX(df_ohlc['high'], df_ohlc['low'], df_ohlc['close'], timeperiod=timeperiod)

    @staticmethod
    # @file_cache(enable_cache=True,granularity='d' )
    def SNR(df_ohlc, timeperiod=30, ):
        # SNR（Signal-to-noise）
        pct_abs = df_ohlc['close'].diff(timeperiod).abs()

        pct_1_abs_avg = df_ohlc['close'].diff(1).abs().rolling(timeperiod).sum()
        snr = pct_abs / pct_1_abs_avg

        return snr.to_frame("SNR")

    @staticmethod
    def VHF(df_ohlc, timeperiod=30, ):
        # VHF（Vertical Horizontal Filter）
        high = df_ohlc['high'].rolling(timeperiod).max()
        low = df_ohlc['low'].rolling(timeperiod).min()
        high_low = high - low

        pct_1_abs_avg = df_ohlc['close'].diff(1).abs().rolling(timeperiod).sum()
        vhf = high_low / pct_1_abs_avg

        return vhf.to_frame('VHF')

    @staticmethod
    @file_cache(enable_cache=True, granularity='d')
    def RegSlope(df_ohlc, timeperiod=30, ):
        # regression slope
        idx = df_ohlc.index
        h = []
        for s, e in zip(idx[:-(timeperiod - 1)], idx[timeperiod - 1:]):
            mask = (df_ohlc.index >= s) & (df_ohlc.index <= e)

            # 假设我们有一组观测数据
            x = np.arange(0, timeperiod, 1) + 1  # 自变量
            y = df_ohlc[mask]['close'].values  # 因变量

            # 使用scipy的linregress函数进行线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            h.append((e, np.abs(slope) / std_err))

        res = pd.DataFrame(h, columns=['end_dt', 'RegSlope']).set_index('end_dt')
        return res

    @staticmethod
    @file_cache(enable_cache=True, granularity='d')
    def ARSlope(df_ohlc, timeperiod=30, lag=1):
        from statsmodels.api import OLS
        # regression slope
        df_ohlc_close = df_ohlc[['close']].copy(deep=True)

        lag_beta = []

        for i in range(1, lag + 1, 1):
            df_ohlc_close[f'close_lag{i}'] = df_ohlc_close['close'].shift(i)
            lag_beta.append(f'close_lag{i}')

        formula = 'close ~ ' + ' + '.join(lag_beta)
        df_ohlc_close = df_ohlc_close.dropna()
        h = []
        idx = df_ohlc_close.index
        for s, e in zip(idx[:-(timeperiod - 1)], idx[timeperiod - 1:]):
            mask = (df_ohlc_close.index >= s) & (df_ohlc_close.index <= e)

            # 假设我们有一组观测数据

            m = OLS.from_formula(formula, df_ohlc_close[mask]).fit()

            params = pd.DataFrame(m.params).T

            params.index = [e]

            h.append(params)

        res = pd.concat(h)
        return res


class TrendFactors(FactorOperatorSingle):

    @classmethod
    def trend_2_ma(cls, df_ohlc, period_slow=21, period_fast=9, return_raw_df=False):
        # 当快速值高于慢值时，表明呈上升趋势。
        # Calculate Moving Averages (fast and slow) using pandas_ta
        df_ohlc[f'MA_{period_fast}'] = cls.SMA(df_ohlc, timeperiod=period_fast)
        df_ohlc[f'MA_{period_slow}'] = cls.SMA(df_ohlc, timeperiod=period_slow)

        # Determine the trend based on Moving Averages
        up = (df_ohlc[f'MA_{period_fast}'] > df_ohlc[f'MA_{period_slow}']) * 1
        down = (df_ohlc[f'MA_{period_fast}'] < df_ohlc[f'MA_{period_slow}']) * -1
        df_ohlc['Trend_2_ma'] = (up + down).fillna('0')

        if return_raw_df:

            return df_ohlc

        else:
            return df_ohlc['Trend_2_ma']

    @classmethod
    def trend_macd_ma(cls, df_ohlc, ma_period=50, macd_fast=12, macd_slow=26, macd_signal=9, return_raw_df=False):
        ### MA + MACD
        # 这两个指标应该保持一致。这意味着要识别上升趋势，收盘价应高于移动平均线，MACD 线应高于 MACD 信号。
        #  Calculate MACD using pandas_ta
        macd_name = f'{macd_fast}_{macd_slow}_{macd_signal}'

        df_ohlc[f'MACD{macd_name}'], df_ohlc[f'MACDsignal{macd_name}'], df_ohlc[f'MACDhist{macd_name}'] = cls.MACD(
            df_ohlc, fastperiod=macd_fast, slowperiod=macd_slow,
            signalperiod=macd_signal)

        #  Calculate Moving Average
        ma_name = f'MA{ma_period}'

        df_ohlc[ma_name] = cls.SMA(df_ohlc, timeperiod=ma_period)

        up_ma = df_ohlc['close'] > df_ohlc[ma_name]
        down_ma = df_ohlc['close'] < df_ohlc[ma_name]

        up_macd = df_ohlc[f'MACD{macd_name}'] > df_ohlc[f'MACDsignal{macd_name}']
        down_macd = df_ohlc[f'MACD{macd_name}'] < df_ohlc[f'MACDsignal{macd_name}']

        up = (up_ma & up_macd) * 1
        down = (down_ma & down_macd) * -1

        trend = up + down

        df_ohlc['Trend_macd_ma'] = trend

        if return_raw_df:

            return df_ohlc

        else:

            return df_ohlc['Trend_macd_ma']

    @classmethod
    def trend_rsi_ma(cls, df_ohlc, rsi_period=14, ma_fast=9, ma_slow=21, return_raw_df=False):
        ### RSI + 快速和慢速移动平均线
        # 快速 MA 应高于慢速 MA，RSI > 50 以确定上升趋势和下降趋势的相反趋势。
        # Calculate RSI using pandas_ta

        df_ohlc[f'RSI{rsi_period}'] = cls.RSI(df_ohlc, rsi_period=rsi_period)

        # Calculate Moving Averages (14-day and 50-day) using pandas_ta
        df_ohlc[f'MA_{ma_fast}'] = cls.SMA(df_ohlc, timeperiod=ma_fast)
        df_ohlc[f'MA_{ma_slow}'] = cls.SMA(df_ohlc, timeperiod=ma_slow)

        # Determine the trend based on RSI and Moving Averages
        # 快速 MA 应高于慢速 MA，RSI > 50 以确定上升趋势

        up = df_ohlc[f'MA_{ma_fast}'] > df_ohlc[f'MA_{ma_slow}']
        down = df_ohlc[f'MA_{ma_fast}'] < df_ohlc[f'MA_{ma_slow}']
        rsi_lg_50 = df_ohlc[f'RSI{rsi_period}'] > 50
        rsi_less_50 = df_ohlc[f'RSI{rsi_period}'] < 50

        m1 = (up * rsi_lg_50) * 1
        m2 = (down * rsi_less_50) * -1

        df_ohlc['Trend_rsi_ma'] = m1 + m2

        if return_raw_df:

            return df_ohlc

        else:

            return df_ohlc['Trend_rsi_ma']

    @classmethod
    def trend_bbands_rsi(cls, df_ohlc, bbands_period=5, bbands_std=2, rsi_period=14, return_raw_df=False):
        ### 布林带 + RSI
        # 当价格高于中间布林带且 RSI 高于 50 时，我们有一个上升趋势。
        # Calculate RSI using pandas_ta
        df_ohlc[f'RSI{rsi_period}'] = cls.RSI(df_ohlc, rsi_period=rsi_period)

        # Calculate Bollinger Bands using pandas_ta

        std = df_ohlc['close'].rolling(bbands_period).std()
        df_ohlc[f'BB_middle_{bbands_period}'] = cls.SMA(df_ohlc, timeperiod=bbands_period)
        df_ohlc[f'BB_higher_{bbands_period}'] = df_ohlc[f'BB_middle_{bbands_period}'] + bbands_std * std
        df_ohlc[f'BB_lower_{bbands_period}'] = df_ohlc[f'BB_middle_{bbands_period}'] - bbands_std * std

        # Determine the trend based on Bollinger Bands and RSI
        # 当价格高于中间布林带且 RSI 高于 50 时，上升趋势。

        rsi_lg_50 = df_ohlc[f'RSI{rsi_period}'] > 50
        rsi_less_50 = df_ohlc[f'RSI{rsi_period}'] < 50

        up = df_ohlc[f'close'] > df_ohlc[f'BB_middle_{bbands_period}']
        down = df_ohlc[f'close'] < df_ohlc[f'BB_middle_{bbands_period}']

        m1 = (up * rsi_lg_50) * 1
        m2 = (down * rsi_less_50) * -1

        df_ohlc['Trend_bbands_rsi'] = m1 + m2

        if return_raw_df:

            return df_ohlc

        else:

            return df_ohlc['Trend_bbands_rsi']

    @classmethod
    def trend_adx_ma(cls, df_ohlc, adx_period=14, fast_ma_period=14, slow_ma_period=50, return_raw_df=False):
        #### 慢速和快速移动平均线 + ADX
        # 通过结合 ADX 和移动平均线，当 ADX 高于 25（表明强劲趋势）且快速 MA 高于慢速 MA 时，我们将确定上升趋势。
        # Calculate ADX using pandas_ta
        df_ohlc[f'ADX{adx_period}'] = cls.ADX(df_ohlc, timeperiod=adx_period, )

        # Calculate Moving Averages (14-day and 50-day) using pandas_ta
        df_ohlc[f'MA_{fast_ma_period}'] = cls.SMA(df_ohlc, timeperiod=fast_ma_period)
        df_ohlc[f'MA_{slow_ma_period}'] = cls.SMA(df_ohlc, timeperiod=slow_ma_period)

        # Determine the trend based on ADX and Moving Averages
        # 当 ADX 高于 25（表明强劲趋势）且快速 MA 高于慢速 MA 时，上升趋势

        up = df_ohlc[f'MA_{fast_ma_period}'] > df_ohlc[f'MA_{slow_ma_period}']
        down = df_ohlc[f'MA_{fast_ma_period}'] < df_ohlc[f'MA_{slow_ma_period}']

        adx_lg_50 = df_ohlc[f'ADX{adx_period}'] > 25
        adx_less_50 = df_ohlc[f'ADX{adx_period}'] < 25

        m1 = (up * adx_lg_50) * 1
        m2 = (down * adx_less_50) * -1

        df_ohlc['Trend_adx_ma'] = m1 + m2

        if return_raw_df:
            return df_ohlc
        else:
            return df_ohlc['Trend_adx_ma']

    ### Ichimoku Cloud + MACD
    # 当价格高于 Ichimoku Cloud 且 MACD 高于信号线时，我们确定上升趋势。
    # @classmethod
    # def calculate_trend_ichimoku_macd(cls, df_ohlc, macd_fast=12, macd_slow=26, macd_signal=9, tenkan=9, kijun=26, senkou=52):
    #     # Calculate Ichimoku Cloud components using pandas_ta
    #     df_ichimoku = df_ohlc.ta.ichimoku(tenkan, kijun, senkou)[0]
    #
    #     # Extract Ichimoku Cloud components
    #     df_ohlc['Ichimoku_Conversion'] = df_ichimoku[f'ITS_{tenkan}']  # Tenkan-sen (Conversion Line)
    #     df_ohlc['Ichimoku_Base'] = df_ichimoku[f'IKS_{kijun}']  # Kijun-sen (Base Line)
    #     df_ohlc['Ichimoku_Span_A'] = df_ichimoku[f'ITS_{tenkan}']  # Senkou Span A
    #     df_ohlc['Ichimoku_Span_B'] = df_ichimoku[f'ISB_{kijun}']  # Senkou Span B
    #
    #     # Calculate MACD using pandas_ta
    #     df_ohlc.ta.macd(close='close', fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
    #
    #     # Determine the trend based on Ichimoku Cloud and MACD
    #     # 当价格高于 Ichimoku Cloud 且 MACD 高于信号线时，我们确定上升趋势。
    #     def identify_trend(row):
    #         if row['close'] > max(row['Ichimoku_Span_A'], row['Ichimoku_Span_B']) and row['MACD_12_26_9'] > row[
    #             'MACDs_12_26_9']:
    #             return 1
    #         elif row['close'] < min(row['Ichimoku_Span_A'], row['Ichimoku_Span_B']) and row['MACD_12_26_9'] < row[
    #             'MACDs_12_26_9']:
    #             return -1
    #         else:
    #             return 0
    #
    #     df_ohlc['Trend'] = df_ohlc.apply(identify_trend, axis=1)
    #     return df_ohlc['Trend']

    @classmethod
    def trend_SNR_VHF_SLOPE2_avg(cls, df_ohlc, timeperiod=4 * 4 * 5, lag=1, rolling=4 * 4 * 20):
        AR_OLS_beta = cls.ARSlope(df_ohlc, timeperiod=timeperiod, lag=lag)
        RegSlope = cls.RegSlope(df_ohlc, timeperiod=timeperiod)

        VHF = cls.VHF(df_ohlc, timeperiod=timeperiod)
        SNR = cls.SNR(df_ohlc, timeperiod=timeperiod)

        lag_beta = [f'close_lag{i}' for i in range(1, lag + 1, 1)]

        required_cols = ['SNR', 'VHF', 'RegSlope', ] + lag_beta

        merged = pd.concat([SNR, VHF, RegSlope, AR_OLS_beta], axis=1)[required_cols].dropna()
        z_scored_merged = (merged - merged.rolling(rolling).mean()) / merged.rolling(rolling).std() + 1

        return z_scored_merged


if __name__ == '__main__':
    # data = 'full_greek_caled_marked20241209.parquet'
    # full_greek_caled_marked = pd.read_parquet(data)
    #
    # put_mask = full_greek_caled_marked['cp'] == 'P'
    #
    # Delta = full_greek_caled_marked[put_mask].pivot_table(index='dt', columns='contract_code', values='Delta')
    #
    # f = full_greek_caled_marked[put_mask].pivot_table(index='dt', values='f')
    # k = full_greek_caled_marked[put_mask].pivot_table(index='dt', columns='contract_code', values='k')
    # fee = full_greek_caled_marked[put_mask].pivot_table(index='dt', columns='contract_code', values='fee')
    # cost = full_greek_caled_marked[put_mask].pivot_table(index='dt', columns='contract_code', values='cost')

    # df_ohlc = pd.read_excel('zz1000_15m_v1.xlsx').set_index('dt')
    # df_ohlc.index = pd.to_datetime(df_ohlc.index, format='%Y-%m-%d %H:%M:%S')

    @file_cache(enable_cache=True, granularity='d')
    def get_idx_quote(code):
        stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol=code).set_index('date')
        stock_zh_index_daily_df.index = pd.to_datetime(stock_zh_index_daily_df.index)
        return stock_zh_index_daily_df


    code = 'sh000016'

    df_ohlc = get_idx_quote(code)

    res = TrendFactors.trend_SNR_VHF_SLOPE2_avg(df_ohlc, timeperiod=30, lag=1)

    res['close'] = df_ohlc['close']
    res.dropna().to_excel(f'res_trend_{code}.xlsx')

    df_ohlc = TrendFactors.trend_2_ma(df_ohlc, period_slow=21, period_fast=9, return_raw_df=True)
    # print(df_ohlc['RSI12'])
    #
    df_ohlc = TrendFactors.trend_macd_ma(df_ohlc, ma_period=50, macd_fast=12, macd_slow=26, macd_signal=9,
                                         return_raw_df=True)
    # # print(df_ohlc['SMA12'])
    #
    df_ohlc = TrendFactors.trend_rsi_ma(df_ohlc, rsi_period=14, ma_fast=9, ma_slow=21, return_raw_df=True)
    #
    df_ohlc = TrendFactors.trend_bbands_rsi(df_ohlc, bbands_period=5, bbands_std=2, rsi_period=14, return_raw_df=True)
    #
    df_ohlc = TrendFactors.trend_adx_ma(df_ohlc, adx_period=14, fast_ma_period=14, slow_ma_period=50,
                                        return_raw_df=True)
    #
    df_ohlc.to_excel(f'df_ohlc_{code}.xlsx')
    print(df_ohlc)

    pass
