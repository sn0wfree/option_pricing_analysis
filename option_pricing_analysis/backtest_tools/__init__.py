# coding=utf-8

import akshare as ak
import backtrader as bt
import pandas as pd
from CodersWheel.QuickTool.file_cache import file_cache


@file_cache(enable_cache=True, granularity='d')
def get_idx_quote(code):
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol=code).set_index('date')
    stock_zh_index_daily_df.index = pd.to_datetime(stock_zh_index_daily_df.index)
    return stock_zh_index_daily_df


class CereBroBase(object):
    def __init__(self, df_ohlc, start_dt=None, end_dt=None):
        self._setup_state = False
        start_dt = df_ohlc.index.min() if start_dt is None else start_dt
        end_dt = df_ohlc.index.max() if end_dt is None else end_dt

        self._cerebro = bt.Cerebro()

        data = bt.feeds.PandasData(dataname=df_ohlc, fromdate=pd.to_datetime(start_dt), todate=pd.to_datetime(end_dt))

        # 加载交易数据
        self._cerebro.adddata(data, name='quote')

    def setup(self, init_cash=100000.0, commission=0.001, position_percents=50):
        # 设置投资金额100000.0
        self._cerebro.broker.setcash(init_cash)
        # 设置佣金为0.001,
        self._cerebro.broker.setcommission(commission=commission)
        # 设置交易模式
        self._cerebro.addsizer(bt.sizers.PercentSizer, percents=position_percents)

        self._setup_state = True

    def run_strategy(self, TradingStrategy=None, **kwargs):
        if not self._setup_state:
            raise ValueError('setup is not completed!')

        # 为Cerebro引擎添加策略
        self._cerebro.addstrategy(TradingStrategy)

        # 引擎运行前打印期出资金
        print('组合期初资金: %.2f' % self._cerebro.broker.getvalue())

        self._cerebro.run(**kwargs)

        # 引擎运行后打期末资金
        print('组合期末资金: %.2f' % self._cerebro.broker.getvalue())

    def run_signal(self, DefaultSignal=None, signal_type=bt.SIGNAL_LONG, **kwargs):

        if not self._setup_state:
            raise ValueError('setup is not completed!')

        # 为Cerebro引擎添加策略
        self._cerebro.add_signal(signal_type, DefaultSignal)

        # 引擎运行前打印期出资金
        print('组合期初资金: %.2f' % self._cerebro.broker.getvalue())

        self._cerebro.run(**kwargs)

        # 引擎运行后打期末资金
        print('组合期末资金: %.2f' % self._cerebro.broker.getvalue())

    def plot(self, **kwargs):
        self._cerebro.plot(**kwargs)


class SignalPandasData(bt.feeds.PandasData):
    params = (('Signal', 'Signal'),)
    # params = (('Signal', -1),)


class SignalGenerator(bt.Indicator):
    lines = ('Signal',)

    def __init__(self):
        self.lines.Signal = self.datas[1].Signal


class SMACloseSignal(bt.Indicator):
    lines = ('signa',)
    params = (('period', 10),)

    def __init__(self):
        s1 = self.data.close - bt.indicators.SMA(period=self.p.period)
        # print(s1)
        self.lines.signa = s1


# class ATRSMACloseSignal(bt.Indicator):
#     lines = ('signa',)
#     params = (('period', 30),)
#
#     def __init__(self):
#         self.atr = bt.indicators.ATR(period=self.p.period)
#
#         self.lines.signa = self.atr >= 70


if __name__ == '__main__':
    # 添加行情数据
    df_ohlc = get_idx_quote(code="sh000852")

    # 假设你有一个包含信号的DataFrame
    data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Signal': [1, 0, -1]  # 1代表买入，-1代表卖出，0代表无信号
    }).set_index('Date')
    data.index = pd.to_datetime(data.index)
    data = data.reindex(index=df_ohlc.index).fillna(0)

    Ch = CereBroBase(df_ohlc, start_dt='2021-01-01', end_dt=None)

    signal_data = SignalPandasData(dataname=data)
    Ch._cerebro.adddata(signal_data, 'Signal')

    Ch.setup(init_cash=100000.0, commission=0.0005, position_percents=50)

    # 为Cerebro引擎添加策略
    Ch.run_signal(SignalGenerator, signal_type=bt.SIGNAL_LONG)
    # Ch.run_signal(None, signal_data=data, signal_type=bt.SIGNAL_LONG)

    Ch.plot()

    print(1)
    pass
