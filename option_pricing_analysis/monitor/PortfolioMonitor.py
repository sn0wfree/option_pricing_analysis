# coding=utf-8
import warnings

# from ClickSQL import BaseSingleFactorTableNode

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决‘-’bug

from option_pricing_analysis.monitor.ui import OptionMonitorUI
from option_pricing_analysis.monitor.data_src import PositionHolder, QuoteHolder
from option_pricing_analysis.monitor.calculator import Calculator


class PortfolioMonitor(OptionMonitorUI):

    def init(self, quote_obj, pos_obj, cal_obj):
        # 使用grid布局管理器，并设置权重
        self.setup_grid(rows=1, columns=1)
        ## top_left_frame
        # 获取行情数据
        t_trading_board = quote_obj.get_op_quote_via_cffex(symbol='mo', end_month="ALL")

        realtime_idx = quote_obj.get_idx_minute_quote_via_ak()

        self.market_display(t_trading_board, realtime_idx, row=0, column=0, height=50, width=300)

        ## Greek计算
        greeks = cal_obj.calculate_greeks()

        ## 获取持仓组合
        position_dict = pos_obj.get_current_position()
        self.position_display(position_dict, greeks, row=1, column=0, height=30, width=300)

        # self.greek_calculator_display(greeks, row=1, column=1, height=30, width=150)

        ## 画图区域
        # self.greek_pic_display(row=1, column=2, height=30, width=100)

    pass


if __name__ == '__main__':
    OMT = PortfolioMonitor(default_size=(1920, 1080))

    OMT.init(QuoteHolder, PositionHolder, Calculator)
    OMT.run()

    print(1)
    pass
