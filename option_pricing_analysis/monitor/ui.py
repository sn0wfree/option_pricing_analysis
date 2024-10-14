# coding=utf-8
import warnings
from tkinter import ttk, Tk

import numpy as np

# from ClickSQL import BaseSingleFactorTableNode

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置简黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决‘-’bug


class OptionMonitorUIHelper(Tk):
    @staticmethod
    def get_height_width(labelframe):
        width = labelframe.winfo_width()
        height = labelframe.winfo_height()
        return height, width

    @staticmethod
    def display_dataframe_in_treeview(treeview, dataframe, frame_height=100, frame_width=200):
        # 清空现有的内容
        treeview.delete(*treeview.get_children())

        # 定义列，并调整每列宽度
        min_col_width = 1  # 设置最小列宽
        max_col_width = frame_width  # 设置最大列宽

        # 设置列
        treeview["columns"] = list(dataframe.columns)
        # treeview["show"] = "headings"  # 只显示标题
        total_width_used = 0
        # 定义每列
        for column in dataframe.columns:
            treeview.heading(column, text=column)
            # 根据内容调整列宽
            # 根据内容和列名长度动态设置列宽，并限制在min_col_width和max_col_width之间
            content_width = dataframe[column].astype(str).map(len).max() * 15
            content_width = 15 if np.isnan(content_width) else content_width

            header_width = len(column) * 15
            print(column, content_width, max_col_width)
            col_width = max(min(content_width, max_col_width), header_width, min_col_width)

            treeview.column(column, anchor="center", width=col_width, stretch=False)

            total_width_used += col_width

            # 如果总宽度超出设定的frame宽度，则停止添加列
            # if total_width_used >= frame_width:
            #     break

        # 插入行数据
        for index, row in dataframe.iterrows():
            treeview.insert("", "end", values=list(row))

        return treeview.winfo_width(), treeview.winfo_height()

    def __init__(self, title='期权交易系统', default_size=(400, 300)):
        super().__init__()
        # 创建主窗口

        self.title(title)
        # 设置窗口最小大小
        width, height = default_size

        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()

        geometry = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        self.geometry(geometry)

        self.minsize(width=width, height=height)

    def setup_grid(self, rows=2, columns=2):
        # 使用grid布局管理器，并设置权重
        for i in range(rows):
            self.grid_rowconfigure(i, weight=1)
        for j in range(columns):
            self.grid_columnconfigure(j, weight=1)

    def _create_tk_notebook_(self, parent):
        frame = ttk.Notebook(parent)
        self.tk_frame_show_mode = self._create_tk_frame_(frame)
        frame.add(self.tk_frame_show_mode, text="展示模式")
        self.tk_frame_setup_mode = self._create_tk_frame_(frame)
        frame.add(self.tk_frame_setup_mode, text="设置模式")
        self.tk_frame_System = self._create_tk_frame_(frame)
        frame.add(self.tk_frame_System, text="System")
        frame.place(relx=0.0000, rely=0.0000, relwidth=1.0000, relheight=0.9990)
        return frame

    def _create_tk_frame_(self, parent, relx=0.0000, rely=0.0000, relwidth=1.0000, relheight=0.9990):
        frame = ttk.Frame(parent)
        frame.place(relx=relx, rely=rely, relwidth=relwidth, relheight=relheight)
        return frame

    def _create_tk_table(self, parent, columns={"ID": 154, "字段#1": 231, "字段#2": 385}, relx=0.0000, rely=0.0000,
                         relwidth=0.6425, relheight=0.5866):
        # 表头字段 表头宽度

        tk_table = ttk.Treeview(parent, show="headings", columns=list(columns), )
        for text, width in columns.items():  # 批量设置列属性
            tk_table.heading(text, text=text, anchor='center')
            tk_table.column(text, anchor='center', width=width, stretch=False)  # stretch 不自动拉伸

        tk_table.place(relx=relx, rely=rely, relwidth=relwidth, relheight=relheight)
        return tk_table

    def link_2_grid(self, pos, frame, sticky):
        frame.grid(row=pos[0], column=pos[1], sticky=sticky)

    def run(self):
        self.mainloop()


class OptionMonitorUI(OptionMonitorUIHelper):

    def market_display(self, t_trading_board, quote, row=0, column=0, height=100, width=200, text="行情展示"):
        top_left_frame_quote = ttk.Panedwindow(self)
        top_left_frame_quote.grid(row=row, column=column, sticky='nsew')

        # 期权行情

        # 左上角：行情展示
        left_market_display = ttk.LabelFrame(top_left_frame_quote, text=text)
        left_market_display.grid(row=row, column=column, sticky="nsew")
        left_market_display.rowconfigure(row, weight=1)
        left_market_display.columnconfigure(column, weight=1)

        # 创建Treeview来展示DataFrame
        tree = ttk.Treeview(left_market_display, show="headings", selectmode="browse", height=height)
        tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 行情展示

        self.display_dataframe_in_treeview(tree, t_trading_board, height, width)

        # 配置PanedWindow的sash（分隔条）属性
        # top_left_frame_quote.sash_pad(10)  # 设置分隔条周围的填充
        # top_left_frame_quote.sash_ReliefRaised()  # 设置分隔条的3D效果

        # 指数行情

        # 右上角：行情展示
        right_market_display = ttk.LabelFrame(top_left_frame_quote, text='标的指数展示')
        right_market_display.grid(row=row, column=column + 1, sticky="nsew")
        right_market_display.rowconfigure(row, weight=2)
        right_market_display.columnconfigure(column + 1, weight=2)

        # 创建Treeview来展示DataFrame
        tree = ttk.Treeview(right_market_display, show="headings", selectmode="browse", height=height)
        tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # 行情展示

        self.display_dataframe_in_treeview(tree, quote, height, width)

        return height, width

    def position_display(self, positions_dict, greek, row=1, column=0, height=100, width=200, text="持仓展示"):
        bottom_left_frame_holding = ttk.Panedwindow(self)
        bottom_left_frame_holding.grid(row=row, column=column, sticky='nsew')

        ## 持仓

        position_display = ttk.LabelFrame(bottom_left_frame_holding, text=text)
        position_display.grid(row=0, column=0, sticky="nsew")
        position_display.rowconfigure(0, weight=1)
        position_display.columnconfigure(0, weight=1)

        tree = ttk.Treeview(position_display, show="headings", selectmode="browse", height=height)
        tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.display_dataframe_in_treeview(tree, positions_dict, height, width)

        # Greek计算
        # greeks = calculate_greeks()
        greek_label = ttk.LabelFrame(bottom_left_frame_holding, text='greek')

        greek_label.grid(row=0, column=1, sticky="nsew")

        greek_label.rowconfigure(0, weight=2)
        greek_label.columnconfigure(1, weight=2)

        tree = ttk.Treeview(greek_label, show="headings", selectmode="browse", height=height)

        tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.display_dataframe_in_treeview(tree, greek, height, width)

        ## 画图
        bottom_right_frame_d = ttk.LabelFrame(bottom_left_frame_holding, text='pic')
        bottom_right_frame_d.grid(row=0, column=2, sticky='nsew')

        # 计算工具
        # 这里可以添加你需要的计算工具，例如按钮、输入框等
        # 例如，一个简单的计算按钮
        calculate_button = ttk.Button(bottom_right_frame_d, text="执行计算", command=lambda: print("计算执行"))
        calculate_button.pack(pady=10)
        pass


if __name__ == '__main__':
    pass
