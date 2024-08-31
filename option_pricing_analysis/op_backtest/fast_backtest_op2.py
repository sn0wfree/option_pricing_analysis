# # coding=utf-8
# import pandas as pd
# import numpy as np
# import datetime
# import warnings
# from functools import partial
#
# import akshare as ak
# import numpy as np
# import pandas as pd
#
# warnings.filterwarnings('ignore')
#
# from collections import ChainMap
#
# from CodersWheel.QuickTool.file_cache import file_cache
#
# from option_analysis_monitor import ProcessReport, DerivativesItem, ReportAnalyst, ProcessReportSingle
#
#
# class OptionPortfolio(object):
#
#     @staticmethod
#     def strategy(dt, ym, op_quote_and_info, hedge_size, min_put_level=3, max_cost=200):
#
#         mask = op_quote_and_info['t'] >= 0.01
#
#         dt_mask = op_quote_and_info['dt'] >= 0.01
#
#
#         a_put =
#         fee = a_put['fee'].max()
#         if fee <= max_cost:
#             #         print(fee)
#             a_put['weight'] = (hedge_size / a_put['Delta'] / 100 / a_put['f']).abs()
#             return a_put
#         else:
#             return strategy(portfolio, hedge_size, min_put_level=min_put_level + 1, max_cost=max_cost)
#         pass
#
#     @classmethod
#     def mapping_strategy(cls, op_quote: pd.DataFrame):
#         pass
#
# def load_full_greek_data(name='full_greek_caled_marked_60m.parquet'):
#     full_greek_caled_marked_60m = pd.read_parquet(name)
#     full_greek_caled_marked_60m['contract_code'] = full_greek_caled_marked_60m['contract_code'].astype(str) + '.CFE'
#     full_greek_caled_marked_60m['OTM'] = ((full_greek_caled_marked_60m['f'] - full_greek_caled_marked_60m['k']) *
#                                           full_greek_caled_marked_60m['cp'].replace({'C': -1, 'P': 1}).astype(int)) >= 0
#
#     return full_greek_caled_marked_60m
#
# if __name__ == '__main__':
#     full_greek_caled_marked = load_full_greek_data(name='full_greek_caled_marked.parquet')
#     pass
