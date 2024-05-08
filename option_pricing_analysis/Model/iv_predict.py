# coding=utf-8
import datetime

import pandas as pd
from ClickSQL import BaseSingleFactorTableNode

from option_pricing_analysis.Model.BSM import ImpliedVolatility, OptionPricing
from option_pricing_analysis.Model.SARB import SABRfromQuantlib
from option_pricing_analysis.strategy.stratrgy_1 import get_market_index_close_data_via_ak, MOQuoteQuickMatrix, \
    load_lastdel_multi

ZERO = 2.e-9


class IVPredict(object):
    def __init__(self, ):
        self.iv_func = ImpliedVolatility(pricing_f=OptionPricing, method='BSPricing').iv_brent
        pass

    @staticmethod
    def chunk_partial_data(K, f, n=200):
        less_K = min(K[K <= f].sort_values().tail(n))
        more_K = max(K[K >= f].sort_values().head(n).values)
        return less_K, more_K

    def single_bs_iv(self, f, k, r, t, fee, cp, g, method='BSPricing'):
        return self.iv_func(f, k, r, t, fee, cp, g, method=method)

    def bs_iv_cross(self, f, K, r, t, FEE, cp, g, method='BSPricing'):
        for k, fee in zip(K, FEE):
            iv, diff = self.single_bs_iv(f, k, r, t, fee, cp, g, method=method)
            yield k, iv, diff

    def sabr_iv_cross(self, f, K, r, t, iv, cons=({'type': 'ineq', 'fun': lambda x: 0.999999 - x[1] - ZERO},  # alpha
                                                  {'type': 'ineq', 'fun': lambda x: x[1] - ZERO},  # beta
                                                  {'type': 'ineq', 'fun': lambda x: x[2] - ZERO},  # mu
                                                  {'type': 'ineq', 'fun': lambda x: x[3] - ZERO},  # rho
                                                  {'type': 'ineq', 'fun': lambda x: 0.99999 - x[3] - ZERO},  # rho
                                                  )):
        sabr_q = SABRfromQuantlib(K, f, t, iv)
        res = sabr_q.fit(params=(0.02, 0.95, 0.01, 0.02), cons=cons)
        return res

    def run_iv_surface(self, f, K, r, t, FEE, cp, g,
                       cons=({'type': 'ineq', 'fun': lambda x: 0.999999 - x[1] - ZERO},  # alpha
                             {'type': 'ineq', 'fun': lambda x: x[1] - ZERO},  # beta
                             {'type': 'ineq', 'fun': lambda x: x[2] - ZERO},  # mu
                             {'type': 'ineq', 'fun': lambda x: x[3] - ZERO},  # rho
                             {'type': 'ineq', 'fun': lambda x: 0.99999 - x[3] - ZERO},  # rho
                             )):
        bs_iv_df = pd.DataFrame(self.bs_iv_cross(f, K, r, t, FEE, cp, g, method='BSPricing'),
                                columns=['strike', 'bs_iv', 'residual'])

        sabr_iv = self.sabr_iv_cross( f, K, r, t, bs_iv_df['bs_iv'], cons=cons)

        bs_iv_df['sabr_iv'] = sabr_iv

        bs_iv_df['fee'] = FEE.values
        bs_iv_df['cp'] = cp
        return bs_iv_df

    @staticmethod
    def run(quote, lastdel_multi, op_quote, op_dt_col, op_ym_col, op_cp_col):
        IVp = IVPredict()
        for (dt, ym, cp), op_df in op_quote.groupby([op_dt_col, op_ym_col, op_cp_col]):
            f = quote.loc[dt, :].values[0]

            contract_code_list = op_df['contract_code'].unique().tolist()
            delta = lastdel_multi.loc[contract_code_list, :]['EXE_DATE'].unique()[0] - pd.to_datetime(dt)
            t = delta.days / 365

            FEE = op_df['close']

            K = op_df['strike']

            sabr_iv = IVp.run_iv_surface(f, K, 0.015, t, FEE, cp, g=0, )

            yield sabr_iv


if __name__ == '__main__':
    zz1000 = get_market_index_close_data_via_ak(None, ['000852.SH'], ['中证1000'], start="2023-12-31",
                                                end=datetime.datetime.today().strftime("%Y-%m-%d"), period='D',
                                                comm_fut_code_dict={'NH0100.NHF': 'NHCI'})

    node = BaseSingleFactorTableNode('clickhouse://default:Imsn0wfree@47.104.186.157:8123/system')

    moqqm = MOQuoteQuickMatrix(node, start='2024-01-01', end='2024-12-31')

    contract_list = moqqm._raw['contract_code'].unique().tolist()

    lastdel_multi = load_lastdel_multi(
        "C:\\Users\\linlu\\Documents\\GitHub\\option_pricing_analysis\\option_pricing_analysis\\strategy\\lastdel_multi.xlsx").set_index(
        '委托合约')

    IVp = IVPredict()

    op_quote = moqqm._raw

    c = list(IVp.run(zz1000, lastdel_multi, op_quote, 'trade_date', 'ym', 'cp'))

    # dt = '2024-04-26'
    # f = zz1000.loc[dt, :].values[0]
    # df = moqqm._raw
    #
    # dt_mask = df['trade_date'] == '2024-04-26'
    # ym_mask = df['ym'] == '2406'
    # cp_mask = df['cp'] == 'p'
    #
    # delta = lastdel_multi.loc[df['contract_code'], :]['EXE_DATE'].unique()[0] - pd.to_datetime(dt)
    #
    # t = delta.days / 365
    #
    # K =
    #
    # IVp.run_iv_surface(f, K, 0.015, t, FEE, cp, g=0,)

    print(1)
    pass
