# coding=utf-8
import akshare as ak
from ClickSQL import BaseSingleFactorTableNode

from option_pricing_analysis.Model.BSM import ImpliedVolatility, OptionPricing


def idx_quote(symbol='sh000852'):
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol=symbol)
    return stock_zh_index_daily_df


def map_cal_option_iv(option_quote):
    iv_func = ImpliedVolatility(pricing_f=OptionPricing, method='MCPricing').implied_volatility_brent
    iv = [iv_func(s, x, r, t, cp_fee, cp_sign, g) for s, x, r, t, cp_fee, cp_sign, g in option_quote.values]
    return iv


def get_option_daily_quote(conn, dt='2024-04-09'):
    sql = f"select contract_code,trade_date,open,high,low,close,delta from quote.derivative_quote where startsWith(contract_code,'MO') and trade_date ='{dt}' "
    df = conn(sql)
    return df


if __name__ == '__main__':
    conn = BaseSingleFactorTableNode("clickhouse://default:Imsn0wfree@47.104.186.157:8123/system")

    # idx_quote = idx_quote(symbol='sh000852')
    # op_quote = get_option_daily_quote(conn, dt='2024-04-09')

    # option_sse_minute_sina_df = ak.option_sse_minute_sina(symbol="mo2405")
    print(1)

    pass
