# coding=utf-8
from option_pricing_analysis.Model.BSM import ImpliedVolatility, OptionPricing


def map_cal_option_iv(df):
    ivfunc = ImpliedVolatility(pricing_model=OptionPricing())

    func = ivfunc.implied_volatility_brent

    for stock_price, strike_price, risk_free, t, cp_fee, cp_sign, g in df.values:
        # iv
        yield func(stock_price, strike_price, risk_free, t, cp_fee, cp_sign, g)


# def summary_data(df):
#     iv_series = map_cal_option_iv(df)
#     df['iv'] = iv_series
#     return df


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_excel('C:\\Users\\user\\Documents\\GitHub\\option_pricing_analysis\\option_pricing_analysis\\济川转债可转债债券历史行情(1).xlsx')
    cols = ['正股价格', '转股价', '到期时间', '权证价格']

    df['risk_free'] = 0.02
    df['g'] = 0
    df['cp_sign'] = 1
    named_cols = ['正股价格', '转股价', 'risk_free', '到期时间', '权证价格', 'cp_sign', 'g']
    df['iv'] = list(map_cal_option_iv(df[named_cols]))
    df.to_csv('result.csv')
    pass
