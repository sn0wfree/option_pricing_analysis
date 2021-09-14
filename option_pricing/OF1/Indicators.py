# -*- coding: utf8 -*-
import numpy as np
import scipy.stats as sps
try:
    from ..APICenter import API
    from ..DataCenter import DataCenter
    from ..OF1 import OptionPricing as OP
except ValueError:
    import sys
    sys.path.append('../')
    from APICenter import API
    from DataCenter import DataCenter
    from OF1 import OptionPricing as OP


NORM_CDF = sps.norm.cdf
NORM_PDF = sps.norm.pdf


Api = API()
Data = DataCenter()

# load function in current path
IV_LOWER_BOUND = 1e-8

# ----------------------

###################################################################
# K : option strike price
# S : price of the underlying asset
# r : risk-free interest rate, 0.03 as 3%
# sigma : asset volatility or implied volatility, 0.25 as 25%
# T : days before expiry, 31/365 as 31 days

# https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model#Black-Scholes_formula
# In practice, some sensitivities are usually quoted in scaled-down terms,
# to match the scale of likely changes in the parameters. For example,
# rho is often reported divided by 10,000 (1 basis point rate change),
# vega by 100 (1 vol point change),
# theta by 365 or 252
# (1 day decay based on either calendar days or trading days per year).


class Sensibility(object):
    """
    # Theta 时间de导数
    # rho   利率的导数
    # Delta是option价值对于股票的导数，
    # gamma是delta对于股票的导数
    # vega是option价格对于波动性的导数
    # --------------
    # theta，rho可以跨不同股票相加
    # delta，gamma只能相同股票的相加
    # vega可以同股票相加，但如果需要精确的exposure应该用vega map，
    # 也就是每一个股票的每一个option maturity的每一个strike都有自己的vega。
    """

    # Delta（Δ），Gamma（γ），Theta（θ），Vega（ν），Rho（ρ）
    def __init__(self, Model='Black-Sholes', OptionStyle='European'):
        self.Model = Model
        self.OptionStyle = OptionStyle

        pass

    def getGreeks(self, UnderlyingPrice,  # UnderlyingPrice
                  Strike,                 # StrikePrice
                  Volatility,             # volatility
                  Time2Maturity,          # Time2Maturity
                  DividendYield,          # DividendYield
                  RiskFreeRate,           # RiskFreeRate
                  OptionType):            # OptionType
        # return Delta, Gamma, Vega, Theta, Rho
        s, k, sigma, t, g, r = UnderlyingPrice, Strike, Volatility, Time2Maturity, DividendYield, RiskFreeRate
        Delta, Gamma, Vega, Theta, Rho = self.Delta(s, k, r, g, sigma, t, OptionType), self.Gamma(s, k, r, sigma, t, OptionType), self.Vega(
            s, k, r, sigma, t, OptionType), self.Theta(s, k, r, sigma, t, OptionType), self.Rho(s, k, r, sigma, t, OptionType)
        return Delta, Gamma, Vega, Theta, Rho

    def BlackScholes_d1(self, s, k, r, sigma, t, g=0):
        """
        This method is to calcuate the d1 which is the first d in the Black-Scholes Model.
        This method should be run firstly before the BlackScholes_d2
        :param s: UnderlyingPrice
        :param k: StrikePrice
        :param r: RiskFreeRate
        :param sigma:  Volatility
        :param t:   Time2Maturity
        :param g:  DividendYield
        :return:  d1
        """
        d1 = (np.log(s / k) + ((r - g) + 0.5 * (sigma)**2)
              * (t)) / (sigma * np.sqrt(t))
        return d1

    def BlackScholes_d2(self, S, K, r, sigma, T):
        """
        This method is to cal the d2 which is the second d in the Black-Scholes Model.
        This method need the d1 which should calculate firstlt
        :param S:   UnderlyingPrice
        :param K:   StrikePrice
        :param r:   RiskFreeRate
        :param sigma:  Volatility
        :param T:   Time2Maturity
        :return: d2
        """
        # return (LOG_F(S/K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        return self.BlackScholes_d1(S, K, r, sigma, T) - (sigma * np.sqrt(T))

    def Delta(self, s, k, r, g, sigma, t, OptionType):  # S, K, r,g, sigma, T
        """
        Delta（Δ）又称对冲值 :option价值对于股票的导数，
                dC/dS
        :param s:  UnderlyingPrice
        :param k:  StrikePrice
        :param r:  RiskFreeRate
        :param g:  DividendYield
        :param sigma: Volatility
        :param t:  Time2Maturity
        :param OptionType:  'Call' or 'Put
        :return:  Delta
        """
        # Delta（Δ）又称对冲值 :option价值对于股票的导数，
        # dC/dS
        if self.Model == 'Black-Sholes' and self.OptionStyle == 'European':
            d1 = (np.log(s / k) + ((r - g) + 0.5 * (sigma)**2)
                  * (t)) / (sigma * np.sqrt(t))
            if OptionType in ('Call', 'call', 'C', 'c'):
                deltavalue = sps.norm.cdf(d1)
            elif OptionType in ('Put', 'put', 'P', 'p'):
                # d1=(np.log(s/k)+((r-g)+0.5*(sigma)**2)*(t))/(sigma*np.sqrt(t))
                # d2=d1-sigma*np.sqrt(t)
                deltavalue = sps.norm.cdf(d1) - 1
            else:
                raise ValueError('Unknown Option type: %s' % OptionType)
        else:
            raise ValueError('Unsupported Model or OptionStyle:(%s,%s), Waiting for upgrading.' % (
                self.Model, self.OptionStyle))
        return deltavalue

    def Gamma(self, S, K, r, sigma, T, OptionType, IV_LOWER_BOUND=1e-8):
        # S, K, r, sigma, T
        """
        # Gamma（γ）Delta对于股票的导数
        # d2C/dS2
        :param S:  UnderlyingPrice
        :param K:  StrikePrice
        :param r:  RiskFreeRate
        :param sigma: Volatility
        :param T:   Time2Maturity
        :param OptionType:  'Call' or 'Put
        :param IV_LOWER_BOUND: the lower Bound of IV which is the breakpoint in the BSpricinga as 1e-8
        :return:  Gamma
        """
        # Gamma（γ）Delta对于股票的导数
        # d2C/dS2

        if sigma > IV_LOWER_BOUND:
            return sps.norm.pdf(self.BlackScholes_d1(S, K, r, sigma, T)) / (S * sigma * np.sqrt(T))
        else:
            return 0

    def Vega(self, S, K, r, sigma, T, OptionType):  # S, K, r, sigma, T
        """

        # Vega（ν）option价格对于波动性的导数
        :param S:  UnderlyingPrice
        :param K:  StrikePrice
        :param r:  RiskFreeRate
        :param sigma:  Volatility
        :param T:   Time2Maturity
        :param OptionType:   'Call' or 'Put
        :return:  Vega
        """
        # Vega（ν）option价格对于波动性的导数

        return sps.norm.pdf(self.BlackScholes_d1(S, K, r, sigma, T)) * S * np.sqrt(T)

    def Theta(self, S, K, r, sigma, T, OptionType):  # S, K, r, sigma, T
        """
        Theta（θ）时间de导数
        :param S:  UnderlyingPrice
        :param K:  StrikePrice
        :param r:  RiskFreeRate
        :param sigma:  Volatility
        :param T:   Time2Maturity
        :param OptionType:   'Call' or 'Put
        :return:  Theta
        """
        # Theta（θ）时间de导数

        if OptionType in ('Call', 'call', 'C', 'c'):

            return (-S * sigma * sps.norm.pdf(self.BlackScholes_d1(S, K, r, sigma, T)) / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * sps.norm.cdf(self.BlackScholes_d2(S, K, r, sigma, T)))
        elif OptionType in ('Put', 'put', 'P', 'p'):
            return (-S * sigma * sps.norm.pdf(self.BlackScholes_d1(S, K, r, sigma, T)) / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * sps.norm.cdf(-self.BlackScholes_d2(S, K, r, sigma, T)))
        else:
            raise ValueError('Unknown Option type: %s' % self.OptionType)

    def Rho(self, S, K, r, sigma, T, OptionType):
        """
        Rho（ρ）利率的导数
        :param S: UnderlyingPrice
        :param K: StrikePrice
        :param r: RiskFreeRate
        :param sigma: Volatility
        :param T: Time2Maturity
        :param OptionType:  'Call' or 'Put
        :return:  Rho
        """
        # Rho（ρ）利率的导数

        if OptionType in ('Call', 'call', 'C', 'c'):
            return K * T * np.exp(-r * T) * sps.norm.cdf(self.BlackScholes_d2(S, K, r, sigma, T))
        elif OptionType in ('Put', 'put', 'P', 'p'):
            return -K * T * np.exp(-r * T) * sps.norm.cdf(-self.BlackScholes_d2(S, K, r, sigma, T))
        else:
            raise ValueError('Unknown Option type: %s' % self.OptionType)


if __name__ == '__main__':
    # get Greeks
    Greek = Sensibility()
    Delta, Gamma, Vega, Theta, Rho = Greek.getGreeks(UnderlyingPrice=2.634,
                                                     Strike=2.45,
                                                     Volatility=.23,
                                                     Time2Maturity=18.0 / 252,
                                                     DividendYield=0,
                                                     RiskFreeRate=0.04,
                                                     OptionType='Call')

    print Delta, Gamma, Vega, Theta, Rho
    # get IV
    IV = OP.ImpliedVolatility()
    # s:Underlying Assets Price
    # k:Strike Price
    # r:risk free interest rate
    # T:avaiable time
    # sigma:square root of annual variance
    # cp: call or put fee
    # g: dividend yield
    # dividends:continuous
    sigma, status, ratio = IV.ImpliedVolatility_OlD(S=2.634,
                                                    K=2.45,
                                                    r=0.04,
                                                    T=18.0 / 252,
                                                    cp=1.00998,
                                                    cp_sign=-1,
                                                    g=0)
    print sigma, status, ratio
