# -*- coding:utf-8 -*-
import datetime

import numpy as np
import scipy.stats as sps
from scipy.optimize import brentq


class BlackScholesOptionPricingModel(object):
    """
    this class is to calculate the option fee by Black-Scholes Model
    """

    def __init__(self, s, k, r, t, sigma, cp_sign, g, dividends='continuous'):
        """

        :param s: Underlying Assets Price
        :param k: Strike Price
        :param r: risk free interest rate
        :param t: Time 2 maturity
        :param sigma: square root of annual variance
        :param cp_sign:  1 if call ;-1 if put
        :param g:  dividend rate ; default 0
        :param dividends: dividend method
        """
        pass

        if dividends != 'continuous':
            d = g
            s = s - d * np.exp(-r * t)
            g = 0
        else:
            pass
        self.variables = s, k, r, t, sigma, cp_sign, g, dividends

    @staticmethod
    def d(s, k, r, t, sigma, g, dividends):
        if dividends != 'continuous':
            d = g
            s = s - d * np.exp(-r * t)
            g = 0
        else:
            pass
        # variables = s, k, r, t, sigma, cp_sign, g, dividends

        d1 = (np.log(s / k) + ((r - g) + 0.5 * (sigma ** 2)) * t) / np.float(sigma * np.sqrt(t))

        d2 = d1 - sigma * np.sqrt(t)

        return d1, d2

    @staticmethod
    def Nd(cp_sign, d1, d2):
        return sps.norm.cdf(cp_sign * d1), sps.norm.cdf(cp_sign * d2)

    @classmethod
    def cal(cls, s, k, r, t, sigma, cp_sign, g, dividends):
        """

        :param s: Underlying Assets Price
        :param k: Strike Price
        :param r: risk free interest rate
        :param t: Time 2 maturity
        :param sigma: square root of annual variance
        :param cp_sign:  1 if call ;-1 if put
        :param g:  dividend rate ; default 0
        :param dividends: dividend method
        :return: optionfee
        """
        d1, d2 = cls.d(s, k, r, t, sigma, g, dividends)
        Nd1, Nd2 = cls.Nd(cp_sign, d1, d2)
        optionfee = cp_sign * s * np.exp(-g * t) * Nd1 - cp_sign * k * np.exp(-r * t) * Nd2
        return optionfee

    # def Cal(self, Combine=True):
    #     s, k, r, t, sigma, cp_sign, g, dividends = self.variables
    #     d1, d2 = self.d(s, k, r, t, sigma, cp_sign, g, dividends)
    #     if Combine:
    #         self.optionfee = cp_sign * s * np.exp(-g * t) * sps.norm.cdf(
    #             cp_sign * d1) - cp_sign * k * np.exp(-r * t) * sps.norm.cdf(cp_sign * d2)
    #     else:
    #         if cp_sign == 1:
    #
    #             self.optionfee = s * \
    #                              np.exp(-g * t) * sps.norm.cdf(d1) - k * \
    #                              np.exp(-r * t) * sps.norm.cdf(d2)
    #         elif cp_sign == -1:
    #
    #             self.optionfee = -s * \
    #                              np.exp(-g * t) * sps.norm.cdf(-d1) + k * \
    #                              np.exp(-r * t) * sps.norm.cdf(-d2)
    #     return self.optionfee


class OptionPricing(object):
    __slots__ = ()
    @staticmethod
    def BSPricing(s, k, r, t, sigma, cp_sign, g=0, dividends='continuous', self_cal=False, IV_LOWER_BOUND=1e-8):
        """

        Black-Scholes Option Pricing Model

        # this is for European Options
        # ------------------------------------
        # 总结如下：
        # 1、不派发股利欧式看涨期权 BS可使用
        # 2、不派发股利欧式看跌期权 BS可使用
        # 3、不派发股利美式看涨期权 BS可使用:不派发股利的美式看涨期权不会提前行权，性质上与欧式期权一致，
        # 4、不派发股利美式看跌期权 BS可使用:不派发股利的美式看跌期权不会提前行权，性质上与欧式期权一致，
        # 5、派发股利欧式看涨期权   BS可使用
        # 6、派发股利欧式看跌期权   BS可使用
        # 7、派发股利美式看涨期权   BS不可使用
        # 8、派发股利美式看跌期权   BS不可使用
        # -----------------------------
        :param s:  Underlying Assets Price
        :param k: Strike Price
        :param r: risk free interest rate
        :param t: Time 2 maturity
        :param sigma: square root of annual variance
        :param cp_sign:  1 if call ;-1 if put
        :param g:  dividend rate ; default 0
        :param dividends: dividend method
        :param self_cal:  selfcal default : True
        :param IV_LOWER_BOUND: the lowerBOUND of IV which optimize calculation;Default 1e-8
        :return:  option fee
        """

        # if cp_sign in ('Call', 'call', 'c', 'C', 'p', 'P', 'Put', 'put'):
        if cp_sign in ('Call', 'call', 'c', 'C'):
            cp_sign = 1
        elif cp_sign in ('p', 'P', 'Put', 'put'):
            cp_sign = -1
        elif cp_sign in (-1, 1):
            pass
        else:
            raise ValueError('cp_sign (%s) is Unknown input' % cp_sign)

        if self_cal:
            if dividends != 'continuous':
                d = g
                s = s - d * np.exp(-r * t)
                g = 0
                # s= S − Dte − rT。
            else:
                pass

            if sigma > IV_LOWER_BOUND:
                d1 = (np.log(s / k) + ((r - g) + 0.5 * (sigma) ** 2)
                      * (t)) / float(sigma * np.sqrt(t))
            else:
                d1 = np.inf if s > k else -np.inf
            d2 = d1 - sigma * np.sqrt(t)
            optionfee = cp_sign * s * np.exp(-g * t) * sps.norm.cdf(
                cp_sign * d1) - cp_sign * k * np.exp(-r * t) * sps.norm.cdf(cp_sign * d2)
        else:
            optionfee = BlackScholesOptionPricingModel(
                s, k, r, t, sigma, cp_sign, g, dividends=dividends).cal(s, k, r, t, sigma, cp_sign, g, dividends)
        return optionfee
    @staticmethod
    def MCPricing(s, k, r, t, sigma, cp_sign, g, dividends='continuous', iteration=1000000):
        """
        Monte Carlo Pricing Model
        ----------------------------------
        :param s: Underlying Assets Price
        :param k: Strike Price
        :param r: risk free interest rate
        :param t: available time
        :param sigma: :square root of annual variance
        :param cp_sign: 1 if call else -1 if put
        :param g: dividend rate
        :param dividends: dividends type: continuous
        :param iteration:  iteration
        :return: option fee
        """
        # Monte Carlo Pricing Model
        # ----------------------------------
        # s: Underlying Assets Price
        # k: Strike Price
        # r: risk free interest rate
        # T: available time
        # sigma: square root of annual variance
        # cp: call or put fee
        # g: dividend yield
        # dividends: continuous

        if cp_sign in ('Call', 'call', 'c', 'C', 'p', 'P', 'Put', 'put'):
            if cp_sign in ('Call', 'call', 'c', 'C'):
                cp_sign = 1
            elif cp_sign in ('p', 'P', 'Put', 'put'):
                cp_sign = -1

        elif cp_sign in (-1, 1):
            pass
        else:
            raise ValueError('cp_sign (%s) is Unknown input' % cp_sign)

        if dividends != 'continuous':
            d = g
            s = s - d * np.exp(-r * t)
            g = 0
            # s= S − Dte − rT。
        else:
            pass
        zt = np.random.normal(0, 1, iteration)
        st = s * np.exp((r - g - .5 * sigma ** 2) * t + sigma * t ** .5 * zt)
        p = []
        for St in st:
            p.append(max(cp_sign * (St - k), 0))
        return np.average(p) * np.exp(-r * t)


class ImpliedVolatility(object):

    # http://www.codeandfinance.com/finding-implied-vol.html
    # Newton's method
    # def implied_volatility_newton(option_market_price, pricing_f, S, K, r, T):
    #    MAX_ITERATIONS = 100
    #    PRECISION = 1e-5
    #    sigma = 0.5
    #    for i in range(0, MAX_ITERATIONS):
    #        price = pricing_f(S, K, r, sigma, T)
    #        diff = option_market_price - price
    #        #print(i, sigma, diff)
    #        if abs(diff) < PRECISION:
    #            return sigma
    #        #x1 = x0 + f(x0) / f'(x0)
    #        sigma = sigma + diff / Vega(S, K, r, sigma, T)
    #    # value wasn't found, return best guess so far
    #    return sigma

    # Calculate the Black-Scholes implied volatility using the Brent method (for reference).
    # Return float, a zero of f between a and b.
    # f must be a continuous function, and [a,b] must be a sign changing interval.
    # it means f(a) and f(b) must have different signs

    def __init__(self, pricing_f=OptionPricing()):
        self.pricing_f = pricing_f
        pass

    def implied_volatility_brent(self, S, K, r, T, option_market_price, cp_sign, g, dividends='continuous',
                                 IV_LOWER_BOUND=1e-8):
        """



        :param S:  Underlying Assets Price
        :param K:  Strike Price
        :param r:  risk free interest rate
        :param T:  available time
        :param option_market_price:  option fee?
        :param cp_sign:  call or put sign
        :param g:  dividend yield
        :param dividends:  continuous
        :param IV_LOWER_BOUND:
        :return:
        """
        try:
            v = brentq(lambda sigma: option_market_price - self.pricing_f.BSPricing(S, K, r, T, sigma, cp_sign, g,
                                                                                    dividends=dividends),
                       0, 10)

            return v if v > IV_LOWER_BOUND else IV_LOWER_BOUND
        except:
            return IV_LOWER_BOUND

    def Implied_volatility_Call(self, S, K, r, T, market_option_price, cp_sign, g=0, dividends='continuous'):
        return self.implied_volatility_brent(market_option_price, S, K, r, T, cp_sign='call', g=g, dividends=dividends)

    def Implied_volatility_Put(self, S, K, r, T, market_option_price, cp_sign, g=0, dividends='continuous'):
        return self.implied_volatility_brent(market_option_price, S, K, r, T, cp_sign='put', g=g, dividends=dividends)

    def ImpliedVolatility(self, S, K, r, T, market_option_price, cp_sign, g=0, dividends='continuous'):
        return self.implied_volatility_brent(market_option_price, S, K, r, T, cp_sign, g=g, dividends=dividends)

    def ImpliedVolatility_OlD(self, s, k, r, t, cp, cp_sign, g, IV_LOWER_BOUND=1e-8, func=OptionPricing().BSPricing):
        """
        this function is calculate the implied volatility of one option self.
        :param s:  Underlying Assets Price
        :param k:  Strike Price
        :param r:  risk free interest rate
        :param t:  available time
        :param cp:  call or put fee
        :param cp_sign: 1 if call else -1 if put
        :param g: dividend rate
        :param IV_LOWER_BOUND: 1e-8
        :param func:
        :return:
        """
        # this function is calculate the implied volatility of one option self.
        # ---------------------------------
        # s: Underlying Assets Price
        # k:Strike Price
        # r:risk free interest rate
        # T: available time
        # sigma:square root of annual variance
        # cp: call or put fee
        # g: dividend yield
        # dividends:continuous

        # func=BSPricing
        sigma = 0.3  # initial volatility
        mktprice = cp

        lower = 1e-10  # initial low Bound
        upper = 1  # initial Upper Bound
        count = 0  # initial counter
        last = 0
        C = func(s, k, r, t, sigma, cp_sign, g)

        # sigma = 0.3 # initial volatility

        while abs(C - mktprice) > IV_LOWER_BOUND:
            C = func(s, k, r, t, sigma, cp_sign, g)
            last = C
            if C - mktprice > 0:
                # print 'large'
                upper = sigma
                sigma = (sigma + lower) / 2  # lower
            else:
                # print 'small'
                lower = sigma
                sigma = (sigma + upper) / 2  # higher
            if upper - lower < IV_LOWER_BOUND:
                count += 1

                """if 500>count:
                        if sigma >=1 and C*1.5>mktprice:
                            pass
                        elif sigma<=0 and C<mktprice*1.5:pass
                    """
                if 10000 > count:
                    pass
                else:
                    if upper <= IV_LOWER_BOUND:
                        sigma = 0
                    elif lower >= 1 - IV_LOWER_BOUND:
                        sigma = 1
                    else:
                        sigma = (upper + lower) / 2
                    break

        if sigma >= 1:
            status = 'Market Price own Invalid Price (abnormal High)'
        elif sigma <= 0:
            status = 'Market Price own Invalid Price (abnormal Low)'
        else:
            status = "Normal"
        # sigma,status,C, mktprice
        # implied volatility , status for testing the IV whether has
        return sigma, status, C / mktprice  # implied volatility


if __name__ == '__main__':
    op = OptionPricing()
    # 川投转债
    s, x, r, t, sigma, cp_sign, g = 8, 6.72, 0.02, 0.36, .5, 1, 0
    # cp = op.BSPricing(s, x, r, t, sigma, cp_sign, g)
    cp = 14.113
    iv = ImpliedVolatility(pricing_f=op).implied_volatility_brent(s, x, r, t, cp, cp_sign, g)
    # iv_old = ImpliedVolatility(pricing_f=op).ImpliedVolatility_OlD(s, x, r, t, cp, cp_sign, g)
    # print cp,sigma,CalImpliedVolatility(s,x,r,t,cp,cp_sign,g)
    print(cp, sigma,iv )
