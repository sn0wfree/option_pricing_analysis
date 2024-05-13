# -*- coding:utf-8 -*-

import numpy as np
import scipy.stats as sps
from CodersWheel.QuickTool.timer import timer
from scipy.optimize import brentq


class OptionBaseTools(object):
    @staticmethod
    def alter_dividends(s, g, r, t, dividends):
        if dividends != 'continuous':
            d = g
            s = s - d * np.exp(-r * t)
            g = 0
        else:
            pass
        return s, g

    @staticmethod
    def detect_cp_sign(cp_sign):
        if cp_sign in ('Call', 'call', 'c', 'C'):
            cp_sign = 1
        elif cp_sign in ('p', 'P', 'Put', 'put'):
            cp_sign = -1
        elif cp_sign in (-1, 1):
            pass
        else:
            raise ValueError('cp_sign (%s) is Unknown input' % cp_sign)
        return cp_sign


class BlackScholesOptionPricingModel(OptionBaseTools):
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

        s, g = self.alter_dividends(s, g, r, t, dividends)
        self.variables = s, k, r, t, sigma, cp_sign, g, dividends

    @staticmethod
    def cal_d(s, k, r, t, sigma, g, dividends):
        if dividends != 'continuous':
            d = g
            s = s - d * np.exp(-r * t)
            g = 0
        else:
            pass

        d1 = (np.log(s / k) + ((r - g) + 0.5 * (sigma ** 2)) * t) / np.float64(sigma * np.sqrt(t))

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
        d1, d2 = cls.cal_d(s, k, r, t, sigma, g, dividends)
        Nd1, Nd2 = cls.Nd(cp_sign, d1, d2)

        fee = cp_sign * s * np.exp(-g * t) * Nd1 - cp_sign * k * np.exp(-r * t) * Nd2

        return fee


class OptionPricing(object):
    __slots__ = ()

    @staticmethod
    # @timer
    def BSPricing(s, k, r, t, sigma, cp_sign, g=0, dividends='continuous', self_cal=True, IV_LOWER_BOUND=1e-8):
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
        # if cp_sign in ('Call', 'call', 'c', 'C'):
        #     cp_sign = 1
        # elif cp_sign in ('p', 'P', 'Put', 'put'):
        #     cp_sign = -1
        # elif cp_sign in (-1, 1):
        #     pass
        # else:
        #     raise ValueError('cp_sign (%s) is Unknown input' % cp_sign)

        cp_sign = OptionBaseTools.detect_cp_sign(cp_sign)

        if self_cal:
            s, g = OptionBaseTools.alter_dividends(s, g, r, t, dividends)

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
    # @timer
    def MCPricing(s, k, r, t, sigma, cp_sign, g, dividends='continuous', iteration=1000000, **kwargs):
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
        # import time

        cp_sign = OptionBaseTools.detect_cp_sign(cp_sign)
        # func = lambda x: max(cp_sign * (x - k), 0)
        s, g = OptionBaseTools.alter_dividends(s, g, r, t, dividends)

        zt = np.random.normal(0, 1, iteration)
        st = s * np.exp((r - g - .5 * sigma ** 2) * t + sigma * t ** .5 * zt)
        # t1 = time.time()
        _p = np.maximum((st - k) * cp_sign, 0)
        # t2 = time.time()
        # p = [func(x) for x in st]
        # t3 = time.time()

        # print(t2 - t1, t3 - t2)

        return np.average(_p) * np.exp(-r * t)


class ImpliedVolatility(object):

    # http://www.codeandfinance.com/finding-implied-vol.html
    # Newton's method

    # Calculate the Black-Scholes implied volatility using the Brent method (for reference).
    # Return float, a zero of f between a and b.
    # f must be a continuous function, and [a,b] must be a sign changing interval.
    # it means f(a) and f(b) must have different signs

    def __init__(self, pricing_f=OptionPricing, method='BSPricing'):
        self.pricing_f = pricing_f
        self.method = method

    def loss_func(self, S, K, r, T, option_market_price, cp_sign, g, dividends='continuous',
                  method='BSPricing', **kwargs):

        if method is None:
            callback = getattr(self.pricing_f, self.method)
        else:
            callback = getattr(self.pricing_f, method)

        def _callback_(sigma):

            return callback(S, K, r, T, sigma, cp_sign, g, dividends=dividends, **kwargs) - option_market_price

        return _callback_

    @timer
    def iv_brent(self, S, K, r, T, option_market_price, cp_sign, g, dividends='continuous',
                 IV_LOWER_BOUND=1e-8, method='BSPricing', **kwargs):
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

        loss = self.loss_func(S, K, r, T, option_market_price, cp_sign, g, dividends=dividends, method=method, )
        # other_loss = self.loss_func(S, K, r, T, option_market_price, cp_sign, g, dividends=dividends,
        #                             method=alter_method, )
        max_iter = 2000
        a, b = 0.01, 1

        fa = loss(a)
        fb = loss(b)
        iter_count = 0
        while fa * fb > 0 and iter_count < max_iter and a < IV_LOWER_BOUND:
            if abs(fa) < abs(fb):
                a /= 2  # Decrease a if f(a) is closer to zero
            else:
                b *= 2  # Increase b if f(b) is closer to zero
                b = min(1, b)  # 防止b过大
            fa = loss(a, )
            fb = loss(b, )
            iter_count += 1

        if fa * fb < 0:
            ret_iv = brentq(loss, a, b)
            # ret_iv = max(v, IV_LOWER_BOUND)
            diff = loss(ret_iv, )
            return ret_iv, diff
        else:
            # b = max(b, 1)
            # ret_iv = brentq(other_loss, a, b)
            diff_b = loss(b, )
            diff_a = loss(a, )

            if diff_b <= diff_a:

                return b, diff_b
            else:
                return IV_LOWER_BOUND, loss(IV_LOWER_BOUND, )

    # def Implied_volatility_Call(self, S, K, r, T, market_option_price, g=0, dividends='continuous'):
    #     return self.iv_brent(S, K, r, T, market_option_price, cp_sign='call', g=g, dividends=dividends)
    #
    # def Implied_volatility_Put(self, S, K, r, T, market_option_price, g=0, dividends='continuous'):
    #     return self.iv_brent(S, K, r, T, market_option_price, cp_sign='put', g=g, dividends=dividends)
    #
    # def ImpliedVolatility(self, S, K, r, T, market_option_price, cp_sign, g=0, dividends='continuous'):
    #     return self.iv_brent(S, K, r, T, market_option_price, cp_sign, g=g, dividends=dividends)
    #
    # def ImpliedVolatility_OlD(self, s, k, r, t, cp, cp_sign, g, IV_LOWER_BOUND=1e-8, func=OptionPricing().BSPricing):
    #     """
    #     this function is calculate the implied volatility of one option self.
    #     :param s:  Underlying Assets Price
    #     :param k:  Strike Price
    #     :param r:  risk free interest rate
    #     :param t:  available time
    #     :param cp:  call or put fee
    #     :param cp_sign: 1 if call else -1 if put
    #     :param g: dividend rate
    #     :param IV_LOWER_BOUND: 1e-8
    #     :param func:
    #     :return:
    #     """
    #     # this function is calculate the implied volatility of one option self.
    #     # ---------------------------------
    #     # s: Underlying Assets Price
    #     # k:Strike Price
    #     # r:risk free interest rate
    #     # T: available time
    #     # sigma:square root of annual variance
    #     # cp: call or put fee
    #     # g: dividend yield
    #     # dividends:continuous
    #
    #     # func=BSPricing
    #     sigma = 0.3  # initial volatility
    #     mktprice = cp
    #
    #     lower = 1e-10  # initial low Bound
    #     upper = 1  # initial Upper Bound
    #     count = 0  # initial counter
    #     last = 0
    #     C = func(s, k, r, t, sigma, cp_sign, g)
    #
    #     # sigma = 0.3 # initial volatility
    #
    #     while abs(C - mktprice) > IV_LOWER_BOUND:
    #         C = func(s, k, r, t, sigma, cp_sign, g)
    #         last = C
    #         if C - mktprice > 0:
    #             # print 'large'
    #             upper = sigma
    #             sigma = (sigma + lower) / 2  # lower
    #         else:
    #             # print 'small'
    #             lower = sigma
    #             sigma = (sigma + upper) / 2  # higher
    #         if upper - lower < IV_LOWER_BOUND:
    #             count += 1
    #
    #             """if 500>count:
    #                     if sigma >=1 and C*1.5>mktprice:
    #                         pass
    #                     elif sigma<=0 and C<mktprice*1.5:pass
    #                 """
    #             if 10000 > count:
    #                 pass
    #             else:
    #                 if upper <= IV_LOWER_BOUND:
    #                     sigma = 0
    #                 elif lower >= 1 - IV_LOWER_BOUND:
    #                     sigma = 1
    #                 else:
    #                     sigma = (upper + lower) / 2
    #                 break
    #
    #     if sigma >= 1:
    #         status = 'Market Price own Invalid Price (abnormal High)'
    #     elif sigma <= 0:
    #         status = 'Market Price own Invalid Price (abnormal Low)'
    #     else:
    #         status = "Normal"
    #     # sigma,status,C, mktprice
    #     # implied volatility , status for testing the IV whether has
    #     return sigma, status, C / mktprice  # implied volatility


if __name__ == '__main__':
    iv_func = ImpliedVolatility(pricing_f=OptionPricing, method='MCPricing').iv_brent
    s, x, r, t, cp_sign, g = 5473.55, 4950, 0.015, 28 / 250, -1, 0

    cp_fee = 51.4

    print('BS')
    iv = iv_func(s, x, r, t, cp_fee, cp_sign, g, method='BSPricing')
    # iv_old = ImpliedVolatility(pricing_f=op).ImpliedVolatility_OlD(s, x, r, t, cp, cp_sign, g)
    # print cp,sigma,CalImpliedVolatility(s,x,r,t,cp,cp_sign,g)
    print(cp_fee, iv)

    print('MC')
    iv = iv_func(s, x, r, t, cp_fee, cp_sign, g, method='MCPricing')
    # # iv_old = ImpliedVolatility(pricing_f=op).ImpliedVolatility_OlD(s, x, r, t, cp, cp_sign, g)
    # # print cp,sigma,CalImpliedVolatility(s,x,r,t,cp,cp_sign,g)
    print(cp_fee, iv)
