# coding=utf-8
import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.optimize import brentq, fmin, minimize


"""
实现步骤：
（1） 使用正股复权后价格计算股价年波动率，数据长度为计算日前 1 年；
（2） Monte Carlo 模拟生成 n 条股价路径，对未来正股价格进行预测；
（3） 每一计息年度，对 n 条路径进行跟踪，一旦股价满足回售条款，则调整转股价，
使得可转债价值大于回售价格；
（4） 在触发回售条款的路径中，统计触发回售之后又触发赎回条款的路径，计算触
发日转股价值，并贴现至计算日，加上存续期截止至转股时的利息贴现，记为
CBVi；
（5） 在未触发回售条款的路径中，统计触发赎回条款的路径，计算触发日转股价值，
并贴现至计算日，加上存续期截止至转股时的利息贴现，记为 CBVi；
（6） 对于未触发赎回条款的其余路径，不受任何可转债条款影响，是否转股只与转
股价值有关，用 LSM 方法对该美式期权进行定价；
到期日可转债价格为
CB S T X coupon end = + max( ( )*100 / ,100 ( ))；
（coupon end ( ) 为最后一期票息）；找出 T-1 时刻转股价值大于持有价值的路
径，将股价记为 Z；将路径对应的 CB(T)贴现到 T-1 时刻，记为 Y，利用最小
二乘法，得到回归方程：
2 Y a bZ cZ =+ + * *
代入 Z，得到期望值，与 T-1 时刻的转股价值比较，决定继续持有还是转股；
如转股，则将现金流设置为转股价值；依次循环，得到 T-2、…、1 时刻的转债
价格，最后均贴现至计算日；
（7） 取各路径转债价值的平均值，即得可转债价值；
"""
"""
T: 可转债剩余到期时间
Bc： 到期赎回价（不包含最后一期利息）
dt： dt = T/200, 表示单位时间间隔
r： 无风险利率
sigma： 股票年化波动率
C： 条件赎回价格
X： 可转债执行价格
P: 条件回售价格
n： n = 100/x, 表示可转债转换比例
"""


class MockStockPath(object):
    @staticmethod
    def create_randn(n: int = (100000, 200)):
        zt = np.random.normal(0, 1, n)
        return zt

    @staticmethod
    def create_routes(r, sigma, T, randn_martix):
        """

        :param r:
        :param sigma:
        :param T:  可转债剩余到期时间
        :param randn_martix:
        :return:
        """
        dt = T / 200
        route = np.exp((r - (sigma ** 2) / 2) * dt + sigma * np.sqrt(dt) * randn_martix)
        return route

    @classmethod
    def create_sn(cls, sn, r, sigma, T, n: int = (100000, 200)):
        """
        mock stock path
        :param sn:
        :param r:
        :param sigma:
        :param T: 可转债剩余到期时间
        :param n:
        :return:
        """
        randn_martix = cls.create_randn(n)
        route = cls.create_routes(r, sigma, T, randn_martix)
        route = pd.DataFrame(route, columns=[f"day_{t + 1}" for t in range(200)]).cumprod(axis=1)
        route.index.name = 'route'
        return route * sn


class Provision(object):
    @staticmethod
    def provised_strike_price(self):
        pass

    @staticmethod
    def formula(st, nd1, provised_k, nd2, r, T, i):
        """

        :param st:
        :param nd1:
        :param provised_k:
        :param nd2:
        :param r:
        :param T: 可转债剩余到期时间
        :param i:
        :return:
        """
        p1 = st * nd1 - provised_k * np.exp(-r * (T) * nd2)
        P = p1 * (100 / provised_k) + (100 + i) * np.exp(-r * (T))
        return P

    @staticmethod
    def d1(st, provised_k, r, sigma, T):
        """

        :param st:
        :param provised_k:
        :param r:
        :param sigma:
        :param T:  可转债剩余到期时间
        :return:
        """
        return (np.log(st / provised_k) + (r + (sigma ** 2) / 2) * (T)) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(d1, sigma, T):
        """

        :param d1:
        :param sigma:
        :param T: 可转债剩余到期时间
        :return:
        """
        return d1 - sigma * (np.sqrt(T))

    @classmethod
    def get_d1_d2(cls, st, provised_k, r, sigma, T):
        """

        :param st:
        :param provised_k:
        :param r:
        :param sigma:
        :param T: 可转债剩余到期时间
        :return:
        """
        d1 = cls.d1(st, provised_k, r, sigma, T)
        d2 = cls.d2(d1, sigma, T)
        return d1, d2

    @staticmethod
    def get_nd1_nd2(d1, d2):
        return sps.norm.cdf(d1), sps.norm.cdf(d2)


class MockStrikePrice(object):
    @staticmethod
    def init_strike_price(k: float, n: int = (100000, 200)):
        k_martix = pd.DataFrame(np.ones(n) * k, columns=[f"day_{t + 1}" for t in range(n[1])])
        return k_martix
        pass

    @staticmethod
    def check_act_downward_adj(kn, sn, m=0.8):
        """
        Downward Adjustment 向下调整
        :return:
        """
        mask = sn < m * kn
        return mask

    @staticmethod
    def d1(st, d_xt, r, sigma, T):
        """

        :param st: stock price
        :param d_xt:  downward strike price
        :param r:  risk free rate
        :param sigma: vol
        :param T:  可转债剩余到期时间

        :return:
        """
        return (np.log(st / d_xt) + (r + (sigma ** 2) / 2) * (T)) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(d1, sigma, T, ):
        """

        :param d1:
        :param sigma:
        :param T: 可转债剩余到期时间
        :return:
        """
        return d1 - sigma * (T)

    @classmethod
    def Nd(cls, st, d_xt, r, sigma, T, ):
        """

        :param st:
        :param d_xt:
        :param r:
        :param sigma:
        :param T: 可转债剩余到期时间
        :return:
        """
        d1 = cls.d1(st, d_xt, r, sigma, T, )
        d2 = cls.d2(d1, sigma, T, )
        return sps.norm.cdf(d1), sps.norm.cdf(d2)

    @classmethod
    def formula_d_xt_P(cls, st, d_xt, r, sigma, T, I):
        """

        :param st:
        :param d_xt:
        :param r:
        :param sigma:
        :param T: 可转债剩余到期时间
        :param I:
        :return:
        """
        nd1, nd2 = cls.Nd(st, d_xt, r, sigma, T, )
        P = (st * nd1 - d_xt * np.exp(-r * (T)) * nd2) * (100 / d_xt) + (100 + I) * np.exp(-r * (T))
        return P

    @classmethod
    def cal_d_xt(cls, P, st, old_kt, r, sigma, T, I):
        """
        计算下修后的转股价格（d_xt）
        :param P: 条件回售价格
        :param st:  stock price
        :param old_kt: 未下修前的kt
        :param r:   risk free rate
        :param sigma:  vol
        :param T:可转债剩余到期时间

        :param I: 利息
        :return:
        """
        func = lambda d_xt: P - cls.formula_d_xt_P(st, d_xt, r, sigma, T, I)

        d_xt_solved_result = minimize(func, old_kt * 0.8)
        if d_xt_solved_result.success:
            return d_xt_solved_result.x
        else:
            raise ValueError(f'{d_xt_solved_result}')

    @classmethod
    def cal_value(cls, st, xt, r, sigma, T, I):
        """

        :param st:  stock price
        :param xt: strike price
        :param r: risk free rate
        :param sigma:  vol
        :param T:  remaining duration
        :param I: interests
        :return:
        """
        V = cls.formula_d_xt_P(st, xt, r, sigma, T, I)
        return V

    @classmethod
    def check_wthr_dwnwrd_adj(cls, st, xt, r, sigma, T, I, P):
        """
        st < 0.7xt
        :param st:
        :param xt:
        :param r:
        :param sigma:
        :param T:
        :param I:
        :param P:
        :return:
        """
        v = cls.cal_value(st, xt, r, sigma, T, I)
        if v < P:
            print('will downward adjustment strike price')
            return cls.cal_d_xt(P, st, xt, r, sigma, T, I)
        else:
            print('will not downward adjustment strike price')
            return xt


if __name__ == '__main__':
    # 济州转债
    r, sigma, T = 0.015, 0.228166, 1.1808
    st = 17.19
    xt = 23.4
    n = (100000, 200)
    I = 0.015 * 100
    P = 11.7  # 回售触发价
    print(st < xt*0.8)
    ## 触发修正价 18.72 = 23.4 * 0.8
    # sn = MockStockPath.create_sn(s, r, sigma, T, n=n)
    # kn = MockStrikePrice.init_strike_price(k, n)
    d_kt = MockStrikePrice.check_wthr_dwnwrd_adj(st, xt, r, sigma, T, I, P)
    pass
