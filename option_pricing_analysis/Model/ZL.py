# coding=utf-8
import numpy as np
import pandas as pd
import scipy.stats as sps
import warnings
from CodersWheel.QuickTool.file_cache import file_cache
from functools import partial
from scipy.optimize import brentq

# TODO 债券利息还没处理好
# TODO 条款信息还没有处理好
"""
ZL 模型思想
推论1：中国可转债发行公司的最优决策是尽可能早地、以尽可能高的转股价格促使投资者将可转债转成公司股票
推论2：在中国特殊的制度背景下，可转债中股性占了绝大部分，而且中国的信用风险溢酬不高，因此将可转债的股性和债性统一起来，全部使用无风险利率进行贴现，
并不会对可转债的价值造成很大的影响。
推论3：因为中国可转债发行条款均规定转股价将根据公司股票的股利政策进行相应的调整，可转债中的转股权不会被提前执行，他实际上是一个欧式看涨期权。
推论4：公司会选择尽可能短的赎回期
推论5：可转债发现公司只有在面临回售压力时才会调低转股价，调低幅度也仅以使得可转债价格稍微超过回售价格为限。


"""

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


#
# class Provision(object):
#     @staticmethod
#     def provised_strike_price(self):
#         pass
#
#     @staticmethod
#     def formula(st, nd1, provised_k, nd2, r, T, i):
#         """
#
#         :param st:
#         :param nd1:
#         :param provised_k:
#         :param nd2:
#         :param r:
#         :param T: 可转债剩余到期时间
#         :param i:
#         :return:
#         """
#         p1 = st * nd1 - provised_k * np.exp(-r * (T) * nd2)
#         P = p1 * (100 / provised_k) + (100 + i) * np.exp(-r * (T))
#         return P
#
#     @staticmethod
#     def d1(st, provised_k, r, sigma, T):
#         """
#
#         :param st:
#         :param provised_k:
#         :param r:
#         :param sigma:
#         :param T:  可转债剩余到期时间
#         :return:
#         """
#         return (np.log(st / provised_k) + (r + (sigma ** 2) / 2) * (T)) / (sigma * np.sqrt(T))
#
#     @staticmethod
#     def d2(d1, sigma, T):
#         """
#
#         :param d1:
#         :param sigma:
#         :param T: 可转债剩余到期时间
#         :return:
#         """
#         return d1 - sigma * (np.sqrt(T))
#
#     @classmethod
#     def get_d1_d2(cls, st, provised_k, r, sigma, T):
#         """
#
#         :param st:
#         :param provised_k:
#         :param r:
#         :param sigma:
#         :param T: 可转债剩余到期时间
#         :return:
#         """
#         d1 = cls.d1(st, provised_k, r, sigma, T)
#         d2 = cls.d2(d1, sigma, T)
#         return d1, d2
#
#     @staticmethod
#     def get_nd1_nd2(d1, d2):
#         return sps.norm.cdf(d1), sps.norm.cdf(d2)


class MockStrikePrice(object):

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
    @file_cache()
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
    def cal_d_xt(cls, P, st, old_kt, r, sigma, T, I, IV_LOWER_BOUND=1e-8):
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

        d_xt_solved_result = brentq(func, 1e-8, old_kt * 1)

        return d_xt_solved_result if d_xt_solved_result > IV_LOWER_BOUND else IV_LOWER_BOUND

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
    def _cal_dwnwrd_adj(cls, st, xt, r, sigma, T, I, P):
        v = cls.cal_value(st, xt, r, sigma, T, I)
        # 当持有可转债价值(V) 小于回售价值（P）的时候，投资者可能选择回售，企业会面临回售压力,从而选择调低转股价格（下修）
        if v < P:
            print('will downward adjustment strike price')
            return cls.cal_d_xt(P, st, xt, r, sigma, T, I)
        else:
            print('will not downward adjustment strike price')
            return xt

    @classmethod
    # @sqlite_cache()
    def check_wthr_dwnwrd_adj(cls, st, xt, r, sigma, T, I, P, buyback_cond: float, force=False):
        """
        st < 0.7xt
        ## todo 30/30 70%
        :param force:
        :param buyback_cond:  回售触发条件
        :param st:
        :param xt:
        :param r:
        :param sigma:
        :param T:
        :param I:
        :param P:
        :return:
        """
        warnings.warn('please correct judge func for  active buyback situation as 30/30 70%')
        warnings.warn('please correct interest func for interest')
        if force:
            print('force cal downward-adjusted strike price')
            return cls._cal_dwnwrd_adj(st, xt, r, sigma, T, I, P), 'force'
        # check whether active buyback situation
        if st < buyback_cond * xt:
            print(' active buyback condition!')
            print(st, buyback_cond * xt, st < buyback_cond * xt)
            return cls._cal_dwnwrd_adj(st, xt, r, sigma, T, I, P), 'active'

        else:
            print('not active buyback condition!')
            return xt, 'non-active'


class Mock(object):
    @staticmethod
    def init_strike_price(k: float, n: int = (100000, 200)):
        k_martix = pd.DataFrame(np.ones(n) * k, columns=[f"day_{t + 1}" for t in range(n[1])])
        return k_martix

    @staticmethod
    def init_value(n: int = (100000, 200)):
        v_martix = pd.DataFrame(np.ones(n), columns=[f"day_{t + 1}" for t in range(n[1])])
        return v_martix

    @staticmethod
    def init_stock_price(sn, r, sigma, T, n: int = (100000, 200)):
        return MockStockPath.create_sn(sn, r, sigma, T, n)

    @classmethod
    @file_cache()
    def check_dwnwrd(cls, S, K, V, r=0.015, sigma=0.222777, T=0.3342,
                     I=0.015 * 100,
                     P=103.0,  # 回售触发价
                     buyback_cond=0.7):
        """

        :param S:
        :param K:
        :param V:
        :param r:
        :param sigma:
        :param T:
        :param I:
        :param P:
        :param buyback_cond:
        :return:
        """
        downward_adj_func = partial(MockStrikePrice.check_wthr_dwnwrd_adj, r=r, sigma=sigma, I=I, P=P,
                                    buyback_cond=buyback_cond)
        for idx in S.index:
            s = S.loc[idx]
            k = K.loc[idx]
            # v = V.loc[idx]
            # check whether downward adj
            ### todo will reduce T
            K.loc[idx] = dwn_k = \
                pd.DataFrame([downward_adj_func(st=s1, xt=k1, T=T, ) for s1, k1 in zip(s, k)],
                             columns=['strike', 'status'], index=k.index)['strike']
            V.loc[idx] = [MockStrikePrice.cal_value(st, kt, r, sigma, T, I) for st, kt in zip(s, dwn_k)]
        return S, K, V

    @staticmethod
    def redeem_value(strike, stock, tc, r, interest_func):
        """

        :param r:
        :param strike:
        :param stock:
        :param tc:
        :param interest_func:  截止时间tc获得利息
        :return:
        """
        n = 100 / strike
        V = np.exp(-r * tc) * (n * stock + interest_func(tc))
        return V

    @staticmethod
    def no_redeem_value(Bc, strike, stock, T, interest_func):
        """

        :param Bc:
        :param strike:
        :param stock:
        :param T:
        :param interest_func:
        :return:
        """
        n = 100 / strike
        V = np.exp(-r * T) * (max(Bc, n * stock) + interest_func(T))
        return V

    @classmethod
    def check_path_value(cls, S, K, V, Bc, redeem=1.3, interest_func=lambda x: 0.015 * 100):
        redeem_mask = (S > redeem * K) * 1  # True redeemed, 符合赎回条件的路径
        redeem_status = redeem_mask.cumsum(axis=1) == 1  # 判断首次首次出现的位置
        for path, rede in enumerate(redeem_status.values):
            f = np.argwhere(rede == True)  # Find the indices of array elements that are non-zero, grouped by element.
            if len(f) > 0:  # redeem
                t = f[0][0]  # 取第一次出现符合赎回条件的时间点
                tc = t / 200
                st = S.iloc[path, t]
                kt = K.iloc[path, t]
                yield path, cls.redeem_value(kt, st, tc, r, interest_func)
            else:  # not redeem
                st = S.iloc[path, -1]
                kt = K.iloc[path, -1]
                tc = S.shape[1] / 200
                yield path, cls.no_redeem_value(Bc, kt, st, tc, interest_func)

    @classmethod
    def mock(cls, st,xt, r, sigma, T, n, I, P, buyback_cond, redeem, Bc, interest_func):
        S = Mock.init_stock_price(st, r, sigma, T, n=n)
        V = Mock.init_value(n)
        K = Mock.init_strike_price(xt, n=n)
        S, K, V = Mock.check_dwnwrd(S, K, V, r=r, sigma=sigma, T=T, I=I, P=P,  # 回售触发价
                                    buyback_cond=buyback_cond)

        c = pd.DataFrame(list(Mock.check_path_value(S, K, V, Bc, redeem=redeem, interest_func=interest_func)),
                         columns=['path', 'value'])

        return S, K, V, c

    @classmethod
    def mock_avg(cls, *args, **kwargs):
        S, K, V, res = cls.mock(*args, **kwargs)
        return res['value'].mean()


# 需要考虑的
# 1. 回售期
# 2. 回售触发条件
# 3. 转股期
# 4. 赎回触发条件
# 5. 下修条件
if __name__ == '__main__':
    np.random.seed(1)
    # 九州转债
    # 林洋转债
    r, sigma, T = 0.015, 0.503558, 2.1123
    st = 11.6
    xt = 8.44
    n = (1000, 200)

    P = 0  # 条件回售价 ???
    buyback_cond = 0.7
    Bc = 106  # 到期赎回价
    I = 0.015 * 100
    interest_func = lambda x: 0.015 * 100
    redeem = 1.3 # 赎回

    # print(st < xt * buyback_cond)
    # # 触发修正价 18.72 = 23.4 * 0.8
    # # sn = MockStockPath.create_sn(s, r, sigma, T, n=n)
    # # kn = MockStrikePrice.init_strike_price(k, n)
    # d_kt = MockStrikePrice.check_wthr_dwnwrd_adj(st, xt, r, sigma, T, I, P, buyback_cond)
    # print(xt, d_kt)
    S, K, V, res = Mock.mock(st,xt, r, sigma, T, n, I, P, buyback_cond, redeem, Bc, interest_func)
    # S.to_csv('S.csv')
    # K.to_csv('K.csv')
    # V.to_csv('V.csv')
    # res.to_csv('res.csv')
    res2 = Mock.mock_avg(st,xt, r, sigma, T, n, I, P, buyback_cond, redeem, Bc, interest_func)
    print(1)

    pass
