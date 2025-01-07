# coding=utf-8
import numpy as np
import pandas as pd
import scipy.stats as sps
import warnings
from functools import partial
from CodersWheel.QuickTool.file_cache import file_cache
from scipy.optimize import brentq
from statsmodels.api import OLS


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
    def check_wthr_dwnwrd_adj(cls, st, xt, r, sigma, T, I, P, buyback_cond: float, force=False, verbose=True):
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
            if verbose:
                print('force cal downward-adjusted strike price')
            return cls._cal_dwnwrd_adj(st, xt, r, sigma, T, I, P), 'force'
        # check whether active buyback situation
        if st < buyback_cond * xt:
            if verbose:
                print(' active buyback condition!')
                print(st, buyback_cond * xt, st < buyback_cond * xt)
            return cls._cal_dwnwrd_adj(st, xt, r, sigma, T, I, P), 'active'

        else:
            if verbose:
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
    def check_dwnwrd(cls, S: pd.DataFrame, K: pd.DataFrame, V: pd.DataFrame, r=0.015, sigma=0.222777, T=0.3342,
                     I=0.015 * 100,
                     P=103.0,  # 回售触发价
                     buyback_cond=0.7):
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
                             columns=['strike', 'status'],
                             index=k.index)['strike']
            V.loc[idx] = [MockStrikePrice.cal_value(st, kt, r, sigma, T, I) for st, kt in zip(s, dwn_k)]
        return S, K, V

    @staticmethod
    def redeem_value(strike, stock, tc, r, interest_func):
        """

        :param strike:
        :param stock:
        :param tc:
        :param interest_func:  截止时间tc获得利息
        :return:
        """
        n = 100 / strike
        V = np.exp(-r * tc) * (n * stock + interest_func(tc))  # 贴现率 * 当期价值
        return V

    @classmethod
    def no_redeem_value(cls, Bc, strike: np.array, stock: np.array, tc, r, interest_func):
        n = 100 / strike[-1]
        n * stock[-1]
        cash_flow = max(Bc, n * stock)

        pass

    # @staticmethod
    # def no_redeem_value(Bc, strike, stock, T, r, interest_func):
    #     """
    #
    #     :param Bc:
    #     :param strike:
    #     :param stock:
    #     :param T:
    #     :param interest_func:
    #     :return:
    #     """
    #     n = 100 / strike
    #     V = np.exp(-r * T) * (max(Bc, n * stock) + interest_func(T))
    #     return V

    @classmethod
    def check_path_value(cls, S, K, V, Bc, redeem=1.3, interest_func=lambda x: 0.015 * 100):
        redeem_mask = (S > redeem * K) * 1  # True redeemed, 符合赎回条件的路径
        redeem_status = redeem_mask.cumsum(axis=1) == 1  # 判断首次首次出现的位置
        no_redeem = []
        for path, rede in enumerate(redeem_status.values):
            f = np.argwhere(rede == True)  # Find the indices of array elements that are non-zero, grouped by element.
            if len(f) > 0:
                # redeem
                print(path, f[0][0])  # 取第一次出现符合赎回条件的时间点
                t = f[0][0]
                tc = t / 200
                st = S.iloc[path, t]
                kt = K.iloc[path, t]
                yield path, cls.redeem_value(kt, st, tc, r, interest_func)

            else:  # not redeem
                no_redeem.append([path, rede])

        path = list(zip(*no_redeem))[0]
        no_redeem_mask = S.index.isin(path)
        cS = S[no_redeem_mask]
        cK = K[no_redeem_mask]
        for p, v in LSM.method(cS, cK, r, Bc, interest_func):
            yield p, v

    @classmethod
    def mock(cls, st, xt, r, sigma, T, n, I, P, buyback_cond, redeem, Bc, interest_func):
        S = Mock.init_stock_price(st, r, sigma, T, n=n)
        V = Mock.init_value(n)
        K = Mock.init_strike_price(xt, n=n)
        S, K, V = Mock.check_dwnwrd(S, K, V, r=r, sigma=sigma, T=T,
                                    I=I,
                                    P=P,  # 回售触发价
                                    buyback_cond=buyback_cond)

        c = pd.DataFrame(list(Mock.check_path_value(S, K, V, Bc, redeem=redeem, interest_func=interest_func)),
                         columns=['path', 'value'])

        return S, K, V, c

    @classmethod
    def mock_avg(cls, *args, **kwargs):
        S, K, V, res = cls.mock(*args, **kwargs)
        return res['value'].mean()


class LSM(object):
    @classmethod
    def method(cls, cS, cK, r, Bc, interest_func, **kwargs):
        """

        :param cS:
        :param cK:
        :param r:
        :param Bc:
        :param interest_func:
        :param kwargs:
        :return:
        """
        days = cS.columns
        # tm = int(days[-1].strip('day_'))
        CashFlow, Time = cls.step1(Bc, cS, cK, days)  # init

        Z = (100 / cK) * cS
        sorted_days = sorted(days, reverse=1, key=lambda x: int(x.strip('day_')))
        front, end = sorted_days[1:], sorted_days[:-1]

        dt = 1
        for dayN1, dayN in zip(front, end):  # mapping every mock period
            # cfn = CashFlow  # skip N=200
            day_S = cS[dayN1]
            day_K = cK[dayN1]
            mask = day_S > day_K

            if day_S[mask].empty:
                #
                msg = "no S >K day, will not change cashflow"
                print(msg)
                pass
            else:
                y = np.exp(-r * dt) * (CashFlow + interest_func(dt))
                tg_y = y[mask]  # filter S > K day;
                tg_z = Z[dayN1][mask]
                tg_z2 = np.square(tg_z)
                yz = pd.concat([tg_y, tg_z, tg_z2], axis=1)
                yz.columns = ['y', 'z', 'z2']

                model = OLS.from_formula('y ~ 1 + z + z2', data=yz).fit()
                EYZ = model.fittedvalues

                ez_mask = tg_z > EYZ
                path_idx = tg_z[ez_mask].index
                if path_idx.empty:
                    print('not update cashflow and time!')
                else:
                    n1 = int(dayN1.strip('day_'))
                    for path in path_idx:
                        update = tg_z[path]
                        CashFlow[path] = update
                        Time[path] = n1
        CashFlowTime = pd.concat([CashFlow, Time], axis=1)
        for path, (cf, t) in CashFlowTime.iterrows():
            y = np.exp(-r * t) * (cf + interest_func(t))
            yield path, y

    @staticmethod
    def step1(Bc, sn, xn, days):
        """

        :param Bc:
        :param sn:
        :param xn:
        :param tm: period
        :return:
        """

        day_int_list = list(map(lambda x: int(x.strip('day_')), days))
        tm = max(day_int_list)
        n = 100 / xn[f'day_{tm}']
        cash_flow = np.maximum(Bc, n * sn[f'day_{tm}'])
        Time = pd.Series([tm] * xn.shape[0], index=n.index)  # init
        return cash_flow, Time

    @classmethod
    def run(cls, Bc, st, r, sigma, T, buyback_cond, redeem, P, I, interest_func: object, n=(1000, 200), avg=True):
        S = Mock.init_stock_price(st, r, sigma, T, n=n)
        V = Mock.init_value(n)
        K = Mock.init_strike_price(xt, n=n)
        S, K, V = Mock.check_dwnwrd(S, K, V, r=r, sigma=sigma, T=T, I=I, P=P,  # 回售触发价
                                    buyback_cond=buyback_cond)
        mock_c = pd.DataFrame(list(Mock.check_path_value(S, K, V, Bc, redeem=redeem, interest_func=interest_func)),
                              columns=['path', 'value']).set_index('path').sort_index()
        if avg:
            return mock_c['value'].mean()
        else:
            return mock_c


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
    redeem = 3  # 1.3
    # S = Mock.init_stock_price(st, r, sigma, T, n=n)
    # V = Mock.init_value(n)
    # K = Mock.init_strike_price(xt, n=n)
    # S, K, V = Mock.check_dwnwrd(S, K, V, r=r, sigma=sigma, T=T, I=I, P=P,  # 回售触发价
    #                             buyback_cond=buyback_cond)
    # mock_c = pd.DataFrame(list(Mock.check_path_value(S, K, V, Bc, redeem=redeem, interest_func=interest_func)),
    #                       columns=['path', 'value']).set_index('path').sort_index()
    # c_avg = mock_c['value'].mean()
    c_avg = LSM.run(Bc, st, r, sigma, T, buyback_cond, redeem, P, I, interest_func, n=n, avg=True)
    print(1)
