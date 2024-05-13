# coding=utf-8
import warnings

import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

"""
ref:  https://zhuanlan.zhihu.com/p/684313868

"""
ZERO = 1e-8
BOUNDS = ((0 + ZERO, np.inf),
          (0 + ZERO, 1 - ZERO),
          (0 + ZERO, np.inf),
          (0 + ZERO, 1 - ZERO))

CONS = ({'type': 'ineq', 'fun': lambda x: x[0] - ZERO},
        {'type': 'ineq', 'fun': lambda x: 0.9995 - x[1] - ZERO},
        {'type': 'ineq', 'fun': lambda x: x[1] - ZERO},
        {'type': 'ineq', 'fun': lambda x: x[2] - ZERO},
        {'type': 'ineq', 'fun': lambda x: x[3] - ZERO},
        {'type': 'ineq', 'fun': lambda x: 0.9995 - x[3] - ZERO},
        )


def sabr(F, alpha, beta, v, rho, K, T):
    """
    F:当前远期价格
    alpha：F的波动率初始值
    beta：决定F的分布，可经验设置为0,0.5或1，也可当参数拟合
    v：波动率的波动率
    rho:相关系数
    K：行权价
    T；剩余期限
    """
    if F != K:  # 按照论文中的写法还原，但是if判断的话会导致无法向量化运算，下面会进行改进
        z = v / alpha * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
        xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        first_term = alpha / (
                (F * K) ** ((1 - beta) / 2) * (
                1 + (1 - beta) ** 2 / 24 * np.log(F / K) ** 2 + (1 - beta) ** 4 / 1920 * np.log(F / K) ** 4))
        second_term = 1 + T * ( \
                    (1 - beta) ** 2 / 24 * alpha ** 2 / (F * K) ** (1 - beta) + \
                    1 / 4 * rho * beta * v * alpha / (F * K) ** ((1 - beta) / 2) + \
                    (2 - 3 * rho ** 2) / 24 * v ** 2 \
            )

        bs_sigma = first_term * second_term * z / xz
    else:
        first_term = alpha / F ** (1 - beta)

        second_term = 1 + T * ( \
                    (1 - beta) ** 2 / 24 * alpha ** 2 / F ** (2 - 2 * beta) + \
                    1 / 4 * rho * beta * v * alpha / F ** (1 - beta) + \
                    (2 - 3 * rho ** 2) / 24 * v ** 2 \
            )
        bs_sigma = first_term * second_term

    return bs_sigma


class SABRCore(object):
    @staticmethod
    def sabr_obloj_opt(F, alpha, beta, v, rho, K, T):

        """
        F:当前远期价格
        alpha：F的波动率初始值
        beta：决定F的分布，可经验设置为0,0.5或1，也可当参数拟合
        v：波动率的波动率
        rho:相关系数
        K：行权价
        T；剩余期限
        obloj对hagan的解做了修正，对beta=0和F=K的特殊情况进行了改良，解决了beta趋近于1时的内部矛盾问题，优化了长期限和行权价格较小时的解的精度
        """

        F1_beta = F ** (1 - beta)
        K1_beta = K ** (1 - beta)
        FK1_beta = F1_beta * K1_beta
        FK1_beta_2 = np.sqrt(FK1_beta)
        logF_K = np.log(F / K)

        # 对于beta为1或者不为0的情况需要特别处理
        if beta == 1:
            z = v * logF_K / alpha
        else:
            z = v / alpha * (F1_beta - K1_beta) / (1 - beta)

        xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        first_term_F_eq_K = alpha / F1_beta
        first_term_F_ineq_K = v * logF_K / xz
        first_term = np.where(np.isclose(F, K, rtol=0, atol=0.1 ** 6), first_term_F_eq_K,
                              first_term_F_ineq_K)  # 对于beta为1或者不为0的情况需要特别处理

        # second_term和SABR原文hagan解是保持一致的
        second_term = 1 + T * (
                (1 - beta) ** 2 / 24 * alpha ** 2 / FK1_beta + 1 / 4 * rho * beta * v * alpha / FK1_beta_2 + (
                2 - 3 * rho ** 2) / 24 * v ** 2)
        bs_sigma = first_term * second_term
        return bs_sigma

    @staticmethod
    def sabr_obloj(F, alpha, beta, v, rho, K, T):
        """
        F:当前远期价格
        alpha：F的波动率初始值
        beta：决定F的分布，可经验设置为0,0.5或1，也可当参数拟合
        v：波动率的波动率
        rho:相关系数
        K：行权价
        T；剩余期限
        obloj对hagan的解做了修正，对beta=0和F=K的特殊情况进行了改良，解决了beta趋近于1时的内部矛盾问题，优化了长期限和行权价格较小时的解的精度
        """
        # 对于beta为1或者不为0的情况需要特别处理
        if beta == 1:
            z = v * np.log(F / K) / alpha
        else:
            z = v / alpha * (F ** (1 - beta) - K ** (1 - beta)) / (1 - beta)

        xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        first_term_F_eq_K = alpha / F ** (1 - beta)
        first_term_F_ineq_K = v * np.log(F / K) / xz
        first_term = np.where(np.isclose(F, K, rtol=0, atol=0.1 ** 6), first_term_F_eq_K,
                              first_term_F_ineq_K)  # 对于beta为1或者不为0的情况需要特别处理

        # second_term和SABR原文hagan解是保持一致的
        second_term = 1 + T * ( \
                    (1 - beta) ** 2 / 24 * alpha ** 2 / (F * K) ** (1 - beta) + \
                    1 / 4 * rho * beta * v * alpha / (F * K) ** ((1 - beta) / 2) + \
                    (2 - 3 * rho ** 2) / 24 * v ** 2 \
            )
        bs_sigma = first_term * second_term
        return bs_sigma

    @staticmethod
    # @timer
    def sabr_vector_opt(F, alpha, beta, v, rho, K, T):

        FK1_beta = (F * K) ** (1 - beta)
        FK1_beta_2 = np.sqrt(FK1_beta)
        logF_K = np.log(F / K)

        z = (v / alpha) * FK1_beta_2 * logF_K
        xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        first_term = alpha / (FK1_beta_2 * (
                1 + (1 - beta) ** 2 / 24 * logF_K ** 2 + (1 - beta) ** 4 / 1920 * logF_K ** 4) \
                              )

        second_term = 1 + T * ( \
                    (1 - beta) ** 2 / 24 * alpha ** 2 / FK1_beta + \
                    1 / 4 * rho * beta * v * alpha / FK1_beta_2 + \
                    (2 - 3 * rho ** 2) / 24 * v ** 2 \
            )
        # 其实F=K的时候就是少了z/xz这一项而已，如果用if判断，那么无法向量化计算，对速度影响很大,
        # 此外下面用了np.isclose取代==判断，防止浮点数==判断不准确的情况
        bs_sigma = np.where(np.isclose(F, K, rtol=0, atol=0.1 ** 6), first_term * second_term,
                            first_term * second_term * z / xz)
        return bs_sigma

        pass

    @staticmethod
    # @timer
    def sabr_vector(F, alpha, beta, v, rho, K, T):  # 支持向量化计算的代码
        """
        F:当前远期价格
        alpha：F的波动率初始值
        beta：决定F的分布，可经验设置为0,0.5或1，也可当参数拟合
        v：波动率的波动率
        rho:相关系数
        K：行权价
        T；剩余期限
        """
        z = v / alpha * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
        xz = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        first_term = alpha / (
                (F * K) ** ((1 - beta) / 2) * \
                (1 + (1 - beta) ** 2 / 24 * np.log(F / K) ** 2 + (1 - beta) ** 4 / 1920 * np.log(F / K) ** 4) \
            )
        second_term = 1 + T * ( \
                    (1 - beta) ** 2 / 24 * alpha ** 2 / (F * K) ** (1 - beta) + \
                    1 / 4 * rho * beta * v * alpha / (F * K) ** ((1 - beta) / 2) + \
                    (2 - 3 * rho ** 2) / 24 * v ** 2 \
            )
        # 其实F=K的时候就是少了z/xz这一项而已，如果用if判断，那么无法向量化计算，对速度影响很大,
        # 此外下面用了np.isclose取代==判断，防止浮点数==判断不准确的情况
        bs_sigma = np.where(np.isclose(F, K, rtol=0, atol=0.1 ** 6), first_term * second_term,
                            first_term * second_term * z / xz)
        return bs_sigma


class SABR_obloj(SABRCore):
    def __init__(self, K, F, T, imp_vol, opt_paras: dict = {"alpha": 0.0005, "beta": 1, "rho": -0.005, "v": 0.01}):
        self.F = F
        self.K = np.array(K)
        self.x = np.log(self.K / F)  # 远期在值程度
        self.T = T  # 到期时间
        self.imp_vol = np.array(imp_vol)  # 隐含波动率

        # opt_paras：优化的alpha,rho,v参数，需要字典形式如，{"alpha":0.0005,"beta":1,"rho":-0.005,"v":0.01},这里加上beta作为可变的
        self.opt_paras = opt_paras

    # 实现定义好SABR的函数,就是前面定义过的函数,参数保留要优化的alpha,v,rho，方便后面代码实现
    def sabr_obloj_func(self, K, alpha, beta, v, rho):
        F, T = self.F, self.T

        return self.sabr_obloj_opt(F, alpha, beta, v, rho, K, T)

    def transform(self, y, l, u):
        return 0.5 * ((u + l) + (u - l) * np.tanh(y))

    def transform_x(self, x):
        y1, y2, y3, y4 = x
        # transform后才是真正的alpha,v,rho
        # alpha 初始波动率
        # rho 相关系数
        # v 波动率的波动率
        alpha, beta, v, rho = self.transform(y1, 0, 1), self.transform(y2, 0, 1), self.transform(y3, 0,
                                                                                                 10), self.transform(y4,
                                                                                                                     -1,
                                                                                                                     1)
        return alpha, beta, v, rho

    def obj_func(self, x):
        alpha, beta, v, rho = x

        err = self.sabr_obloj_func(self.K, alpha, beta, v, rho) - np.array(self.imp_vol)

        return (err ** 2).mean() ** 0.5

    def fit(self, x0=(0.2, 1, 0.5, 0.05), cons=(
            {'type': 'ineq', 'fun': lambda x: 0.99 - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[3]}
    ),

            bounds=None

            ):
        # 初始值设定，这边就简单直接设定具体的值，可以根据transform的前后的值大概选取一个经验值
        alpha, beta, v, rho = self.transform_x(x0)

        result = minimize(self.obj_func, (alpha, beta, v, rho), bounds=bounds, constraints=cons)

        # result = least_squares(self.obj_func, x0, method="lm", verbose=0).x

        new_alpha, new_beta, new_v, new_rho = result.x  # self.transform_x()

        self.opt_paras["alpha"], self.opt_paras["beta"], self.opt_paras["v"], self.opt_paras[
            "rho"] = new_alpha, new_beta, new_v, new_rho

        # print(self.opt_paras)

        return self.opt_paras

    def predict(self, K=None, ):
        K = self.K if K is None else K

        sabr_imp_vol = self.sabr_obloj_func(K, self.opt_paras["alpha"], self.opt_paras["beta"], self.opt_paras["v"],
                                            self.opt_paras["rho"])

        return sabr_imp_vol


class SABRModelBase(object):
    __slot__ = ('_params', '_K', '_F', '_T', '_iv', '_result')

    def __init__(self, K, F, T, iv: (list, tuple)):
        self._K = K
        self._F = F
        self._T = T
        self._iv = iv

        self._params = None
        self._result = None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params

    @staticmethod
    def cal_vols(strikes, fwd, expiryTime, params, func=ql.sabrVolatility):
        vols = [func(strike, fwd, expiryTime, *params) for strike in strikes]
        return vols


class SABRfromQuantlib(SABRModelBase):
    __slot__ = ('_params', '_K', '_F', '_T', '_iv', '_result')

    @classmethod
    def _loss_func(cls, strikes, fwd, expiryTime, marketVols, params, func=ql.sabrVolatility):
        vols = cls.cal_vols(strikes, fwd, expiryTime, params, func=func)
        return ((np.array(vols) - np.array(marketVols)) ** 2).mean() ** .5

    def loss_func(self, params=(0.1, 0.99, 0.1, 0.1)):
        #  alpha,  beta,  nu,  rho
        return self._loss_func(self._K, self._F, self._T, self._iv, params, func=ql.sabrVolatility)

    def fit(self, params=(0.1, 0.1, 0.1, 0.1), bounds=None, cons=(
            {'type': 'ineq', 'fun': lambda x: 0.99 - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[3]}
    )):
        result = minimize(self.loss_func, params, bounds=bounds, constraints=cons)
        self._result = result
        self.params = result['x']

    def predict(self, strikes=None, ):
        if strikes is None:
            strikes = self._K
        return self.cal_vols(strikes, self._F, self._T, self.params)


class MappingSABR(object):

    @classmethod
    def cal_sarbr_model(cls, data, bounds, cons, params=(0.2, 0.999950, 0.5, 0.5), n=10):
        k, bs_iv, f, t = cls.create_kftiv(data, n=n)
        sabr_mod = SABRfromQuantlib(k, f, t, bs_iv)
        sabr_mod.fit(params=params, bounds=bounds, cons=cons)
        return sabr_mod

    @staticmethod
    def chunk_data(K, f, n=200):
        less_K = min(K[K <= f].sort_values().tail(n).values)
        more_K = max(K[K >= f].sort_values().head(n).values)
        return less_K, more_K

    @staticmethod
    def create_kftiv(data, n=1000):
        K = data['k']
        f = data['f'].unique().tolist()[0]
        t = data['t'].unique().tolist()[0]

        bs_iv_mask = ~(data['resid'].abs() > 0)
        sub_data = data[bs_iv_mask]

        k = sub_data['k']
        bs_iv = sub_data['bs_iv']

        return k, bs_iv, f, t

    @classmethod
    def _calibrating(cls, data, bounds, cons, params=(0.2, 0.99, 0.1, 0.1), n=10):
        sabr_mod = cls.cal_sarbr_model(data, bounds, cons, params=params, n=n)

        alpha, beta, nu, rho = sabr_mod.params

        raw_sabr_iv = sabr_mod.predict(data['k'])
        data['sabr_iv'] = raw_sabr_iv
        data['sabr_alpha'] = alpha
        data['sabr_beta'] = beta
        data['sabr_nu'] = nu
        data['sabr_rho'] = rho
        data['sabr_loss'] = sabr_mod.loss_func(sabr_mod.params)

        return data, sabr_mod.params

    # ql.sabrVolatility(106, 120, 17/365, alpha, beta, nu, rho)

    @classmethod
    def calibrating(cls, tasks_df, last_params=(0.2, 0.995, 0.1, 0.1),
                    cons=({'type': 'ineq', 'fun': lambda x: x[0]},),
                    bounds=((0.001, 0.7),  # alpha,
                            (0.01, 1),  # beta,
                            (0.01, 10),  # nu,
                            (0, 0.995))  # rho,
                    ):
        from functools import partial

        func = partial(cls._calibrating, bounds=bounds, cons=cons, params=last_params, n=5)
        h = []

        for (dt, cp, ym), data in tasks_df.groupby(['dt', 'cp', 'ym']):
            calibrated_data, params = func(data, )
            h.append(calibrated_data)
        res_bs_sabr_iv_df = pd.concat(h)
        return res_bs_sabr_iv_df

    @classmethod
    def calibrating_boost(cls, tasks_df, last_params=(0.2, 0.995, 0.1, 0.1),
                          cons=(),
                          bounds=((0.001, 0.7),  # alpha,
                                  (0.01, 1),  # beta,
                                  (0.01, 10),  # nu,
                                  (0, 0.995))):
        from functools import partial
        from CodersWheel.QuickTool.boost_up import boost_up

        func = partial(cls._calibrating, bounds=bounds, cons=cons, params=last_params, n=5)

        tasks = (data for (dt, cp, ym), data in tasks_df.groupby(['dt', 'cp', 'ym']))

        h,params = list(zip(*boost_up(func, tasks, star=False)))
        res_bs_sabr_iv_df = pd.concat(h)
        return res_bs_sabr_iv_df


if __name__ == '__main__':
    K = [105, 106, 107, 108, 109, 110, 111, 112]
    F = 120.44
    T = 17 / 365
    marketVols = [0.4164, 0.408, 0.3996, 0.3913, 0.3832, 0.3754, 0.3678, 0.3604]

    import time

    t1 = time.time()
    sabr_q = SABR_obloj(K, F, T, marketVols)
    sabr_q.fit()
    newVols = sabr_q.predict()

    t2 = time.time()
    sabr_q2 = SABRfromQuantlib(K, F, T, marketVols)
    sabr_q2.fit()
    newVols2 = sabr_q2.predict()
    t3 = time.time()
    print(t2 - t1, t3 - t2)
    print(1)

    pass
