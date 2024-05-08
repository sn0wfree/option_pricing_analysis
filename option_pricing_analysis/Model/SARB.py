# coding=utf-8
import warnings

import QuantLib as ql
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, minimize

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

"""
ref:  https://zhuanlan.zhihu.com/p/684313868

"""


def timeit(num):
    def sub_timeit(func):
        import time
        def _timeit(*args, **kwargs):
            start = time.time()

            for _ in range(num):
                res = func(*args, **kwargs)
            end = time.time()
            print(f'run {func.__name__} {num} times:', end - start)
            return res

        return _timeit

    return sub_timeit


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


class SABTBase(object):
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


class SABR_obloj(SABTBase):
    def __init__(self, F, K, T, imp_vol, opt_paras: dict):
        self.F = F
        self.K = K
        self.x = np.log(K / F)  # 远期在值程度
        self.T = T  # 到期时间
        self.imp_vol = imp_vol  # 隐含波动率

        # opt_paras：优化的alpha,rho,v参数，需要字典形式如，{"alpha":0.0005,"beta":1,"rho":-0.005,"v":0.01},这里加上beta作为可变的
        self.opt_paras = opt_paras

    # 实现定义好SABR的函数,就是前面定义过的函数,参数保留要优化的alpha,v,rho，方便后面代码实现
    def sabr_obloj_func(self, K, alpha, v, rho):

        F, beta, T = self.F, self.opt_paras["beta"], self.T

        return self.sabr_obloj_opt(F, alpha, beta, v, rho, K, T)

    def transform(self, y, l, u):
        return 0.5 * ((u + l) + (u - l) * np.tanh(y))

    def obj_func(self, x):
        y1, y2, y3 = x

        # transform后才是真正的alpha,v,rho
        # alpha 初始波动率
        # rho 相关系数
        # v 波动率的波动率
        alpha, v, rho = self.transform(y1, 0, 1), self.transform(y2, 0, 10), self.transform(y3, -1, 1)

        err = self.sabr_obloj_func(self.K, alpha, v, rho) - self.imp_vol

        return err

    def fit(self, x0=(-1, -2.5, 0.05)):
        # 初始值设定，这边就简单直接设定具体的值，可以根据transform的前后的值大概选取一个经验值

        result = least_squares(self.obj_func, x0, method="lm", verbose=0).x

        alpha, v, rho = self.transform(result[0], 0, 1), self.transform(result[1], 0, 10), self.transform(result[2], -1,
                                                                                                          1)
        self.opt_paras["alpha"], self.opt_paras["v"], self.opt_paras["rho"] = alpha, v, rho
        print(self.opt_paras)
        return self.opt_paras

    def predict(self, K, output="imp_vol"):

        sabr_imp_vol = self.sabr_obloj_func(K, self.opt_paras["alpha"], self.opt_paras["v"], self.opt_paras["rho"])

        if output == "imp_vol":
            return sabr_imp_vol
        elif output == "w":
            sabr_w = sabr_imp_vol ** 2 * self.T
            # 一般输出的是隐含波动率，不过有时候输出总方差也会更方便
            return sabr_w
        else:
            raise ValueError('unknown output!')


class sabr_surface(object):
    def __init__(self, data, opt_method):
        self.data = data
        self.opt_method = opt_method

    def get_fit_curve(self):
        fit_result = []

        # 循环每个月份，获得相应的拟合函数，返回一个包含svi实例的列表
        for month in self.data["contract_month"].unique():
            fit_option = self.data[(self.data["实虚值"] == "虚值") & (self.data["contract_month"] == month)]
            F = fit_option["F"].values[0]
            K = fit_option["exerciseprice"].values
            T = fit_option["maturity"].values[0]
            imp_vol = fit_option["market_imp_vol"].values
            opt_paras = {"alpha": 0.05, "beta": 1, "rho": -0.05, "v": 0.01}  # 先大概指定个初始值，除beta外后续会变化为最优解
            sabr = self.opt_method(F, K, T, imp_vol, opt_paras)
            sabr.fit()
            fit_result.append(sabr)

        return fit_result

    def plot_fit_curve(self):
        fit_result = self.get_fit_curve()
        fig, ax = plt.subplots(nrows=len(self.data["contract_month"].unique()), ncols=1, figsize=(8, 20))
        for i, month in enumerate(self.data["contract_month"].unique()):
            fit_option = self.data[(self.data["实虚值"] == "虚值") & (self.data["contract_month"] == month)]
            sabr = fit_result[i]
            fit_option["sabr_vol"] = sabr.predict(sabr.K, output="imp_vol")

            ax[i].scatter(x=fit_option["exerciseprice"], y=fit_option["market_imp_vol"], marker='+', c="r")
            ax[i].plot(fit_option["exerciseprice"], fit_option["sabr_vol"])
            ax[i].set_title(month)

    #  根据拟合的SABR函数，和平远期插值，生成100*100的隐含波动率网格
    def gen_imp_vol_grid(self):
        x = np.log(self.data["exerciseprice"] / self.data["F"])
        t_array = np.linspace(self.data["maturity"].min(), self.data["maturity"].max(), 100)
        x_array = np.linspace(x.min(), x.max(), 100)
        t, x = np.meshgrid(t_array, x_array)

        # 计算4个期限上的SABR拟合的总方差，并存储在100*4的矩阵里
        fit_result = self.get_fit_curve()
        w = np.zeros((100, len(fit_result)))

        for m in range(len(fit_result)):
            F_list = self.data["F"].unique()
            K = F_list[m] * np.exp(x_array)  # SABR需要用行权价而不是在值程度，这里返回求解K
            w[:, m] = fit_result[m].predict(K, output="w")

        # 在x的维度上循环100次，每次循环在t维度上平远期插值计算
        v = np.zeros_like(t)

        for n in range(100):
            f = interp1d(x=self.data["maturity"].unique(), y=w[n], kind="linear")
            v[n] = np.sqrt(f(t[n]) / t[n])  # 返回的还是隐含波动率而不是总方差

        return t, x, v

    def plot_surface(self):
        fig = plt.figure(figsize=(12, 7))
        ax = plt.axes(projection='3d')
        norm = mpl.colors.Normalize(vmin=0.1, vmax=0.2)
        # 绘图主程序
        t, x, v = self.gen_imp_vol_grid()
        surf = ax.plot_surface(t, x, v, rstride=1, cstride=1,
                               cmap=plt.cm.coolwarm, norm=norm, linewidth=0.5, antialiased=True)
        # 设置坐标轴
        ax.set_xlabel('maturity')
        ax.set_ylabel('strike')
        ax.set_zlabel('market_imp_vol')
        ax.set_zlim((0.1, 0.25))
        fig.colorbar(surf, shrink=0.25, aspect=5)


class SABRfromQuantlib(object):
    def __init__(self, K, F, T, iv: (list, tuple)):
        self._K = K
        self._F = F
        self._T = T
        self._iv = iv

        self._params = None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params
        pass

    @staticmethod
    def cal_vols(strikes, fwd, expiryTime, params):
        vols = np.array([
            ql.sabrVolatility(strike, fwd, expiryTime, *params)
            for strike in strikes
        ])
        return vols

    @classmethod
    def f(cls, strikes, fwd, expiryTime, marketVols, params):
        print(params)
        vols = cls.cal_vols(strikes, fwd, expiryTime, params)
        return ((vols - np.array(marketVols)) ** 2).mean() ** .5

    def loss_func(self, params=(0.1, 0.99, 0.1, 0.1)):
        #  alpha,  beta,  nu,  rho
        return self.f(self._K, self._F, self._T, self._iv, params)

    def fit(self, params=(0.1, 0.1, 0.1, 0.1), cons=(
            {'type': 'ineq', 'fun': lambda x: 0.99 - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[3]}
    )):
        result = minimize(self.loss_func, list(params), constraints=cons)
        print(result.status)
        self.params = result['x']

        newVols = self.cal_vols(self._K, self._F, self._T, self.params)
        return newVols

    def predict(self, strikes, ):
        return self.cal_vols(strikes, self._F, self._T, self.params)


if __name__ == '__main__':
    K = [105, 106, 107, 108, 109, 110, 111, 112]
    F = 120.44
    T = 17 / 365
    marketVols = [0.4164, 0.408, 0.3996, 0.3913, 0.3832, 0.3754, 0.3678, 0.3604]
    sabr_q = SABRfromQuantlib(K, F, T, marketVols)

    newVols = sabr_q.fit()

    print(1)

    pass
