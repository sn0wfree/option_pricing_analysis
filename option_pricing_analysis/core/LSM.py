# coding=utf-8
# create LSM
import numpy as np
import pandas as pd

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


class LSMStockPathMock(object):
    @staticmethod
    def create_randn(n: int = (100000, 200)):
        zt = np.random.normal(0, 1, n)
        return zt

    @staticmethod
    def create_routes(r, sigma, T, randn_martix):
        dt = T / 200
        route = np.exp((r - (sigma ** 2) / 2) * dt + sigma * np.sqrt(dt) * randn_martix)
        return route

    @classmethod
    def create_sn(cls, sn, r, sigma, T, n: int = (100000, 200)):
        randn_martix = cls.create_randn(n)
        route = cls.create_routes(r, sigma, T, randn_martix)
        route = pd.DataFrame(route).cumprod(axis=1)
        return route * sn


class LSMStrikePriceAdjustedProvision(object):
    """
    首先，寻找在回售期
    """

    pass


class LSM(LSMStockPathMock):
    pass


if __name__ == '__main__':
    r, sigma, T = 0.01, 0.1, 5
    rt = LSM.create_sn(1, r, sigma, T, n=(100000, 200))
    pass
