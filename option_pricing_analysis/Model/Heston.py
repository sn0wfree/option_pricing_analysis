# coding=utf-8
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_call_price(F, K, T, r, sigma):
    d1 = (np.log(F / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return F * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(F, K, T, r, sigma):
    d1 = (np.log(F / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - F * norm.cdf(-d1)


def heston_vol(kappa, theta, sigma, rho, v0, F, K, T, cp, r=0, q=0, N=1000, M=10000):
    dt = T / N
    V = np.zeros((M, N + 1))
    S = np.zeros((M, N + 1))
    V[:, 0] = v0
    S[:, 0] = F

    for i in range(1, N + 1):
        Z1 = np.random.normal(size=M)
        Z2 = np.random.normal(size=M)
        W1 = np.sqrt(dt) * Z1
        W2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2)

        V[:, i] = V[:, i - 1] + kappa * (theta - V[:, i - 1]) * dt + sigma * np.sqrt(V[:, i - 1] * dt) * W1
        S[:, i] = S[:, i - 1] * np.exp((r - q - 0.5 * V[:, i - 1]) * dt + np.sqrt(V[:, i - 1] * dt) * W2)

    op_prices = np.maximum(S[:, -1] - K, 0)
    op_price = np.mean(op_prices) * np.exp(-r * T)

    # 使用Black-Scholes公式反推隐含波动率

    price_func = bs_call_price if cp in ['c', 'C', 'call', 'Call', '1', 1] else bs_put_price

    implied_vol = brentq(lambda x: price_func(F, K, T, r, x) - op_price, 0.01, 5.0)
    return implied_vol


# 目标函数：最小化混合模型隐含波动率与市场隐含波动率的误差
def _loss_func(params, K, F, T, market_iv):
    model_iv = np.array([heston_vol(params, F, k, T) for k in K])
    return np.sum((model_iv - market_iv) ** 2)


def objective(K, F, T, market_iv):
    def _objective(params):
        return _loss_func(params, K, F, T, market_iv)

    return _objective


if __name__ == '__main__':
    from scipy.optimize import minimize

    K = [105, 106, 107, 108, 109, 110, 111, 112]
    F = 120.44
    T = 17 / 365
    marketVols = [0.4164, 0.408, 0.3996, 0.3913, 0.3832, 0.3754, 0.3678, 0.3604]

    loss_func = objective(K, F, T, marketVols)

    # 使用优化算法校准参数
    result = minimize(objective, initial_params, args=(), method='Nelder-Mead')
    optimized_params = result.x
    pass
