# coding=utf-8
import operator
from functools import reduce

import numpy as np


# 拉格朗日插值（Lagrange Interpolation）


class Interpolate(object):

    @classmethod
    def interpolate(cls, x, y, method: str, new_x=None, ):
        if new_x is None:
            new_x = x
        learn_func = getattr(cls, method)

        func = learn_func(x, y)

        return func(new_x)

    @staticmethod
    def lagrange_interpolation_learn(x, y):
        n = len(x)
        # 拉格朗日插值法
        #  长度大于1 且对齐
        assert len(x) > 1 and (len(x) == len(y))

        def L(i, x_val):
            # 计算第i个拉格朗日基多项式
            term = 1
            for j in range(n):
                if i != j:
                    term *= (x_val - x[j]) / (x[i] - x[j])
            return term

        # 返回插值多项式函数
        def polynomial(x_val):
            return sum(y[i] * L(i, x_val) for i in range(n))

        return polynomial

    # @staticmethod
    # def difference_quotient(x_val, y_val):
    #     assert len(x_val) == len(y_val)
    #     n = len(x_val)
    #
    #     # 初始化差商数组
    #     d = np.zeros(n)
    #     d[0] = y_val[0]
    #
    #     # 计算差商
    #     for j in range(1, n):
    #         for i in range(j, 0, -1):
    #             d[i] = (d[i - 1] - d[i]) / (x_val[i] - x_val[i - j])
    #
    #     # 返回插值函数
    #     def interpolate(x):
    #         p = 0
    #         for i in range(n):
    #             # 计算除x外其他点与当前点x_val[i-1]的差值的乘积
    #             prod = np.prod([x_val[k] - x_val[i - 1] for k in range(n) if k != i - 1])
    #             p += d[i] * (x - x_val[i - 1]) * prod
    #         return p
    #
    #     return interpolate


# 差商
def difference_quotient(x_val, y_val):
    assert len(x_val) == len(y_val)
    n = len(x_val)
    p = np.zeros((n, n + 1))
    p[:, 0] = x_val
    p[:, 1] = y_val
    for j in range(2, n + 1):
        p[j - 1: n, j] = (p[j - 1: n, j - 1] - p[j - 2: n - 1, j - 1]) / (x_val[j - 1: n] - x_val[: n + 1 - j])
    q = np.diag(p, k=1)
    return p, q


# 牛顿插值（Newton Interpolation）
def newton(x_val, y_val, x):
    _, q = difference_quotient(x_val, y_val)

    def basis(i):
        if i > 0:
            l_i = [(x - x_val[j]) for j in range(i)]
        else:
            l_i = [1]
        return reduce(operator.mul, l_i) * q[i]

    return sum(basis(i) for i in range(len(x_val)))


# 分段线性插值（Piecewise Linear Interpolation）
def piecewise_linear_interp(x_val, y_val, x_lst):
    x_lst = [x_lst] if isinstance(x_lst, (float, int)) else x_lst
    x_loc = np.searchsorted(x_val, x_lst)

    res_lst = []
    for x, i in zip(x_lst, x_loc):
        L_i = y_val[i - 1] * (x - x_val[i]) / (x_val[i - 1] - x_val[i]) + \
              y_val[i] * (x - x_val[i - 1]) / (x_val[i] - x_val[i - 1])
        res_lst.append(L_i)
    return res_lst


def cubic_hermite_interp(x_val, y_val, x_deriv, x_interp):
    x_lst = [x_interp] if isinstance(x_interp, (float, int)) else x_interp
    x_loc = np.searchsorted(x_val, x_lst)

    res_lst = []
    for x, i in zip(x_lst, x_loc):
        h_i = x_val[i] - x_val[i - 1]
        H_i = (1 + 2 * (x - x_val[i - 1]) / h_i) * ((x - x_val[i]) / h_i) ** 2 * y_val[i - 1] + \
              (1 + 2 * (x_val[i] - x) / h_i) * ((x - x_val[i - 1]) / h_i) ** 2 * y_val[i] + \
              (x - x_val[i - 1]) * ((x - x_val[i]) / h_i) ** 2 * x_deriv[i - 1] + \
              (x - x_val[i]) * ((x - x_val[i - 1]) / h_i) ** 2 * x_deriv[i]

        res_lst.append(H_i)

    return res_lst


def cubic_spline_interp(x_val, y_val, x_interp):
    x_lst = [x_interp] if isinstance(x_interp, (float, int)) else x_interp

    x_loc = np.searchsorted(x_val, x_lst)
    x_loc = np.clip(x_loc, 1, len(x_val) - 1)
    p, q = difference_quotient(x_val, y_val)

    n = len(x_val) - 1
    h_vec = x_val[1:] - x_val[:-1]
    _u_vec = h_vec[-1:] / (h_vec[:-1] + h_vec[1:])
    u_vec = _u_vec[1:]
    lam_vec = 1 - u_vec

    diag_mat = np.diag(u_vec, k=-1) + np.diag([2] * (n - 1)) + np.diag(lam_vec, k=1)
    d_vec = 6 * p[2:, 3]
    m_vec = np.insert(np.append(np.linalg.solve(diag_mat, d_vec), 0), 0, 0)

    res_lst = []
    for x, i in zip(x_lst, x_loc):
        h_i = x_val[i] - x_val[i - 1]
        S_i = m_vec[i - 1] * (x_val[i] - x) ** 3 / (6 * h_i) + m_vec[i] * (x - x_val[i - 1]) ** 3 / (6 * h_i) + \
              (y_val[i - 1] - 1 / 6 * m_vec[i - 1] * h_i ** 2) * (x_val[i] - x) / h_i + \
              (y_val[i] - 1 / 6 * m_vec[i] * h_i ** 2) * (x - x_val[i - 1]) / h_i
        res_lst.append(S_i)
    if len(res_lst) == 1:
        return res_lst[0]
    return res_lst


def interpolated(test_data, x_col, y_col, default_col, output_name='interpolated'):
    default_mask = test_data[default_col] == 1

    if test_data[default_mask].empty:
        test_data[output_name] = test_data[y_col]
        return test_data
    else:
        y_val = test_data.loc[~default_mask, y_col].values
        x_val = test_data.loc[~default_mask, x_col].values
        test_data.loc[~default_mask, output_name] = y_val
        x_interp = test_data.loc[default_mask, x_col].values.tolist()
        y_iterp = np.array(cubic_spline_interp(x_val, y_val, x_interp))
        test_data.loc[default_mask, output_name] = np.where(y_iterp < 0, 0, y_iterp)
    return test_data


def interpolated_recured(test_data, x_col, y_col, default_col, output_name='interpolated'):
    test_data = interpolated(test_data, x_col, y_col, default_col, output_name=output_name)
    v2 = (test_data[output_name] == 0) * 1

    if v2.sum() == 0:

        return test_data
    else:
        print(v2.sum())
        test_data[default_col + '_r'] = v2

        return interpolated_recured(test_data, x_col, output_name, default_col + '_r', output_name=output_name)


if __name__ == '__main__':
    pass
