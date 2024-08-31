# coding=utf-8
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CodersWheel.QuickTool.file_cache import file_cache

from Indicators import Sensibility

sensi_f = Sensibility()

from scipy.optimize import minimize, LinearConstraint


def cal_greek_with_given_f(row, f, DividendYield=0):
    UnderlyingPrice = f
    Strike = row['k']
    Volatility = row['bs_iv']
    Time2Maturity = row['t']
    RiskFreeRate = row['r']
    OptionType = row['cp']
    return sensi_f.getGreeks(UnderlyingPrice,
                             Strike,
                             Volatility,
                             Time2Maturity,
                             DividendYield,
                             RiskFreeRate,
                             OptionType, )


def cal_pnl(f, selected_with_weight):
    # call

    f_k = f - selected_with_weight['k']

    call_mask = selected_with_weight['cp'] == 'C'

    call_pnl = np.dot((f_k[call_mask] >= 0) * f_k[call_mask], selected_with_weight[call_mask]['weight'])

    # put
    put_mask = selected_with_weight['cp'] == 'P'
    put_pnl = np.dot((f_k[put_mask] <= 0) * f_k[put_mask] * -1, selected_with_weight[put_mask]['weight'])

    return call_pnl + put_pnl


class OptionPortfolioWithDT(object):
    __slots__ = ('_selected', '_multiply_num', '_weight', '_fake_f')

    def __init__(self, selected, multiply_num=100):
        if len(selected['dt'].unique().tolist()) != 1:
            raise ValueError('only accept cross data! dt must be same!')
        self._selected = selected
        self._multiply_num = multiply_num
        self._fake_f = CachedData.create_fake_greek_for_contract(selected)

        self._weight = []

    def __len__(self):
        return len(self.avaiable_contract)

    def get(self, contract_code):
        mask = self._selected['contract_code'] == contract_code
        return self._selected[mask]

    @property
    def call(self):
        call_mask = self._selected['cp'] == 'C'
        return self._selected[call_mask]

    @property
    def put(self):
        put_mask = self._selected['cp'] == 'P'
        return self._selected[put_mask]

    def level_put(self, level=1):
        put = self.put.sort_values('f_k_diff')
        cp = 'P'
        otm_mask = put['f_k_diff'] > 0 if cp == 'P' else put['f_k_diff'] < 0
        tt = put[otm_mask]
        return tt.head(level)['contract_code'].values[-1]

    def level_call(self, level=1):
        call = self.call.sort_values('f_k_diff', ascending=False)
        cp = 'C'
        otm_mask = call['f_k_diff'] > 0 if cp == 'P' else call['f_k_diff'] < 0
        tt = call[otm_mask]
        return tt.head(level)['contract_code'].values[-1]

    def get_level_put(self, level=1):
        code = self.level_put(level=level)
        return self.get(code)

    def get_level_call(self, level=1):
        code = self.level_call(level=level)
        return self.get(code)

    @property
    def f(self):
        return self._selected['f'].unique().tolist()[0]

    @property
    def avaiable_contract(self):
        return self._selected['contract_code'].unique().tolist()

    @property
    def avaiable_call_contract(self):

        return self.call['contract_code'].unique().tolist()

    @property
    def avaiable_put_contract(self):

        return self.put['contract_code'].unique().tolist()

    @property
    def avaiable_main_put_contract(self):
        put_mask = self._selected['cp'] == 'P'
        main_mask = self._selected['main_mark'] != 0
        #         avail_put = self.available_put_contract
        return self.reduce_quote(self._selected[put_mask & main_mask])

    @property
    def avaiable_main_call_contract(self):
        call_mask = self._selected['cp'] == 'C'
        main_mask = self._selected['main_mark'] != 0
        #         avail_put = self.available_put_contract
        return self.reduce_quote(self._selected[call_mask & main_mask])

    @staticmethod
    def reduce_quote(quote,
                     q_cols=['contract_code', 'f', 'k', 'fee', 'bs_iv', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho', ]):

        return quote[q_cols].set_index('contract_code')

    def _cal_delta_size(self, share, contract_code, f):
        code_mask = self._selected['contract_code'] == contract_code
        Delta = self._selected[code_mask]['Delta']
        return share * self._multiply_num * 100 * Delta * f

    def _cal_gamma_size(self, share, contract_code):
        code_mask = self._selected['contract_code'] == contract_code
        Gamma = self._selected[code_mask]['Gamma']
        return share * self._multiply_num * 100 * Gamma

    def _cal_vega_size(self, share, contract_code):
        code_mask = self._selected['contract_code'] == contract_code
        Vega = self._selected[code_mask]['Vega']
        return share * self._multiply_num * 100 * Vega

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, w):
        self._weight = w

    def portfolio_delta(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected['Delta']) * self._multiply_num * self.f

    def portfolio_gamma(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected['Gamma']) * self._multiply_num

    def portfolio_vega(self, w=None):
        w = self.weight if w is None else w
        return np.dot(w, self._selected['Vega']) * self._multiply_num

    def cal_greek_with_given_f(self, f, DividendYield=0):
        for _, row in self._selected.iterrows():
            yield cal_greek_with_given_f(row, f, DividendYield=DividendYield)

    def _create_portfolio(self, hedge_size, p1=0.5):

        p1q = self.get_level_put(level=1)
        p1s = required_share(hedge_size * p1, p1q['Delta'], p1q['f']).astype(int)
        yield p1q['contract_code'].values.tolist()[0], p1s.values.tolist()[0]

        p2q = self.get_level_put(level=2)
        p2s = required_share(hedge_size * p1 * 0.1 * -1, p2q['Delta'], p2q['f']).astype(int)
        yield p2q['contract_code'].values.tolist()[0], p2s.values.tolist()[0]

        c1q = self.get_level_call(level=1)
        c1s = required_share(hedge_size * p1 * 0.6 * 1, c1q['Delta'], c1q['f']).astype(int)
        yield c1q['contract_code'].values.tolist()[0], c1s.values.tolist()[0]

        c2q = self.get_level_call(level=2)
        c2s = required_share(hedge_size * p1 * 0.6 * 1, c2q['Delta'], c2q['f']).astype(int)
        yield c2q['contract_code'].values.tolist()[0], c2s.values.tolist()[0]

        c3q = self.get_level_call(level=3)
        c3s = required_share(hedge_size * p1 * 0.5 * 0.6 * -1, c3q['Delta'], c3q['f']).astype(int)
        yield c3q['contract_code'].values.tolist()[0], c3s.values.tolist()[0]

        c4q = self.get_level_call(level=4)
        c4s = required_share(hedge_size * p1 * 0.5 * 0.6 * -1, c3q['Delta'], c3q['f']).astype(int)
        yield c4q['contract_code'].values.tolist()[0], c4s.values.tolist()[0]

    def create_init_weight(self, hedge_size, p1=0.5):

        otm_selected = self._selected

        weight_df = pd.DataFrame(list(self._create_portfolio(hedge_size, p1=p1)), columns=['contract_code', 'weight'])
        for code in weight_df['contract_code']:
            w = weight_df[weight_df['contract_code'] == code]['weight'].values[0]
            otm_selected.loc[code, 'init_weight'] = w
        otm_selected['init_weight'] = otm_selected['init_weight'].fillna(0)

        # otm_selected['init_weight'] = self.create_init_weight(hedge_size)

        return otm_selected['init_weight']

    def create_greek_matrix_all(self):
        otm_selected = self._selected.copy(deep=True)
        Delta, Gamma, Vega, Theta, Rho = CachedData.create_greek_matrix_all(otm_selected, self._fake_f)

        return Delta, Gamma, Vega, Theta, Rho

    def run_opt(self, hedge_size, Delta, Gamma, Vega, Theta, Rho):
        # Delta, Gamma, Vega, Theta, Rho = self.create_greek_matrix_all()
        otm_selected = self._selected

        initial_weights = self.create_init_weight(hedge_size)

        res = OptBundle.run_opt(initial_weights, otm_selected, hedge_size, Delta, Gamma, Vega, Rho, method='SLSQP')

        otm_selected['weight'] = res.x

        return res


class CachedData(object):
    def __init__(self):
        pass

    @staticmethod
    def load_quote_greek(select='select.parquet', main_mark=1):
        selected = pd.read_parquet(select)
        masks = selected['main_mark'] >= main_mark
        mask_selected = selected[masks]
        mask_selected.index = mask_selected['contract_code']
        return mask_selected

    @staticmethod
    @file_cache(enable_cache=True, granularity='d', )
    def create_fake_greek_for_contract(mask_selected, fake_f_list=range(4100, 8000, 50)):
        # fake_f_list = sorted(mask_selected['k'].unique())
        h = []
        for fake_f in fake_f_list:
            for _, row_data in mask_selected.iterrows():
                dt = row_data['dt']
                contract_code = row_data['contract_code']
                current_fee = row_data['fee']
                fake_greeks = cal_greek_with_given_f(row_data, fake_f, DividendYield=0)
                h.append((fake_f, dt, contract_code, current_fee, *fake_greeks))

        fake_contract_code = pd.DataFrame(h,
                                          columns=['fake_f', 'dt', 'contract_code', 'current_fee', 'Delta', 'Gamma',
                                                   'Vega', 'Theta', 'Rho'])
        return fake_contract_code

    @staticmethod
    def create_fake_greek_matrix(selected_fake_contract_code, f_cols='fake_f'):
        fake_delta = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Delta')
        fake_gamma = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Gamma')
        fake_vega = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Vega')
        fake_rho = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Rho')
        fake_theta = selected_fake_contract_code.pivot_table(index='contract_code', columns=f_cols, values='Theta')

        return fake_delta.T, fake_gamma.T, fake_vega.T, fake_theta.T, fake_rho.T

    @staticmethod
    def create_greek_matrix(selected_contract_code):

        # selected_contract_code = selected_contract_code
        # print(selected_contract_code)
        delta = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                          values='Delta')
        gamma = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                          values='Gamma')
        vega = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                         values='Vega')
        rho = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                        values='Rho')
        theta = selected_contract_code.reset_index(drop=True).pivot_table(index='contract_code', columns='k',
                                                                          values='Theta')

        return delta.T, gamma.T, vega.T, theta.T, rho.T

    @classmethod
    def create_greek_matrix_all(cls, real, fake):
        r1 = real[['contract_code', 'f', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']]

        f1 = fake[['contract_code', 'fake_f', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']]
        f1.columns = ['contract_code', 'f', 'Delta', 'Gamma', 'Vega', 'Theta', 'Rho']

        greek_df = pd.concat([f1, r1], axis=0)
        return cls.create_fake_greek_matrix(greek_df, f_cols='f')

        pass

    @classmethod
    def create_fake_portfolio_greek(cls, fake_contract_code, weights, code_list, crrt_f):
        # crrt_f = fake_contract_code['crrt_f'].unique().tolist()[0]
        fake_delta, fake_gamma, fake_vega, fake_theta, fake_rho = cls.create_fake_greek_matrix(fake_contract_code)

        portfolio_delta = fake_delta.reindex(columns=code_list).dot(weights) * 100 * crrt_f
        portfolio_gamma = fake_gamma.reindex(columns=code_list).dot(weights) * 100
        portfolio_vega = fake_vega.reindex(columns=code_list).dot(weights) * 100
        portfolio_rho = fake_rho.reindex(columns=code_list).dot(weights) * 100
        portfolio_theta = fake_theta.reindex(columns=code_list).dot(weights) * 100

        return portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta, portfolio_rho


class OptBundle(object):

    @staticmethod
    def delta_exposure_constraint(weights, Delta, f, target_delta_exposure):
        return np.dot(weights, Delta.loc[f, :]) * 100 * f - target_delta_exposure

    @staticmethod
    def delta(weights, Delta, f, ):
        return np.dot(weights, Delta.loc[f, :]) * 100 * f

    @staticmethod
    def gamma_constraint(weights, Gamma, f):
        return np.sum(weights * Gamma.loc[f, :] * 100)

    @staticmethod
    def vega_constraint(weights, Vega, f):
        return np.sum(weights * Vega.loc[f, :] * 100)

    @staticmethod
    def fee(weights, fee):
        cost = np.sum(weights * fee * 100) * -1
        return cost

    # 定义目标函数和约束条件
    # 定义Delta曲线约束
    @staticmethod
    def delta_curve_stability(weights, delta_martix, f, price_range):
        merged = delta_martix.reindex(index=price_range).dot(weights)

        # print(merged, price_range)

        return np.mean(np.power(np.gradient(merged, price_range - f), 2))

    #     @staticmethod
    #     def delta_curve_stability(weights, delta_martix, f, price_range):
    #         merged = delta_martix.reindex(index=price_range).dot(weights)
    #         return  np.mean( np.power(np.gradient(merged, price_range - f),2))

    # 定义合约数约束条件
    @staticmethod
    def contract_count_constraint(lots, max_contracts):
        return max_contracts - np.count_nonzero(lots)

    @staticmethod
    # 定义PNL计算
    def calculate_pnl(lots, deltas, prices, current_price):
        return np.sum(deltas * (prices - current_price) * lots)

    @classmethod
    # 定义PNL曲线稳定性约束
    def pnl_stability_constraint(cls, lots, fee, f, selected, k_list, price_range, tolerance=10):
        selected['weight'] = np.int32(lots)

        pnls = pd.Series({f_: cal_pnl(f_, selected) - fee for f_ in price_range}).to_frame('pnl')
        # print(fee, pnls)

        # 定义行权价附近的价格区间
        sub_pnls = pnls[pnls.index.isin(price_range)]['pnl']
        # print('sub_pnls:', sub_pnls.shape)
        # dx = np.diff(price_range).mean()  # 计算平均间隔
        # 计算价格间隔
        if isinstance(price_range, (list, np.ndarray)) and len(price_range) > 1:
            dx = np.diff(price_range).mean()  # 使用价格的平均间隔
            stability_measure = np.abs(np.gradient(sub_pnls, dx))
        else:
            # 价格间隔为1的默认情况
            stability_measure = np.abs(np.gradient(sub_pnls, edge_order=1))

        return tolerance - np.max(stability_measure)

    @classmethod
    def objective(cls, weights, fee, delta, f, mask_selected, price_range):
        cost = cls.fee(weights, fee)

        delta_curve = cls.delta_curve_stability(weights, delta, f, price_range)

        pnl_curve = cls.pnl_stability_constraint(weights, cost, f, mask_selected, delta.index, price_range)

        num_contracts = np.count_nonzero(weights)  # 合约数量
        total_share = np.sum(np.abs(weights))

        print('num_contracts: ', num_contracts, 'total_share: ', total_share)

        return np.abs(cost) + (np.abs(delta_curve) + np.abs(pnl_curve)) ** 2

    @classmethod
    def run_opt(cls, initial_weights, mask_selected, target_delta_exposure, Delta, Gamma, Vega, Rho, method='SLSQP'):
        # create_fake_portfolio_greek(fake_contract_code, weight_df)

        f = mask_selected['f'].unique().tolist()[0]

        price_range = Delta[(Delta.index <= f + 100) & (Delta.index >= f - 200)].index

        fee = mask_selected['fee']

        total_share_constraint = LinearConstraint(np.ones(len(fee)), lb=-50, ub=50)

        # gamma_constraint_c = NonlinearConstraint(lambda w: cls.gamma_constraint(w, Gamma, f), lb=0, ub=10)

        # stability_constraint = {'type': 'ineq', 'fun': cls.delta_curve_stability, 'args': (Delta, f, price_range)}
        #
        # max_contracts = 8  # 设定最大合约数量

        c1 = (
            {'type': 'eq', 'fun': cls.delta_exposure_constraint, 'args': (Delta, f, target_delta_exposure)},
            # gamma_constraint_c,
            {'type': 'eq', 'fun': cls.gamma_constraint, 'args': (Gamma, f)},
            {'type': 'ineq', 'fun': cls.vega_constraint, 'args': (Vega, f)},
            total_share_constraint,
            #             stability_constraint,
            # {'type': 'ineq', 'fun': cls.fake_delta_upper_constraint, 'args': (Delta, f, price_range)}
        )

        objective = lambda w: cls.objective(w, fee, Delta, f, mask_selected, price_range)

        # 边界条件
        bounds = [(-50, 50)] * len(initial_weights)

        # 执行优化
        result = minimize(objective, initial_weights, constraints=c1, bounds=bounds, method=method)

        opt_weight = np.int32(result.x)

        # 打印结果
        print(result)
        print('损失函数：', objective(opt_weight))
        print('f:', f)

        print("优化的合约:", mask_selected.index.tolist())
        print("优化前的权重:", initial_weights.values)
        print("优化后的权重:", opt_weight)
        print("Delta:", cls.delta(opt_weight, Delta, f))
        print("Gamma:", cls.gamma_constraint(opt_weight, Gamma, f))
        print("Vega:", cls.vega_constraint(opt_weight, Vega, f))
        print('cost:', cls.fee(opt_weight, fee))
        return result


def required_share(require_hedged_size, delta, f):
    return require_hedged_size / delta / f / 100


def _cal_portfolio_greek(weight, Delta, Gamma, Vega, Theta, Rho):
    d = Delta.dot(weight).to_frame('Delta')
    g = Gamma.dot(weight).to_frame('Gamma')
    v = Vega.dot(weight).to_frame('Vega')
    t = Theta.dot(weight).to_frame('Theta')
    r = Rho.dot(weight).to_frame('Rho')

    return pd.concat([d, g, v, t, r], axis=1)


def draw_greek_surface_pic2(res, selected, num_plots=6, num_cols=3,
                            greek_alphabet=('Delta', 'Gamma', 'Vega', 'Theta', 'Rho')):
    f = selected['f'].unique().tolist()[0]
    fee = np.dot(res.x, selected['fee'])

    d = _cal_portfolio_greek(res.x, Delta, Gamma, Vega, Theta, Rho)

    num_rows = (num_plots + num_cols - 1) // num_cols  # 计算行数，确保每行5张图
    # 使用subplot创建图形排列
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_cols * num_rows), sharex=True, sharey=False)
    # pnl

    pnl = pd.Series({f_: cal_pnl(f_, selected) - fee for f_ in d.index}).to_frame('pnl')
    ax = axs.flatten()[0]
    ax.plot(pnl['pnl'])
    ax.set_title(f'PnL Surface-Portfolio', fontsize=10)
    ax.set_xlim(pnl.index.min(), pnl.index.max())
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('PnL')
    ax.axvline(x=f, color='r')  # 'r' 表示红色
    ax.axhline(y=0, color='b')  # 'b' 表示蓝色

    for c, greek in enumerate(greek_alphabet):
        ax = axs.flatten()[c + 1]
        ax.plot(d[greek])
        ax.set_title(f'{greek} Surface-Portfolio', fontsize=10)
        # ax.set_ylim(min_iv,max_iv)
        ax.set_xlim(d.index.min(), d.index.max())
        ax.set_xlabel('Strike Price')
        ax.set_ylabel(greek)
        ax.axvline(x=f, color='r')  # 'r' 表示红色
    #         ax.axhline(y=0, color='b')  # 'b' 表示蓝色

    # 隐藏多余的子图
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axs.flatten()[i])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # select one day
    start_dt = '2023-01-01'
    hedge_size = -500 * 10000  # 设定对冲市值
    draw_pic = True

    full_greek_caled_marked = pd.read_parquet('full_greek_caled_marked.parquet')
    full_greek_caled_marked['OTM'] = ((full_greek_caled_marked['f'] - full_greek_caled_marked['k']) *
                                      full_greek_caled_marked['cp'].replace({'C': -1, 'P': 1}).astype(int)) >= 0

    # 只选主力合约
    main_contract_mask = full_greek_caled_marked['t'] >= 0.02
    # select main contract
    main_mark = full_greek_caled_marked['main_mark'] >= 3
    start_dt_mask = full_greek_caled_marked['dt'] >= start_dt

    for dt, df in full_greek_caled_marked[main_mark & main_contract_mask & start_dt_mask].groupby('dt'):
        if dt >= pd.to_datetime(start_dt):
            # use 主力合约
            ym_mask = df['ym'] == df['ym'].min()
            selected = df[ym_mask]

            otm_selected = selected[selected['OTM']]
            otm_selected.index = otm_selected['contract_code']
            otm_selected.index = otm_selected.index.astype(str)
            # 这个设定为初始参数，方便快速配权

            op_portfolio = OptionPortfolioWithDT(otm_selected)
            Delta, Gamma, Vega, Theta, Rho = op_portfolio.create_greek_matrix_all()
            # res = OptBundle.run_opt(initial_weights, otm_selected, hedge_size, Delta, Gamma, Vega, Rho, method='SLSQP')
            res = op_portfolio.run_opt(hedge_size, Delta, Gamma, Vega, Theta, Rho)

            op_portfolio._selected['success'] = res.success
            op_portfolio._selected['msg'] = res.message

            if draw_pic:
                draw_greek_surface_pic2(res, op_portfolio._selected, num_plots=6, num_cols=3, )

    pass
