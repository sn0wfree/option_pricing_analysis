import unittest

from option_pricing.Model.ZL import MockStrikePrice


class MyTestCaseMockStrikePriceDwnAdj(unittest.TestCase):
    def test_DwnAdj(self):
        # 九州转债 @ 20210915
        r = 0.015  # 无风险利率
        sigma = 0.222777  # 波动率
        T = 0.3342  # duration
        st = 16.00  # 正股价格
        xt = 17.8300  # 转股价格
        n = (100000, 200)  # （模拟次数,模拟步长）
        I = 0.015 * 100  # 利息
        P = 103.0  # 回售触发价，回售条款 30/30,70%,103
        buyback_cond = 0.7
        # print(st < xt * 0.7)  #
        d_kt, status = MockStrikePrice.check_wthr_dwnwrd_adj(st, xt, r, sigma, T, I, P, buyback_cond)
        print(status)
        self.assertEqual(d_kt, xt)
        self.assertEqual(status, 'non-active')


if __name__ == '__main__':
    unittest.main()
