# coding=utf-8

import warnings

warnings.filterwarnings('ignore')
from ClickSQL import BaseSingleFactorTableNode


class DataQuoteCK(object):
    def __init__(self, src='clickhouse://default:Imsn0wfree@0.0.0.0:8123/system'):
        self._conn = BaseSingleFactorTableNode(src)
        pass

    def get_quote_via_sql(self, sql: str):
        return self._conn(sql)

    def get_mo_60m_quote_all(self, sql='select * from tick_minute_trade.MO_60m_trade'):
        return self.get_quote_via_sql(sql)
        pass


class DataQuote(DataQuoteCK):
    pass


class GSMain(object):
    # 如何维持delta中性
    # 如何维持gamma中性
    # 如何选择合约以及如何套利
    ## 如何置换合约,如何低买高卖

    # 如何确定买入时点和卖出时点
    ## 需要判断趋势和拐点
    ## 确定再平衡的周期
    pass



if __name__ == '__main__':
    DQCK = DataQuoteCK(src='clickhouse://default:Imsn0wfree@10.67.20.52:8123/system')

    MO_60m_trade = DQCK.get_mo_60m_quote_all(sql='select * from tick_minute_trade.MO_60m_trade limit 10')
    print(1)
    pass
