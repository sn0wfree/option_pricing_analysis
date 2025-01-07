# coding=utf-8
from collections import namedtuple

import pandas as pd


class TempData(object):
    @staticmethod
    def load_full_greek_caled_marked_rmed_60m(file_name='full_greek_caled_marked_rmed_60m.parquet'):
        full_greek_caled_marked_rmed_60m = pd.read_parquet(file_name)

        full_greek_caled_marked_rmed_60m['cost'] = full_greek_caled_marked_rmed_60m['fee'] / (
                full_greek_caled_marked_rmed_60m['f'] * full_greek_caled_marked_rmed_60m['Delta'])

        return full_greek_caled_marked_rmed_60m

    @staticmethod
    def create_matrix(df):
        Delta = df.pivot_table(index='dt', columns='contract_code', values='Delta')
        Gamma = df.pivot_table(index='dt', columns='contract_code', values='Gamma')

        f = df.pivot_table(index='dt', values='f')
        k = df.pivot_table(index='dt', columns='contract_code', values='k')
        fee = df.pivot_table(index='dt', columns='contract_code', values='fee')
        cost = df.pivot_table(index='dt', columns='contract_code', values='cost')

        return f, k, fee, cost, Delta, Gamma

    @staticmethod
    def create_mask(df):
        day_end_mask = df['dt'].dt.hour == 15
        put_mask = df['cp'] == 'P'
        fee_mask = df['fee'] <= 300

        cost_mask = df['cost'].abs() <= 0.05

        otm_mask = df['f'] > df['k']

        return day_end_mask, put_mask, fee_mask, cost_mask, otm_mask


class Items(object):
    def __init__(self, dt: str, share: int, amt: float, direct: str, code: str, **kwargs):
        self._named_tuple_cls = namedtuple('init', ['dt', 'share', 'amt', 'direct', 'code', 'kwargs'])
        self.init_info = self._named_tuple_cls(dt, share, amt, direct, code, kwargs)

        self._record = []

    def add_record(self, dt, share, amt, direct, code, **kwargs):
        self._record.append(self._named_tuple_cls(dt, share, amt, direct, code, kwargs))

    @property
    def underlying_delta(self):
        return self._named_tuple_cls['share']

    @property
    def record_delta(self, dt, Delta_df):
        single_day_df = Delta_df.loc[dt, :]

        h = []

        for record in self._record:
            delta = single_day_df[record['code']]
            h.append(record['share'] * delta * 100)
        print(1)

    @property
    def Delta(self):
        underlying_delta = self._named_tuple_cls['share']


if __name__ == '__main__':
    init_cost = 10000000
    """
        # 规则
        1. 选择cost小于5%的
        2. delta最大
        3. 虚值期权
        4. 只买put
        5. 根据整体的收益加put，每涨1%，加 涨1%的50%put
        """

    full_greek_caled_marked_rmed_60m = TempData.load_full_greek_caled_marked_rmed_60m(
        file_name='full_greek_caled_marked_rmed_60m.parquet')

    day_end_mask, put_mask, fee_mask, cost_mask, otm_mask = TempData.create_mask(full_greek_caled_marked_rmed_60m)

    f, k, fee, cost, Delta, Gamma = TempData.create_matrix(full_greek_caled_marked_rmed_60m[day_end_mask & put_mask])

    buy_signal = f.pct_change(1) >= 0.01

    protect_multi = 0.5

    ihoder = Items('2022-07-22', 1421, 1421 * 7034.5975, 'BUY', '000852.SH')

    for dt, signal in buy_signal.iterrows():
        if signal['f']:
            dt_mask = full_greek_caled_marked_rmed_60m['dt'] == dt

            selected = full_greek_caled_marked_rmed_60m[dt_mask & put_mask & cost_mask & otm_mask & fee_mask]

            # get_contract
            s = selected[selected['k'] == selected.groupby('dt')['k'].max().values[0]]

            required_cols = ['Delta', 'Gamma']

            share = 1
            amt = share * s.loc[s.index, 'fee'].values[0] * 100
            code = s['contract_code'].values[0]

            kwargs = s[required_cols].to_dict('records')[0]

            ihoder.add_record(dt, share, amt, 'BUY', code, **kwargs)

    ihoder.record_delta('2024-07-12',Delta)

    print(1)
    pass
