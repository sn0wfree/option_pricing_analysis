# coding=utf-8
import datetime
import hashlib
import os
import pickle
import sqlite3
# coding=utf-8
from collections import OrderedDict
from functools import wraps

import pandas as pd
from CodersWheel.QuickTool.file_cache import file_cache as fc

__refresh__ = False
DEFAULT = './'
format_dict = {'Y': '%Y', 'm': "%Y-%m", 'd': "%Y-%m-%d",
               'H': '%Y-%m-%d %H', 'M': '%Y-%m-%d %H:%M', 'S': '%Y-%m-%d %H:%M:%S'}


class CacheFunc(object):

    @staticmethod
    def get_cache_path(enable_cache: bool = False):
        dt = datetime.datetime.today().strftime('%Y%m%d')
        # __cache_path__ == f"{default}"
        cache_path = os.path.join(DEFAULT, dt)
        if not os.path.exists(cache_path) and enable_cache:
            os.mkdir(cache_path)
        return cache_path

    @staticmethod
    def date_format(granularity: str):
        if granularity in format_dict.keys():
            return format_dict.get(granularity)
        else:
            raise ValueError(f'date_format not support: {granularity}')

    @classmethod
    def prepare_args(cls, func, arg, kwargs: dict, granularity: str = 'H', exploit_func_name: bool = True,
                     enable_cache: bool = False):
        """

        :param func:
        :param arg:
        :param kwargs:
        :param granularity:
        :param exploit_func_name:  cache file name whether include func name
        :param enable_cache:
        :return:
        """
        time_format_dimension = cls.date_format(granularity)
        dt_str = datetime.datetime.now().strftime(time_format_dimension)
        kwargs = OrderedDict(sorted(kwargs.items(), key=lambda t: t[0]))  # sort kwargs to fix hashcode if sample input
        func_name = func.__name__.__str__()
        cls_obj = func.__qualname__ != func_name
        cls_name = func.__qualname__.split('.')[0] if cls_obj else None

        if len(arg) != 0:
            arg_cls_name = arg[0].__name__ if hasattr(arg[0], '__name__') else arg[0].__class__.__name__
        else:
            arg_cls_name = None

        if cls_obj and arg_cls_name is not None and arg_cls_name == cls_name:
            arg_tuple = tuple([cls_name] + list(map(str, arg[1:])))
        else:
            arg_tuple = arg

        key = pickle.dumps([func_name, arg_tuple, kwargs, dt_str])  # get the unique key for the same input
        if exploit_func_name:
            name = f"{func_name}_{hashlib.sha1(key).hexdigest()}_{dt_str}.cache"  # create cache file name
        else:
            name = hashlib.sha1(key).hexdigest() + '.cache'  # create cache file name
        file_path = cls.get_cache_path(enable_cache=enable_cache)
        return file_path, name

    @staticmethod
    def vacuum(sqlite_path):
        with sqlite3.connect(sqlite_path) as conn:
            vacuum_sql = 'VACUUM;'
            cursor = conn.cursor()
            cursor.execute(vacuum_sql)
            conn.commit()
            cursor.close()

    @staticmethod
    def write(dta, table, sqlite_path):
        with sqlite3.connect(sqlite_path) as conn:
            dta.to_sql(table, conn, if_exists='append', index=False)
            # 去重
            drop_duplicate_sql = f"DELETE FROM '{table}' WHERE rowid NOT IN (SELECT MIN(rowid) FROM '{table}' GROUP BY date,field);"
            cursor = conn.cursor()
            cursor.execute(drop_duplicate_sql)
            cursor.fetchall()

            conn.commit()
            cursor.close()

    @staticmethod
    def read(sqlite_path, sql, ):
        with sqlite3.connect(sqlite_path) as conn:
            res = pd.read_sql(sql, conn)
        return res

    @classmethod
    def quick_read(cls, arg, store_path):
        code, start, end, field = arg

        read_sql = f" select * from '{code}' where date >= '{start}' and date <= '{end}'  and field == '{field.upper()}' "

        cached_dta = cls.read(store_path, read_sql, )

        return cached_dta.drop_duplicates()

    @classmethod
    def slow_read(cls, arg, store_path):
        code, start, end, field = arg

        read_sql = f" select * from '{code}' where  field == '{field.upper()}' "

        cached_dta = cls.read(store_path, read_sql, )

        return cached_dta.drop_duplicates()

    @classmethod
    def fast_dt_read(cls, arg, store_path):
        code, start, end, field = arg

        read_sql = f" select distinct date   from '{code}' where  field == '{field.upper()}' "

        cached_dt = cls.read(store_path, read_sql, )

        cached_list = cached_dt['date'].unique().tolist()

        max_cached_dt = cached_dt['date'].max()
        min_cached_dt = cached_dt['date'].min()

        trading_days = cls.trading_days()
        try:
            missing_required_dts = sorted(
                filter(lambda x: (x >= start) & (x <= end) & (x not in cached_list), trading_days))
        except Exception as e:
            print(e)

        return cached_dt, max_cached_dt, min_cached_dt, missing_required_dts

    @staticmethod
    def detect_args(func, *arg):
        func_name = func.__name__.__str__()

        cls_obj = func.__qualname__ != func_name
        cls_name = func.__qualname__.split('.')[0] if cls_obj else None

        if len(arg) != 0:
            arg_cls_name = arg[0].__name__ if hasattr(arg[0], '__name__') else arg[0].__class__.__name__
        else:
            arg_cls_name = None

        if cls_obj and arg_cls_name is not None and arg_cls_name == cls_name:
            arg_tuple = arg[1:]
        else:
            arg_tuple = arg
        return arg_tuple

    @classmethod
    def new_write(cls, store_path, func, *arg, **kwargs):

        arg_tuple = cls.detect_args(func, *arg)

        code = arg_tuple[0]

        res = func(*arg, **kwargs)
        res.index.name = 'date'

        res_df = res.stack(-1).reset_index()
        res_df.columns = ['date', 'field', 'value']
        res_df['code'] = code
        res_df['field'] = res_df['field'].str.upper()

        cls.write(res_df, code, store_path)

        return res

    @staticmethod
    def res_2_res_df(res, min_cached_dt, code):
        if res.shape[0] == 1:
            res_df = res.stack(-1).reset_index()
            res_df.columns = ['code', 'field', 'value']
            res_df['date'] = min_cached_dt
        else:
            res_df = res.stack(-1).reset_index()
            res_df.columns = ['date', 'field', 'value']
            res_df['code'] = code
            res_df['field'] = res_df['field'].str.upper()
        return res_df
        # cls.write(res_df, code, store_path)

    @staticmethod
    @fc(enable_cache=True, granularity='d')
    def trading_days():

        import akshare as ak

        tool_trade_date_hist_sina_df = ak.tool_trade_date_hist_sina()

        return tool_trade_date_hist_sina_df['trade_date'].map(lambda x: x.strftime("%Y-%m-%d")).values.tolist()

    @classmethod
    def partial_update_write(cls, store_path, func, *arg, retry_count=0, max_retry_count=2, **kwargs):
        arg_tuple = cls.detect_args(func, *arg)

        cached_dt, max_cached_dt, min_cached_dt, missing_required_dts = cls.fast_dt_read(arg_tuple, store_path, )

        code, start, end, field = arg_tuple

        # max_cached_dt, min_cached_dt = cached_dta['date'].max(), cached_dta['date'].min()

        ## 修正end,修正期货合约结束,添加交易日判断

        if max_cached_dt >= end and min_cached_dt <= start and retry_count <= max_retry_count and len(
                missing_required_dts) == 0:

            cached_dta = cls.quick_read(arg_tuple, store_path, )

            mask = (cached_dta['date'] >= start) & (cached_dta['date'] <= end)

            return cached_dta[mask].drop_duplicates().pivot_table(index='date', columns='field', values='value')
        else:
            if retry_count <= max_retry_count:

                if len(missing_required_dts) != 0:
                    new_start = min(missing_required_dts)
                    new_end = max(missing_required_dts)

                    if len(arg) != len(arg_tuple):

                        res = func(arg[0], code, new_start, new_end, field,
                                   **kwargs).dropna()

                    else:

                        res = func(code, new_start, new_end, field, **kwargs)

                    res.index.name = 'date'

                    res_df = cls.res_2_res_df(res, min_cached_dt, code)
                    cls.write(res_df, code, store_path)
                    retry_count += 1

                if max_cached_dt < end:

                    new_end = pd.to_datetime(max_cached_dt) - pd.DateOffset(weeks=2)

                    if len(arg) != len(arg_tuple):

                        res = func(arg[0], code, new_end.strftime("%Y-%m-%d"), end, field, **kwargs)

                    else:

                        res = func(code, new_end.strftime("%Y-%m-%d"), end, field, **kwargs)
                    res.index.name = 'date'

                    res_df = cls.res_2_res_df(res, min_cached_dt, code)

                    cls.write(res_df, code, store_path)

                    retry_count += 1

                if min_cached_dt > start:
                    new_start = pd.to_datetime(start) - pd.DateOffset(weeks=2)
                    if len(arg) != len(arg_tuple):

                        res = func(arg[0], code, new_start.strftime("%Y-%m-%d"), min_cached_dt, field,
                                   **kwargs).dropna()

                    else:

                        res = func(code, new_start.strftime("%Y-%m-%d"), end, field, **kwargs)

                    res.index.name = 'date'

                    res_df = cls.res_2_res_df(res, min_cached_dt, code)
                    cls.write(res_df, code, store_path)

                    retry_count += 1

                cached_dta = cls.quick_read(arg_tuple, store_path, )
                mask = (cached_dta['date'] >= start) & (cached_dta['date'] <= end)

                return cached_dta[mask].drop_duplicates().pivot_table(index='date', columns='field', values='value')

                # return cls.partial_update_write(store_path, func, *arg, retry_count=retry_count,
                #                                 max_retry_count=max_retry_count, **kwargs)
            elif retry_count > max_retry_count:
                cached_dta = cls.quick_read(arg_tuple, store_path, )
                mask = (cached_dta['date'] >= start) & (cached_dta['date'] <= end)

                return cached_dta[mask].drop_duplicates().pivot_table(index='date', columns='field', values='value')

            else:
                raise ValueError(f'date error: {start}, {end}, [{min_cached_dt},  {max_cached_dt}]')

    @classmethod
    def _cache(cls, func, arg, kwargs, enable_cache: bool = False, store_path='./quote_cache.sqlite'):
        """

        :param func:
        :param arg:
        :param kwargs:
        :param granularity:
        :param enable_cache:
        :param exploit_func:   cache file name whether include func name
        :return:
        """

        if enable_cache:
            # check code exists
            holding_sql = "SELECT name FROM sqlite_master WHERE type='table' ;"
            if os.path.exists(store_path):
                # detect exists

                code_list = cls.read(store_path, holding_sql, )['name'].unique()
            else:
                code_list = []

            arg_tuple = cls.detect_args(func, *arg)

            code = arg_tuple[0]

            if code not in code_list:

                return cls.new_write(store_path, func, *arg, **kwargs)

            elif code in code_list:

                cached_dta = cls.quick_read(arg_tuple, store_path, )

                if cached_dta.empty:
                    return cls.new_write(store_path, func, *arg, **kwargs)
                else:
                    return cls.partial_update_write(store_path, func, *arg, **kwargs)
                    # 需要一个交易日期判断是否存在日期缺失

                    # 需要检查列名是否缺失
            else:
                raise ValueError('no code no data')

        else:
            res = func(*arg, **kwargs)
            return res


def sqlite_cache(**deco_arg_dict):
    def _deco(func):
        @wraps(func)
        def __deco(*args, **kwargs):
            return CacheFunc._cache(func, args, kwargs, **deco_arg_dict)

        # sqlite_path = deco_arg_dict['store_path']
        # CacheFunc.vacuum(sqlite_path=sqlite_path)

        return __deco

    return _deco


if __name__ == '__main__':
    from WindPy import w

    w.start()


    @sqlite_cache(enable_cache=True, store_path='quote_cache.sqlite')
    def get_data(code, start, end, fields):
        """
        Generic method to fetch data from the Wind database.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :param fields: The fields to retrieve (e.g., 'close', 'high').
        :return: DataFrame with the requested data.
        """
        error, data_frame = fc(enable_cache=True, granularity='d')(w.wsd)(code, fields, start, end, "", usedf=True)
        if error != 0:
            raise ValueError(f"Wind API error: {error}")
        return data_frame


    d = get_data('MO2412-P-6000.CFE', '2024-09-10', '2024-11-19', 'close')
    print(d)

    pass
