# coding=utf-8
import pandas as pd

from CodersWheel.QuickTool.file_cache import file_cache


class WindHelper(object):
    """
        A helper class for interacting with the Wind financial database.
    """

    def __init__(self):
        from WindPy import w
        w.start()
        self.w = w

    @file_cache(enable_cache=True, granularity='d')
    def get_data(self, code, start, end, fields):
        """
        Generic method to fetch data from the Wind database.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :param fields: The fields to retrieve (e.g., 'close', 'high').
        :return: DataFrame with the requested data.
        """
        error, data_frame = self.w.wsd(code, fields, start, end, "", usedf=True)
        if error != 0:
            raise ValueError(f"Wind API error: {error}")
        return data_frame

    def get_close_prices(self, code, start, end):
        """
        Retrieve closing prices for a given code and date range.

        :param code: The stock or instrument code.
        :param start: The start date for data retrieval.
        :param end: The end date for data retrieval.
        :return: DataFrame with closing prices.
        """
        return self.get_data(code, start, end, 'close')

    def wind_wsd_high(self, code: str, start: str, end: str):
        return self.get_data(code, start, end, "high")

    def wind_wsd_low(self, code: str, start: str, end: str):
        return self.get_data(code, start, end, "low")

    def wind_wsd_open(self, code: str, start: str, end: str):
        """

        :param code:
        :param start:
        :param end:
        :return:
        """
        return self.get_data(code, start, end, "open")

    def wind_wsd_volume(self, code: str, start: str, end: str):
        """

        :param code:
        :param start:
        :param end:
        :return:
        """
        return self.get_data(code, start, end, "volume")

    def wind_wsd_quote(self, code: str, start: str, end: str, required_cols=('open', 'high', 'low', 'close', 'volume')):
        """

        :param code:
        :param start:
        :param end:
        :param required_cols:
        :return:
        """

        res_generator = (self.get_data(code, start, end, col) for col in required_cols)
        df = pd.concat(res_generator, axis=1)
        df['symbol'] = code

        return df

    def wind_wsd_quote_reduce(self, code: str, start: str, end: str, required_cols=('close', 'volume')):
        df = self.wind_wsd_quote(code, start, end, required_cols=required_cols)
        return df

    @file_cache(enable_cache=True, granularity='d')
    def get_future_info_last_delivery_date_underlying(self, code_list=None,
                                                      date_fields=['EXE_ENDDATE', 'LASTDELIVERY_DATE', 'FTDATE_NEW',
                                                                   'STARTDATE'],
                                                      multiplier_fields=['CONTRACTMULTIPLIER'],
                                                      underlying_code=['UNDERLYINGWINDCODE'],
                                                      futures_margin=['MARGIN']
                                                      ):
        """
        Retrieve future information including last delivery dates and contract multipliers.

        :param code_list: List of future contract codes.
        :return: DataFrame with future information.
        """
        if code_list is None:
            code_list = ["IF2312.CFE"]

        cols_str = ",".join(date_fields + multiplier_fields + underlying_code + futures_margin).lower()
        # "ftdate_new,startdate,lastdelivery_date,exe_enddate,contractmultiplier"

        err, last_deliv_and_multi = self.w.wss(','.join(code_list), cols_str, usedf=True)

        if err != 0:
            raise ValueError(f"Wind API error: {err}")

        for field in date_fields:
            last_deliv_and_multi[field] = last_deliv_and_multi[field].replace('1899-12-30', None)
            # last_deliv_and_multi[field] = pd.to_datetime(last_deliv_and_multi[field], errors='coerce')

        last_deliv_and_multi['EXE_DATE'] = last_deliv_and_multi['EXE_ENDDATE'].combine_first(
            last_deliv_and_multi['LASTDELIVERY_DATE'])
        last_deliv_and_multi['START_DATE'] = last_deliv_and_multi['STARTDATE'].combine_first(
            last_deliv_and_multi['FTDATE_NEW'])

        return last_deliv_and_multi


if __name__ == '__main__':
    from WindPy import w

    w.start()
    error, zz1000_15m = w.wsi("000852.SH", "open,high,low,close,volume,amt,chg,pct_chg,begin_time,end_time",
                       "2022-07-22 08:59:59",
                       "2023-01-03 15:00:03", "RSI_N=6;BarSize=15", usedf=True)

    zz1000_15m.to_excel('zz1000_15m_20220722_20230103.xlsx')
    pass
