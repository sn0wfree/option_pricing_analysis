# coding=utf-8
import datetime
import os
import subprocess
import sys
import time
import zipfile
from glob import glob

import pandas as pd
import warnings

def unzip(p, store='/tmp/test'):
    with zipfile.ZipFile(p) as zip_file:
        # 解压
        zip_extract = zip_file.extractall(store)


def progress_test(counts, lenfile, speed):
    bar_length = 20
    w = (lenfile - counts) * speed
    eta = time.time() + w
    precent = counts / float(lenfile)

    ETA = datetime.datetime.fromtimestamp(eta)
    hashes = '#' * int(precent * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("""\r%d%%|%s|read %d projects|Speed : %.4f |ETA: %s """ % (
        precent * 100, hashes + spaces, counts, speed, ETA))

    # sys.stdout.write("\rthis spider has already read %d projects, speed: %.4f/projects" % (counts,f2-f1))

    # sys.stdout.write("\rPercent: [%s] %d%%,remaining time: %.4f mins"%(hashes + spaces,precent,w))
    sys.stdout.flush()


def process_bar(iterable_obj, counts=None):
    if hasattr(iterable_obj, '__len__'):
        counts = len(iterable_obj)
    elif isinstance(counts, int):
        pass  # counts = counts
    else:
        iterable_obj = list(iterable_obj)
        counts = len(iterable_obj)
    for count, i in enumerate(iterable_obj):
        f = time.time()
        yield i

        progress_test(count, counts, time.time() - f)


class DownloadFromOptionQuote(object):
    @staticmethod
    def unzip_all(base_store_path='/home/liu.bo/Recorder/', unzip_store_path='/tmp/csv/'):
        tasks = [f'unzip -o {x} -d {unzip_store_path}' for x in glob(os.path.join(base_store_path, '*.zip'))]
        for task in process_bar(tasks):
            subprocess.Popen(task, shell=True)

    @staticmethod
    def unzip_update(base_store_path='/home/liu.bo/Recorder/', unzip_store_path='/tmp/csv/'):
        tasks = [f'unzip -o {x} -d {unzip_store_path}' for x in glob(os.path.join(base_store_path, '*.zip'))]
        for task in process_bar(tasks):
            subprocess.Popen(task, shell=True)

    @staticmethod
    def unzip_today(base_store_path='/home/liu.bo/Recorder/', unzip_store_path='/tmp/csv/'):
        now = datetime.datetime.now().strftime("%Y%m%d")
        x = os.path.join(base_store_path, f'{now}.zip')
        task = f'unzip -o {x} -d {unzip_store_path}'
        subprocess.Popen(task, shell=True)

    @staticmethod
    def scan_zip(store_path="/home/liu.bo/Recorder/"):
        zip_file = glob(os.path.join(store_path, '*.zip'))
        return dict(zip(map(lambda x: os.path.split(x)[-1].split('.')[0], zip_file), zip_file))

    @staticmethod
    def scan_exist(store_path="/tmp/csv/"):
        folder = glob(os.path.join(store_path, '*'))
        return dict(zip(map(lambda x: os.path.split(x)[-1], folder), folder))

    @classmethod
    def scan_compare(cls, store_path="/home/liu.bo/Recorder/", csv_path='/tmp/csv/'):
        if not os.path.exists(csv_path):
            os.path.mkdir(csv_path)

        zip_dict = cls.scan_zip(store_path=store_path)
        csv_dict = cls.scan_exist(store_path="/tmp/csv/")
        print(csv_dict.keys())
        for d, zip_path in process_bar(zip_dict.items()):
            if d not in csv_dict.keys():
                unzip(zip_path, csv_path)
                print(d, zip_path)

    @staticmethod
    def load_csv(f_p):
        # dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        df = pd.read_csv(f_p, parse_dates=[['TradeDate', 'UpdateTime']])  # .sort_values('UpdateTime')
        df = df.rename(columns={'TradeDate_UpdateTime': 'TradeDateTime'})
        return df

    @classmethod
    def load_upload(cls, conn, csv_path='/tmp/csv/',
                    sql='select distinct(toYYYYMMDD(TradeDateTime)) as dt from hq_quote.hq_quote_fut_der', force=False):
        csv_dict = cls.scan_exist(store_path=csv_path)

        dt_list = conn(sql)['dt'].unique().tolist()
        for dt, path in process_bar(csv_dict.items()):
            if not force:
                if dt not in dt_list:
                    for csv_sub_path in glob(os.path.join(path, '*.csv')):
                        try:
                            df = cls.load_csv(csv_sub_path)
                            conn.insert_df(df, 'hq_quote', 'hq_quote_fut_der', chunksize=10000)
                            print(csv_sub_path, ' uploaded!')
                            conn('optimize table hq_quote.hq_quote_fut_der final')
                        except Exception as e:
                            warnings.warn(csv_sub_path + ' load failure!')
            else:
                for csv_sub_path in glob(os.path.join(path, '*.csv')):
                    try:
                        df = cls.load_csv(csv_sub_path)
                        conn.insert_df(df, 'hq_quote', 'hq_quote_fut_der', chunksize=10000)
                        print(csv_sub_path, ' uploaded!')
                        conn('optimize table hq_quote.hq_quote_fut_der final')
                    except Exception as e:

                        warnings.warn(csv_sub_path + ' load failure!')

    @classmethod
    def auto(cls, conn_list, store_path="/home/liu.bo/Recorder/", csv_path='/tmp/csv/',
             sql='select distinct(toYYYYMMDD(TradeDateTime)) as dt from hq_quote.hq_quote_fut_der', force=False):
        cls.scan_compare(store_path=store_path, csv_path=csv_path)
        if not isinstance(conn_list, list):
            conn_list = [conn_list]
        for conn in conn_list:
            # print(conn)
            cls.load_upload(conn, csv_path=csv_path, sql=sql, force=force)

        pass


if __name__ == '__main__':
    # DownloadFromOptionQuote.auto(node, store_path="/home/liu.bo/Recorder/", csv_path='/tmp/csv/')
    pass
