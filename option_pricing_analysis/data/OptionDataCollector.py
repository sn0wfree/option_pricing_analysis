# -*- coding:utf-8 -*-

import datetime
import numpy as np
import pandas as pd
import random
import sys
import time


class GenerateSinaAPIUrl(object):
    """ this is to generate the url of SinaAPI """

    # this is to generate the url of SinaAPI

    def __init__(self, generUrl='http://hq.sinajs.cn/list='):
        """
        initial the Sina API
        :param generUrl:
        """
        self.generUrl = generUrl
        pass

    def getGreekIndicatorsurl(self, OptionCode):
        """
        this function is for get url of the greek indicators
        :param OptionCode: option code
        :return: apiurl (greek indicator)
        """

        if isinstance(OptionCode, str):
            if 'CON_OP_' in OptionCode:
                sfGreek = OptionCode.replace('CON_OP_', 'CON_SO_')

                return self.generUrl + sfGreek
            else:
                raise ValueError('Unknown OptionCode:%s' % OptionCode)
        elif isinstance(OptionCode, (list, set, tuple)):
            OptionCodestr = ','.join(OptionCode)
            if 'CON_OP_' in OptionCode:
                sfGreek = OptionCode.replace('CON_OP_', 'CON_SO_')
                return self.generUrl + sfGreek
            else:
                raise ValueError('Unknown OptionCode:%s' % OptionCodestr)

    def getCodeofOptionurl(self, TargetAssetsCode, yearmonthcode, OptionType='Call', returnparameters=False):
        """
        this function is to get the api of  the  option code list
        :param TargetAssetsCode: Underlying code ,such as 510050
        :param yearmonthcode: also called YM; it is the special code for the duration or maturity
        :param OptionType:  the type of Option
        :param returnparameters: Bool to control the whether return parameterss
        :return: apiurl (option code)
        """
        # getCodeofCallOptionurl='http://hq.sinajs.cn/list=OP_UP_%d%s'%(TargetAssetsCode,yearmonthcode)#获得某月到期的看涨期权代码列表
        # getCodeofPutOptionurl='http://hq.sinajs.cn/list=OP_DOWN_%d%s'%(TargetAssetsCode,yearmonthcode)#获得某月到期的看跌期权代码列表
        if isinstance(yearmonthcode, str):
            if OptionType == 'Call':
                if returnparameters:
                    # http://hq.sinajs.cn/list=
                    return self.generUrl + 'OP_UP_%d%s' % (TargetAssetsCode, yearmonthcode), 'OP_UP_%d%s' % (
                        TargetAssetsCode, yearmonthcode)
                else:
                    # http://hq.sinajs.cn/list=
                    return self.generUrl + 'OP_UP_%d%s' % (TargetAssetsCode, yearmonthcode)
            elif OptionType == 'Put':

                if returnparameters:
                    # http://hq.sinajs.cn/list=
                    return self.generUrl + 'OP_DOWN_%d%s' % (TargetAssetsCode, yearmonthcode), 'OP_DOWN_%d%s' % (
                        TargetAssetsCode, yearmonthcode)
                else:
                    # http://hq.sinajs.cn/list=
                    return self.generUrl + 'OP_DOWN_%d%s' % (TargetAssetsCode, yearmonthcode)
        elif isinstance(yearmonthcode, (set, tuple, list)):
            if OptionType == 'Call':
                if returnparameters:
                    c = ['OP_UP_%d%s' % (TargetAssetsCode, ymc)
                         for ymc in yearmonthcode]
                    # http://hq.sinajs.cn/list=
                    r = self.generUrl + ','.join(c)
                    return r, c
                else:
                    # http://hq.sinajs.cn/list=
                    return self.generUrl + ','.join(['OP_UP_%d%s' % (TargetAssetsCode, ymc) for ymc in yearmonthcode])
            elif OptionType == 'Put':
                if returnparameters:
                    c = ['OP_DOWN_%d%s' % (TargetAssetsCode, ymc)
                         for ymc in yearmonthcode]
                    # http://hq.sinajs.cn/list=
                    r = self.generUrl + +','.join(c)
                    return r, c
                else:
                    # http://hq.sinajs.cn/list=
                    return self.generUrl + +','.join(
                        ['OP_DOWN_%d%s' % (TargetAssetsCode, ymc) for ymc in yearmonthcode])

    def getRealtimeOptionQuotesurl(self, OptionCode):
        """
        this functin is to obtain the realtime option contract quote
        :param OptionCode: option code
        :return:  sina api url (realtime option)
        """
        #
        # print 'http://hq.sinajs.cn/list=%s'%(OptionCode)
        return 'http://hq.sinajs.cn/list=%s' % (OptionCode)  # 根据合约代码获得实时期权行情

    def getMonthofContracturl(self):
        """
        获得现在有哪几个月份的合约

        :return: list
        """
        return 'http://stock.finance.sina.com.cn/futures/api/openapi.php/StockOptionService.getStockName'  # 获得现在有哪几个月份的合约


class SinaAPI(object):

    def __init__(self, parentClass=GenerateSinaAPIUrl()):
        self.GenerateSinaAPIUrl = parentClass
        import requests
        # initial the requests session
        self.session = requests.session()

    def get(self, url):
        """
        requests url by requests session
        :param url: apiurl
        :return: raw_data of response
        """
        response = self.session.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise ValueError('Requests Error Code:%d' % response.status_code)

    def getMonth(self):
        """
        this method is to get avaible Month for current released (option) contracts
        :return: cateId, contractMonth
        """
        requesttext = eval(
            self.get(self.GenerateSinaAPIUrl.getMonthofContracturl()))
        cateId = requesttext['result']['data']['cateId']
        contractMonth = requesttext['result']['data']['contractMonth']
        return cateId, contractMonth

    def getCodeofOption(self, TargetAssetsCode, yearmonthcode, OptionType='Call'):
        """
        obtain the option code
        :param TargetAssetsCode: Underlying Security Code
        :param yearmonthcode: YM
        :param OptionType: option type
        :return: calllist
        """

        CodeofOptionurl, para = self.GenerateSinaAPIUrl.getCodeofOptionurl(
            TargetAssetsCode, yearmonthcode, OptionType=OptionType, returnparameters=True)
        calltext = self.get(CodeofOptionurl)
        # calltext.split(';\n')[:-1]
        # header='var hq_str_%s='%(para)
        for ctext in calltext.split(';\n')[:-1]:
            if 'var hq_str_' in ctext:
                code = ctext.split('=')[-1].split(',')
                calllist = [c.strip('"') for c in code if 'CON_OP_' in c]
                return calllist
            else:
                raise ValueError('Unknown Response Text of ')

    def getGreekIndicators(self, OptionCode, RequestTimeStamp):
        # RequestTimeStamp = time.time()
        dgvt = self.get('http://hq.sinajs.cn/list=' +
                        ','.join(OptionCode).replace('CON_OP_', 'CON_SO_'))
        colname = u'期权合约简称,,,,成交量,Delta,Gamma,Theta,Vega,隐含波动率,最高价,最低价,交易代码,行权价,最新价,理论价值,Unknown Var'.split(
            ',')
        df = pd.DataFrame([item.split('=')[-1].strip('"').split(',')
                           for item in dgvt.split(';\n')[:-1]], columns=colname)
        df['RequestTimeStamp'] = RequestTimeStamp
        df = df[[u'期权合约简称', u'成交量', u'Delta',
                 u'Gamma', u'Theta', u'Vega',
                 u'隐含波动率', u'最高价', u'最低价',
                 u'交易代码', u'行权价', u'最新价',
                 u'理论价值', u'Unknown Var', u'RequestTimeStamp']]
        df.columns = ['OptionContShortName', 'TradingVolume', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV', 'High', 'Low',
                      'TrasactionCode', 'StrikePrice', 'LastestPrice', 'TheoreticalValue', 'Unknown Var',
                      'RequestTimeStamp']
        # Unknown Var = sign position：default：M if occuring info change then
        # change it to A, change again then to B Untill Z
        return df

    def getRealtimeOptionQuotes(self, OptionCode, RequestTimeStamp):
        """index=[u'买量',u'买价',u'最新价',u'卖价',u'卖量',
                   u'持仓量',u'涨幅',u'行权价',u'昨收价',u'开盘价',
                   u'涨停价',u'跌停价',u'申卖价五',u'申卖量五',u'申卖价四',
                   u'申卖量四',u'申卖价三',u'申卖量三',u'申卖价二',u'申卖量二',
                   u'申卖价一',u'申卖量一',u'申买价一',u'申买量一',u'申买价二',
                   u'申买量二',u'申买价三',u'申买量三',u'申买价四',u'申买量四',
                   u'申买价五',u'申买量五',u'行情时间',u'主力合约标识',u'状态码',
                   u'标的证券类型',u'标的股票',u'期权合约简称',u'振幅',u'最高价',
                   u'最低价',u'成交量',u'成交额'] """
        index = [u'BuyVolume', u'BuyPrice', u'LastestPrice', u'SellPrice', u'SellVol',
                 u'HoldingVolume', u'Change', u'StrikePrice', u'YesterdayClosePrice', u'OpenPrice',
                 u'HighPriceinTheory', u'LowPriceinTheory', u'SellPriceT5', u'SellVolumeT5', u'SellPriceT4',
                 u'SellVolumeT4', u'SellPriceT3', u'SellVolumeT3', u'SellPriceT2', u'SellVolumeT2',
                 u'SellPriceT1', u'SellVolumeT1', u'BuyPriceT1', u'BuyVolumeT1', u'BuyPriceT2',
                 u'BuyVolumeT2', u'BuyPriceT3', u'BuyVolumeT3', u'BuyPriceT4', u'BuyVolumeT4',
                 u'BuyPriceT5', u'BuyVolumeT5', u'QuotatesTime', u'MainContractCode', u'StatusCode',
                 u'UnderlyingSecCode', u'UnderlyingSec', u'OptionContShortName', u'Amplitude', u'High',
                 u'Low', u'TradingVolume', u'Amount']
        if isinstance(OptionCode, str):
            RealtimeOptionQuotesurl = self.GenerateSinaAPIUrl.getRealtimeOptionQuotesurl(
                OptionCode)
            # print RealtimeOptionQuotesurl
            # RequestTimeStamp = time.time()
            stext = self.get(RealtimeOptionQuotesurl)
            header = 'var hq_str_%s=' % (OptionCode)
            # print 1,stext,header
            data = stext.split(header)[-1].strip(';\n').strip('"').split(',')
            otherindex = [u'Unknown Var'] * (len(data) - len(index))
            data.append(RequestTimeStamp)

            return pd.DataFrame(data, index=index + otherindex + ['RequestTimeStamp'], columns=[OptionCode])
        elif isinstance(OptionCode, (list, set, tuple)):
            OptionCodestr = ','.join(OptionCode)
            # print OptionCodestr
            RealtimeOptionQuotesurl = self.GenerateSinaAPIUrl.getRealtimeOptionQuotesurl(
                OptionCodestr)
            # RequestTimeStamp = time.time()
            stext = self.get(RealtimeOptionQuotesurl)
            headers = ['var hq_str_%s=' % (s) for s in OptionCode]
            # data.append(RequestTimeStamp)
            c = dict()
            # print stext
            for r in stext.split(';\n')[:-1]:
                name, data = r.split('=')
                data = data.strip('"').split(',')
                # print data
                name = name.strip('var hq_str_')
                # otherindex=[u'Unknown Var']*(len(data)-len(index))
                c[name] = data
            tempdf = pd.DataFrame(c)

            otherindex = [u'Unknown Var'] * (tempdf.shape[0] - len(index))
            tempdf.index = index + otherindex
            tempdf = tempdf.T
            tempdf['RequestTimeStamp'] = RequestTimeStamp
            return tempdf


class Collector(object):

    def __init__(self, api=SinaAPI(GenerateSinaAPIUrl())):
        self.api = api
        self.watchlist = None
        self.task = []
        self.lastreqtime = 0

    def startOnce(self, TargetAssetsCode=510050):
        """
        this function is to obtain current existing option categories and general information for further operations
        :param TargetAssetsCode: Underlying Code
        :return: dict
        """
        cateId, contractMonth = self.api.getMonth()
        OptionContractMonth = sorted(
            [d.split('-')[0][-2:] + d.split('-')[1] for d in set(contractMonth)])
        todayjobs = {}
        for yearmonthcode in OptionContractMonth:
            calllist = self.api.getCodeofOption(
                TargetAssetsCode, yearmonthcode, OptionType='Call')
            putlist = self.api.getCodeofOption(
                TargetAssetsCode, yearmonthcode, OptionType='Put')
            todayjobs[yearmonthcode] = {'Call': calllist, 'Put': putlist}
        return todayjobs

    def beforeStart(self, TargetAssetsCode=510050):
        """
        this function is to prepare collect data from api high frequency
        rewrite the self.watchlist for further collection
        :param TargetAssetsCode: Underlying Code
        :return: download_hq_quote;
        """

        sssO = self.startOnce(TargetAssetsCode=TargetAssetsCode)
        self.watchlist = pd.DataFrame([(c, cp, k) for k, v in sssO.iteritems(
        ) for cp, v2 in v.iteritems() for c in v2], columns=['SinaCode', 'OptionType', 'YM'])

        returnvalue = self.watchlist['SinaCode'].values
        self.watchlist = self.watchlist.set_index('SinaCode')

        return returnvalue

    def regularGetGI(self, watchlist, RequestTimeStamp, Optimizer=True):
        """
        this method is to regularly requests real-time Option greek indicators
        :param watchlist: the list of required download_hq_quote
        :param RequestTimeStamp:  the stamp of request time
        :param Optimizer: Bool, Optimizer
        :return: dataframe (Greeks)
        """
        watchlist = set(watchlist)

        df = self.api.getGreekIndicators(watchlist, RequestTimeStamp)

        return self.dataFrameOptimizer(df, Optimizer=Optimizer)

    def regularGetRTOQ(self, watchlist, RequestTimeStamp, merge=True, Optimizer=False):
        """
        this method is to regularly requests real-time Option Quotes time
        :param watchlist: the list of required download_hq_quote
        :param RequestTimeStamp: the stamp of request time
        :param merge: Bool. decide whether merge as one dataframe or as list ;Default True
        :param Optimizer: Bool, Optimizer
        :return: dataframe (Option)
        """

        # self.task=set(watchlist)
        watchlist = set(watchlist)
        df = self.api.getRealtimeOptionQuotes(watchlist, RequestTimeStamp)
        if merge:
            return self.dataFrameOptimizer(pd.concat([df, self.watchlist], axis=1).reset_index(), Optimizer=Optimizer)
        else:
            return self.dataFrameOptimizer(df, Optimizer=Optimizer)

    def dataFrameOptimizer(self, df2, Optimizer=True, **rules):
        """
        dataFrameOptimizer
        :param df2:
        :param Optimizer:
        :param rules:
        :return:
        """
        if Optimizer:
            for v in df2.columns:
                try:
                    df2[v] = df2[v].astype(np.float)
                except ValueError as e:
                    pass
            for key, nptype in rules.iteritems():
                try:
                    if key in df2.columns:
                        df2[key] = df2[key].astype(nptype)
                except Exception as e:
                    pass

        else:
            pass
        return df2


class inMemoryDB():

    def __init__(self):
        self.conn = []
        self.memory = True

    def createSqliteDB(self, path='data/', optionname='50ETF', memory=False):
        import sqlite3
        n = datetime.datetime.now()
        if memory:
            dbname = ':memory:'
        else:
            dbname = path + \
                     'Option-%s-%s.sqlite' % (optionname, n.strftime('%Y-%m-%d'))
        self.conn = sqlite3.connect(dbname)
        return self.conn

    def read(self, sql):
        return self.conn(sql).fetchall()

    def pdread(self, sql):
        return pd.read_sql(sql, self.conn)


def autoupdate(addmonth=0):
    import datetime
    FourdigitDate = datetime.datetime.today()
    year = str(FourdigitDate.year)[-2:]
    month = '0' + str(FourdigitDate.month) if len(str(FourdigitDate.month)
                                                  ) == 1 else str(FourdigitDate.month)
    return year + month


def timestampe2str(t=time.time()):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def checkmem(var):
    # print sys.getsizeof(var)
    return sys.getsizeof(var)


class GetData(object):

    def __init__(self):
        pass
        self.connectFunGroup = DBConnect()

    def readDB(self, varlist, table, dbpath, dbtype='sqlite'):
        special = False
        if dbtype == 'sqlite':
            if varlist != '*' and isinstance(varlist, (list, set, tuple)):
                varcommand = ','.join(varlist)
            elif varlist == '*':
                varcommand = '*'
            elif isinstance(varlist, str):
                if 'from' in varlist:
                    special = True
                else:
                    varcommand = varlist
            conn = self.connectFunGroup.SqliteConnection(dbpath)
            if special:
                return pd.read_sql(varlist, conn)
            else:
                return pd.read_sql('select %s from %s' % (varcommand, table), conn)


class DBConnect(object):

    def __init__(self):
        pass

    def SqliteConnection(self, target):
        import sqlite3
        if target != 'default':
            conn = sqlite3.connect(target, timeout=10)
        else:
            conn = sqlite3.connect(":memory:")
        return conn


def stamp2str(stamp):
    if isinstance(stamp, (str, np.float, np.int)):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stamp))
    elif isinstance(stamp, (np.ndarray, pd.Series, list, set, tuple)):
        return [stamp2str(s) for s in stamp]
    elif isinstance(stamp, pd.Series):
        return [stamp2str(s) for s in stamp]


class Inprogress():

    def __init__(self, total):
        self.total = total
        self.counts = 0
        self.progress = self.progress_coroutine()
        self.progress.next()

    def __progress(self, spend, times=1, bar_length=20):
        speed = 1 / spend

        w = spend * (self.total - self.counts)

        precent = self.counts / float(time.time() + w)

        ETA = datetime.datetime.fromtimestamp(time.time() + w)
        hashes = '#' * int(precent * bar_length)
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("""\r%d%%|%s|read %d items|Speed : %.1f/s |ETA: %s """ % (
            precent * 100, hashes + spaces, self.counts, speed, ETA))

        # sys.stdout.write("\rthis spider has already read %d projects, speed: %.4f/projects" % (counts,f2-f1))

        # sys.stdout.write("\rPercent: [%s] %d%%,remaining time: %.4f mins"%(hashes + spaces,precent,w))
        sys.stdout.flush()
        # time.sleep()
        pass

    def progress_coroutine(self):
        spend = yield
        while 1:

            if spend == 'END':
                break
            else:

                self.__progress(spend)
                spend = yield
        yield


def main(cc, TargetAssetsCode, iDB, force, inmemory=True, savepath='data/'):
    # in_progress = Inprogress(4 * 60 * 60)
    optiondfcolumns = [u'BuyVolume', u'BuyPrice', u'LastestPrice', u'SellPrice', u'SellVol',
                       u'HoldingVolume', u'Change', u'StrikePrice', u'YesterdayClosePrice', u'OpenPrice',
                       u'HighPriceinTheory', u'LowPriceinTheory', u'SellPriceT5', u'SellVolumeT5', u'SellPriceT4',
                       u'SellVolumeT4', u'SellPriceT3', u'SellVolumeT3', u'SellPriceT2', u'SellVolumeT2',
                       u'SellPriceT1', u'SellVolumeT1', u'BuyPriceT1', u'BuyVolumeT1', u'BuyPriceT2',
                       u'BuyVolumeT2', u'BuyPriceT3', u'BuyVolumeT3', u'BuyPriceT4', u'BuyVolumeT4',
                       u'BuyPriceT5', u'BuyVolumeT5', u'QuotatesTime', u'MainContractCode', u'StatusCode',
                       u'UnderlyingSecCode', u'UnderlyingSec', u'OptionContShortName', u'Amplitude', u'High',
                       u'Low', u'TradingVolume', u'Amount', u'Unknown Var',
                       u'RequestTimeStamp', u'OptionType', u'YM']
    greekdfcolumns = ['OptionContShortName', 'TradingVolume', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV', 'High', 'Low',
                      'TrasactionCode', 'StrikePrice', 'LastestPrice', 'TheoreticalValue', 'Unknown Var',
                      'RequestTimeStamp']
    ss = cc.beforeStart()
    conn = iDB.createSqliteDB(memory=inmemory)
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    mstart = time.mktime(time.strptime(
        date + ' 09:30:00', '%Y-%m-%d %H:%M:%S'))
    mend = time.mktime(time.strptime(date + ' 11:30:00', '%Y-%m-%d %H:%M:%S'))
    afstart = time.mktime(time.strptime(
        date + ' 13:00:00', '%Y-%m-%d %H:%M:%S'))
    afend = time.mktime(time.strptime(date + ' 15:00:00', '%Y-%m-%d %H:%M:%S'))

    # savepath+'Option-%s-%s-%s.sqlite'%(optionname,'Trading',n.strftime('%Y-%m-%d'))
    while True:
        if force != 0 and force != False:
            f = time.time()
            RequestTimeStamp = f
            # this is to force running the main function for test purpose
            print('You have chosen run program fiercely, will run %d times remaining!' % force)
            # obtain option quotes
            realtimeoption = cc.regularGetRTOQ(ss, RequestTimeStamp, Optimizer=True)
            # obatin option greeks
            realtimegreek = cc.regularGetGI(ss, RequestTimeStamp, Optimizer=True)

            realtimeoption.to_sql('option', conn, if_exists='append')
            realtimegreek.to_sql('greek', conn, if_exists='append')
            time.sleep(random.random())
            force -= 1
            print(force)
            if force == 0:
                option = iDB.pdread('select * from option')[optiondfcolumns]
                greek = iDB.pdread('select * from greek')[greekdfcolumns]
                # dbname=path+'Option-%s-%s-%s.sqlite'%(optionname,typeofinfo,n.strftime('%Y-%m-%d'))
                optionname = '50ETF'
                n = datetime.datetime.now()
                with DBConnect().SqliteConnection(
                        savepath + 'Option-%s-%s.sqlite' % (optionname, n.strftime('%Y-%m-%d'))) as discconn:
                    option.to_sql('RealtimeOption', discconn,
                                  if_exists='append')
                    greek.to_sql('RealtimeGreek', discconn, if_exists='append')

                break
        elif mend >= time.time() >= mstart or afstart <= time.time() <= afend:
            # running during trading time
            f = time.time()
            RequestTimeStamp = f
            # start=timestampe2str()
            # realtimeoption = cc.regularGetRTOQ(ss, RequestTimeStamp, Optimizer=True)
            # realtimegreek = cc.regularGetGI(ss, RequestTimeStamp, Optimizer=True)
            realtimeoption = cc.regularGetRTOQ(ss, RequestTimeStamp, Optimizer=True)
            # greek=cc.api.getGreekIndicators(ss)
            # cc.regularGet(ss)[u'行情时间'].max(),n.strftime(format='%Y-%m-%d %H:%M:%S')
            greek = cc.regularGetGI(ss, RequestTimeStamp, Optimizer=True)
            # for i in xrange(110*2):
            # f=time.time()
            realtimeoption.to_sql('option', conn, if_exists='append')
            greek.to_sql('greek', conn, if_exists='append')
            time.sleep(1)
            if time.time() <= mend:
                # in_progress.counts = time.time() - mstart
                pass
            elif afstart <= time.time() <= afend:
                # in_progress.counts = time.time() - mstart + 2 * 60 * 60
                pass
            # in_progress.progress.send(1)
        elif mend < time.time() < afstart:
            # between 11:30 and 13:30, Noon Break
            print('Noon Break,sleep: %d seconds,Waiting......' % (afstart - time.time()))
            time.sleep(afstart - time.time())

        elif time.time() < mstart:
            # before trading time
            sleep = mstart - time.time()
            print('Before Open,sleep: %d seconds,Waiting......' % sleep)
            time.sleep(sleep)

        elif time.time() > afend:
            # after trading time
            print('After Close,Cleaning')
            option = iDB.pdread('select * from option')[optiondfcolumns]
            greek = iDB.pdread('select * from greek')[greekdfcolumns]
            optionname = '50ETF'
            n = datetime.datetime.now()
            with DBConnect().SqliteConnection(
                    savepath + 'Option-%s-%s.sqlite' % (optionname, n.strftime('%Y-%m-%d'))) as discconn:

                option.to_sql('RealtimeOption', discconn, if_exists='append')
                greek.to_sql('RealtimeGreek', discconn, if_exists='append')

            conn.close()
            break
    print('Done')


if __name__ == '__main__':
    cc = Collector()
    TargetAssetsCode = 510050

    iDB = inMemoryDB()

    force = False
    savepath = 'data/'
    inmemory = True

    main(cc, TargetAssetsCode, iDB, force, inmemory=inmemory)
