# coding=utf-8
import os.path

import pandas as pd
from ClickSQL import BaseSingleFactorTableNode


class CreateTableTools(object):
    transactions_status_create_sql = """CREATE TABLE if not exists monitor.transactions_status
                                        (
                                            ContractCode LowCardinality(String) COMMENT '合约代码，唯一标识交易合约',
                                            PositionDirection LowCardinality(String) COMMENT '持仓方向，如多头或空头',
                                            TradeDate Date COMMENT '持仓日期',
                                            ValueName LowCardinality(String) COMMENT '变量名',
                                            Value Nullable(Float64) COMMENT '变量值',
                                            UpdateTime DateTime MATERIALIZED  now() COMMENT '记录的更新时间'

                                        )
                                        ENGINE = ReplacingMergeTree()
                                        PARTITION BY toYYYYMM(TradeDate)
                                        ORDER BY (ContractCode, TradeDate,PositionDirection,ValueName)
                                        SETTINGS index_granularity = 8192;
    """

    contract_summary_sql = """CREATE TABLE if not exists monitor.contract_metrics
    (
        TradeDate Date COMMENT '交易日期，每条记录的日期标识',
        ContractShortName LowCardinality(String) COMMENT '合约的简称',
        StatisticalDimension LowCardinality(String) COMMENT '统计维度，用于分类或特定的数据分析',
        ClosingValueCumulative Float64 COMMENT '累计平仓价值',
        OpeningCostCumulative Float64 COMMENT '累计开仓成本',
        ResidualValue Float64 COMMENT '合约残值',
        ExerciseProfit Float64 COMMENT '行权收益',
        NetLossCumulative Float64 COMMENT '累计净损益，可用于右轴显示',
        TotalValue Float64 COMMENT '累计价值，包括残值、行权收益及平仓收益',
        HoldingYieldCumulative Float64 COMMENT '累计持仓收益率',
        CumulativeNetValue Float64 COMMENT '累计净值',
        UpdateTime DateTime MATERIALIZED now() COMMENT '记录更新的时间，每次写入时自动更新为当前时间'
    )
    ENGINE = ReplacingMergeTree()
    PARTITION BY toYYYYMM(TradeDate)
    ORDER BY (TradeDate,ContractShortName,StatisticalDimension)
    SETTINGS index_granularity = 8192;

    """

    trader_summary_sql = """CREATE TABLE if not exists monitor.trader_metrics
    (
        TradeDate Date COMMENT '交易日期，每条记录的日期标识',
        TraderShortName LowCardinality(String) COMMENT '交易员的简称',
        StatisticalDimension LowCardinality(String) COMMENT '统计维度，用于分类或特定的数据分析',
        ClosingValueCumulative Float64 COMMENT '累计平仓价值',
        OpeningCostCumulative Float64 COMMENT '累计开仓成本',
        ResidualValue Float64 COMMENT '合约残值',
        ExerciseProfit Float64 COMMENT '行权收益',
        NetLossCumulative Float64 COMMENT '累计净损益，可用于右轴显示',
        TotalValue Float64 COMMENT '累计价值，包括残值、行权收益及平仓收益',
        HoldingYieldCumulative Float64 COMMENT '累计持仓收益率',
        CumulativeNetValue Float64 COMMENT '累计净值',
        UpdateTime DateTime MATERIALIZED now() COMMENT '记录更新的时间，每次写入时自动更新为当前时间'
    )
    ENGINE = ReplacingMergeTree()
    PARTITION BY toYYYYMM(TradeDate)
    ORDER BY (TradeDate,TraderShortName,StatisticalDimension)
    SETTINGS index_granularity = 8192;

    """


class UploadDailyInfo(CreateTableTools):

    def __init__(self, summary_file_path: str, db='monitor'):
        if isinstance(summary_file_path, dict):
            pass
        elif not os.path.exists(summary_file_path):
            raise ValueError(f'{summary_file_path} is not exists!')

        self.__file_path__ = summary_file_path

        self.__db__ = db

    @property
    def _mapping_sheet_name(self):
        if isinstance(self.__file_path__, str):

            with pd.ExcelFile(self.__file_path__) as f:
                return f.sheet_names
        elif isinstance(self.__file_path__, dict):
            return self.__file_path__.keys()
        else:
            raise ValueError('got unknown summary_file_path ')

    def load_single_sheet(self, sheet_name):
        if isinstance(self.__file_path__, str):

            return pd.read_excel(self.__file_path__, sheet_name=sheet_name)
        elif isinstance(self.__file_path__, dict):
            return self.__file_path__[sheet_name]
        else:
            raise ValueError('got unknown summary_file_path ')

    def load_all_sheets(self, ):
        if isinstance(self.__file_path__, str):

            data_dict = pd.read_excel(self.__file_path__, sheet_name=None)
            return data_dict
        elif isinstance(self.__file_path__, dict):
            return self.__file_path__
        else:
            raise ValueError('got unknown summary_file_path ')

    @staticmethod
    def upload(node, df: pd.DataFrame, db: str, table: str, dt_col: str):
        df[dt_col] = pd.to_datetime(df[dt_col]).dt.strftime("%Y-%m-%d")
        node.insert_df(df, db, table, chunksize=100)

    @staticmethod
    def check_updated_dt(node, sql_dict={
        'parsed_data': 'select min(max_dt) as last_end_dt from ( select ContractCode,PositionDirection,ValueName,max(TradeDate) as max_dt from monitor.transactions_status group by (ContractCode,PositionDirection,ValueName) ) ',
        'contract_data': 'select min(max_dt) as last_end_dt from ( select ContractShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.contract_metrics group by (ContractShortName,StatisticalDimension) )',
        'person_data': 'select min(max_dt) as last_end_dt from ( select TraderShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.trader_metrics group by (TraderShortName,StatisticalDimension) )'
    }):
        for key, sql in sql_dict.items():
            yield key, node(sql)['last_end_dt'].values[0]

    def upload_parsed_data(self, node, mappings_link={'衍生品多头持仓价值截面': ('holding_value', '多头'),
                                                      '衍生品多头累计开仓成本': ('cum_cost_value', '多头'),
                                                      # cum_cost_value
                                                      '衍生品多头累计平仓价值': ('cum_sold_value', '多头'),
                                                      # cum_sold_value
                                                      '衍生品多头累计行权收益': ('cum_executed_yield', '多头'),
                                                      # cum_executed_yield
                                                      '衍生品多头累计净损益': ('cum_net_yield', '多头'),
                                                      # cum_net_yield
                                                      '衍生品多头剩余合约数': ('remaining_share', '多头'),
                                                      # remaining_share

                                                      '衍生品空头持仓价值截面': ('holding_value', '空头'),
                                                      '衍生品空头累计开仓成本': ('cum_cost_value', '空头'),
                                                      # cum_cost_value
                                                      '衍生品空头累计平仓价值': ('cum_sold_value', '空头'),
                                                      # cum_sold_value
                                                      '衍生品空头累计行权收益': ('cum_executed_yield', '空头'),
                                                      # cum_executed_yield
                                                      '衍生品空头累计净损益': ('cum_net_yield', '空头'),
                                                      # cum_net_yield
                                                      '衍生品空头剩余合约数': ('remaining_share', '空头'),
                                                      # remaining_share
                                                      }, db=None, table='transactions_status',
                           sql_dict={
                               'parsed_data': 'select min(max_dt) as last_end_dt from ( select ContractCode,PositionDirection,ValueName,max(TradeDate) as max_dt from monitor.transactions_status group by (ContractCode,PositionDirection,ValueName) ) ',
                               'contract_data': 'select min(max_dt) as last_end_dt from ( select ContractShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.contract_metrics group by (ContractShortName,StatisticalDimension) )',
                               'person_data': 'select min(max_dt) as last_end_dt from ( select TraderShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.trader_metrics group by (TraderShortName,StatisticalDimension) )'
                           }, reduce=False):
        if reduce:
            updated_dt_dict = dict(self.check_updated_dt(node, sql_dict=sql_dict))

        db = self.__db__ if db is None else db

        for name, info in mappings_link.items():
            en_name, position_direction = info
            d = self.load_single_sheet(name).set_index('date').stack(-1).reset_index()
            d.columns = ['TradeDate', 'ContractCode', 'Value']
            d['ValueName'] = name
            d['PositionDirection'] = position_direction
            if reduce:
                d = d[d['TradeDate'] > pd.to_datetime(updated_dt_dict['parsed_data'])]
                if d.empty:
                    continue
            # d['TradeDate'] = pd.to_datetime(d['TradeDate']).dt.strftime("%Y-%m-%d")

            self.upload(node, d, db, table, 'TradeDate')

            # node.insert_df(d, 'monitor', 'transactions_status', chunksize=100)
            print('uploaded ', name, db, table)
        node(f'optimize table {db}.{table} final')

    def upload_contract_data(self, node, sheet_key_word='输出', db=None, table='contract_metrics', sql_dict={
        'parsed_data': 'select min(max_dt) as last_end_dt from ( select ContractCode,PositionDirection,ValueName,max(TradeDate) as max_dt from monitor.transactions_status group by (ContractCode,PositionDirection,ValueName) ) ',
        'contract_data': 'select min(max_dt) as last_end_dt from ( select ContractShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.contract_metrics group by (ContractShortName,StatisticalDimension) )',
        'person_data': 'select min(max_dt) as last_end_dt from ( select TraderShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.trader_metrics group by (TraderShortName,StatisticalDimension) )'
    }, reduce=False):

        db = self.__db__ if db is None else db
        if reduce:
            updated_dt_dict = dict(self.check_updated_dt(node, sql_dict=sql_dict))

        task_sheet_names = filter(lambda x: sheet_key_word in x, self._mapping_sheet_name)

        for name in task_sheet_names:
            contract = name[:2]
            target_cols = ['date', '累计平仓价值', '累计开仓成本', f'{contract}残值', '行权收益', '累计净损益(右轴)',
                           f'{contract}累计价值（残值+行权收益+平仓收益）', '累计持仓收益率', '累计净值']
            en_cols = ['TradeDate', 'ClosingValueCumulative', 'OpeningCostCumulative', 'ResidualValue',
                       'ExerciseProfit', 'NetLossCumulative',
                       f'TotalValue', 'HoldingYieldCumulative', 'CumulativeNetValue']

            d2 = self.load_single_sheet(name).rename(columns=dict(zip(target_cols, en_cols)))

            d2['ContractShortName'] = contract
            d2['StatisticalDimension'] = name

            if reduce:
                d2 = d2[d2['TradeDate'] > updated_dt_dict['parsed_data']]
                if d2.empty:
                    continue
            # d['TradeDate'] = pd.to_datetime(d['TradeDate']).dt.strftime("%Y-%m-%d")

            self.upload(node, d2, db, table, 'TradeDate')

            # node.insert_df(d, 'monitor', 'transactions_status', chunksize=100)
            print('uploaded ', name, db, table)
        node(f'optimize table {db}.{table} final')

    def upload_trader_data(self, node, traders=['ll', 'gr', 'wj'], db=None, table='trader_metrics', sql_dict={
        'parsed_data': 'select min(max_dt) as last_end_dt from ( select ContractCode,PositionDirection,ValueName,max(TradeDate) as max_dt from monitor.transactions_status group by (ContractCode,PositionDirection,ValueName) ) ',
        'contract_data': 'select min(max_dt) as last_end_dt from ( select ContractShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.contract_metrics group by (ContractShortName,StatisticalDimension) )',
        'person_data': 'select min(max_dt) as last_end_dt from ( select TraderShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.trader_metrics group by (TraderShortName,StatisticalDimension) )'
    }, reduce=False):

        db = self.__db__ if db is None else db
        if reduce:
            updated_dt_dict = dict(self.check_updated_dt(node, sql_dict=sql_dict))

        task_sheet_names = filter(lambda x: x in traders, self._mapping_sheet_name)

        for name in task_sheet_names:
            contract = name[:2]
            target_cols = ['date', '累计平仓价值', '累计开仓成本', f'{contract}残值', '行权收益', '累计净损益(右轴)',
                           f'{contract}累计价值（残值+行权收益+平仓收益）', '累计持仓收益率', '累计净值']
            en_cols = ['TradeDate', 'ClosingValueCumulative', 'OpeningCostCumulative', 'ResidualValue',
                       'ExerciseProfit', 'NetLossCumulative',
                       f'TotalValue', 'HoldingYieldCumulative', 'CumulativeNetValue']

            d2 = self.load_single_sheet(name)[target_cols].rename(columns=dict(zip(target_cols, en_cols)))

            d2['TraderShortName'] = contract
            d2['StatisticalDimension'] = '多空输出'
            # d['TradeDate'] = pd.to_datetime(d['TradeDate']).dt.strftime("%Y-%m-%d")
            if reduce:
                d2 = d2[d2['TradeDate'] > updated_dt_dict['parsed_data']]
                if d2.empty:
                    continue

            self.upload(node, d2, db, table, 'TradeDate')

            # node.insert_df(d, 'monitor', 'transactions_status', chunksize=100)
            print('uploaded ', name, db, table)
        node(f'optimize table {db}.{table} final')

    def upload_all(self, node, mappings_link={'衍生品多头持仓价值截面': ('holding_value', '多头'),
                                              '衍生品多头累计开仓成本': ('cum_cost_value', '多头'),
                                              # cum_cost_value
                                              '衍生品多头累计平仓价值': ('cum_sold_value', '多头'),
                                              # cum_sold_value
                                              '衍生品多头累计行权收益': ('cum_executed_yield', '多头'),
                                              # cum_executed_yield
                                              '衍生品多头累计净损益': ('cum_net_yield', '多头'),  # cum_net_yield
                                              '衍生品多头剩余合约数': ('remaining_share', '多头'),
                                              # remaining_share

                                              '衍生品空头持仓价值截面': ('holding_value', '空头'),
                                              '衍生品空头累计开仓成本': ('cum_cost_value', '空头'),
                                              # cum_cost_value
                                              '衍生品空头累计平仓价值': ('cum_sold_value', '空头'),
                                              # cum_sold_value
                                              '衍生品空头累计行权收益': ('cum_executed_yield', '空头'),
                                              # cum_executed_yield
                                              '衍生品空头累计净损益': ('cum_net_yield', '空头'),  # cum_net_yield
                                              '衍生品空头剩余合约数': ('remaining_share', '空头'),
                                              # remaining_share
                                              }, sheet_key_word='输出', traders=['ll', 'gr', 'wj'], db=None, sql_dict={
        'parsed_data': 'select min(max_dt) as last_end_dt from ( select ContractCode,PositionDirection,ValueName,max(TradeDate) as max_dt from monitor.transactions_status group by (ContractCode,PositionDirection,ValueName) ) ',
        'contract_data': 'select min(max_dt) as last_end_dt from ( select ContractShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.contract_metrics group by (ContractShortName,StatisticalDimension) )',
        'person_data': 'select min(max_dt) as last_end_dt from ( select TraderShortName,StatisticalDimension,max(TradeDate) as max_dt from monitor.trader_metrics group by (TraderShortName,StatisticalDimension) )'
    },
                   reduce=False):
        self.upload_parsed_data(node, mappings_link=mappings_link, db=db, table='transactions_status',
                                sql_dict=sql_dict, reduce=reduce)

        self.upload_contract_data(node, sheet_key_word=sheet_key_word, db=db, table='contract_metrics',
                                  sql_dict=sql_dict, reduce=reduce)

        self.upload_trader_data(node, traders=traders, db=db, table='trader_metrics', sql_dict=sql_dict, reduce=reduce)


if __name__ == '__main__':
    from option_pricing_analysis.analysis.option_analysis_monitor import Configs
    from glob import glob

    config = Configs()

    node = BaseSingleFactorTableNode(config['src'])

    file_name = max(list(glob('日度衍生品交易收益率统计及汇总@*v3.xlsx')))

    # result_dict = pd.read_excel(file_name, sheet_name=None)
    summary = ['person_by_year_summary', 'person_cum_sub', 'commodity_cum_sub', 'holding_summary_merged_sorted', ]
    sql_dict = config['sql_dict']
    traders = config['output_config']['汇总']

    mappings_link = config['mappings_link']

    UDI = UploadDailyInfo(file_name)
    UDI.upload_all(node, mappings_link=mappings_link, sheet_key_word='输出', traders=traders, db=None, reduce=True)

    print(1)
    pass
