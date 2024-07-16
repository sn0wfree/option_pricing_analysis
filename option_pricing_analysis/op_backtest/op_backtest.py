# coding=utf-8

import pandas as pd
from ClickSQL import BaseSingleFactorTableNode

from option_pricing_analysis.analysis.option_analysis_monitor import WindHelper, Configs, Tools
from option_pricing_analysis.analysis.summary import DerivativeSummary


class MockBackTest(DerivativeSummary):
    def __init__(self, uri, transac_name, trial_table, db='mock_op_backtest',
                 contract_2_person_rule={'MO\d{4}-[CP]-[0-9]+.CFE': 'll', }):
        node_local = BaseSingleFactorTableNode(uri)
        sql = f"select * from {db}.{trial_table} where trade_number == '{transac_name}' "
        report = node_local(sql).rename(columns=MockBackTest.rename_en_cn())
        report['报单日期'] = pd.to_datetime(report['报单日期']).dt.strftime("%Y%m%d")

        super(MockBackTest, self).__init__(report_file_path=report, contract_2_person_rule=contract_2_person_rule)

    @staticmethod
    def rename_cn_en():
        report_cols = ['报单日期', '委托合约', '成交均价', '报单时间', '手数', '未成交', '买卖', '开平', '盈亏',
                       '手续费',
                       '成交号', '挂单状态', '详细状态', '备注']

        rename_cols = ["order_dt", "contract_code", "average_trade_price", "order_time", "volume", "unexecuted_volume",
                       "buy_sell_indicator", "open_close_indicator", "profit_loss", "commission", "trade_number",
                       "order_status",
                       "detailed_status", "remarks"]

        report_name_dict = dict(zip(report_cols, rename_cols))
        return report_name_dict

    @staticmethod
    def rename_en_cn():
        report_cols = ['报单日期', '委托合约', '成交均价', '报单时间', '手数', '未成交', '买卖', '开平', '盈亏',
                       '手续费',
                       '成交号', '挂单状态', '详细状态', '备注']

        rename_cols = ["order_dt", "contract_code", "average_trade_price", "order_time", "volume", "unexecuted_volume",
                       "buy_sell_indicator", "open_close_indicator", "profit_loss", "commission", "trade_number",
                       "order_status",
                       "detailed_status", "remarks"]

        report_name_dict = dict(zip(rename_cols, report_cols))
        return report_name_dict

    def auto(self, wh,
             output_strategy_xlsx_name,

             # output_config={'分品种': 'ALL', '期货多头': ['ll'], '期货对冲': ['IM'], '汇总': ['ll']},
             start_with='2022-09-04',
             trade_type_mark={"卖开": 1, "卖平": -1,
                              "买开": 1, "买平": -1,
                              "买平今": -1, }):

        # today = datetime.datetime.today()
        contracts = self.reduced_contracts()

        quote = self.get_quote_and_info(contracts, wh, start_with=start_with)

        lastdel_multi = self.get_info_last_delivery_multi(contracts, wh)

        info_dict = self.parse_transactions_with_quote_v2(quote, lastdel_multi,
                                                          trade_type_mark=trade_type_mark

                                                          )

        # person_holder, merged_summary_dict, contract_summary_dict = self.group_by_summary(info_dict, return_data=True)

        person_holder_dict, merged_summary_dict, contract_summary_dict = self.group_by_summary(info_dict,
                                                                                               return_data=True,
                                                                                               store_2_excel=False)

        # person_by_year_summary, person_cum_sub, commodity_cum_sub, holding_summary_merged_sorted = self.output_v2(
        #     info_dict, lastdel_multi, output_config, dt=today, trade_type_mark=trade_type_mark)

        store_path = output_strategy_xlsx_name

        with pd.ExcelWriter(store_path) as f:
            # person_by_year_summary.to_excel(f, 'person_by_year_summary')
            # person_cum_sub.to_excel(f, 'person_cum_sub')
            # commodity_cum_sub.to_excel(f, 'commodity_cum_sub')
            # holding_summary_merged_sorted.to_excel(f, 'holding_summary_merged_sorted')

            for name, data in sorted(person_holder_dict.copy().items(), key=lambda d: d[0][:2], reverse=True):
                data.index = data.index.strftime('%Y-%m-%d')
                data.to_excel(f, name)
                Tools.create_draw_from_opened_excel(f, data.shape[0], target_sheet=name)
                print(f"{name} output!")

            self._transactions.to_excel(f, 'report')

            ## 分人分项目分年度计算盈亏
            # for name, data in sorted(person_ls_summary_dict.copy().items(), key=lambda d: d[0][:2], reverse=True):
            #     data.index = data.index.strftime('%Y-%m-%d')
            #     data.to_excel(f, name)
            #     Tools.create_draw_from_opened_excel(f, data.shape[0], target_sheet=name)
            #     print(f"{name} output!")

            ## 分合约计算盈亏
            for name, data in sorted(merged_summary_dict.copy().items(), key=lambda d: d[0][:2], reverse=True):
                data.index = data.index.strftime('%Y-%m-%d')
                data.to_excel(f, name)
                Tools.create_draw_from_opened_excel(f, data.shape[0], target_sheet=name)
                print(f"{name} output!")

            for name, data in sorted(contract_summary_dict.copy().items(), key=lambda d: d[0][:2], reverse=True):
                data.index = data.index.strftime('%Y-%m-%d')
                data.to_excel(f, name)
                Tools.create_draw_from_opened_excel(f, data.shape[0], target_sheet=name)
                print(f"{name} output!")

            info_dict['衍生品多头持仓价值'].reindex(columns=contracts).to_excel(f, '衍生品多头持仓价值截面')
            info_dict['衍生品多头累计开仓成本'].reindex(columns=contracts).to_excel(f, '衍生品多头累计开仓成本')
            info_dict['衍生品多头累计平仓价值'].reindex(columns=contracts).to_excel(f, '衍生品多头累计平仓价值')
            info_dict['衍生品多头累计行权收益'].reindex(columns=contracts).to_excel(f, '衍生品多头累计行权收益')
            info_dict['衍生品多头累计净损益'].reindex(columns=contracts).to_excel(f, '衍生品多头累计净损益')
            info_dict['衍生品多头剩余份数'].reindex(columns=contracts).to_excel(f, '衍生品多头剩余合约数')

            info_dict['衍生品空头持仓价值'].reindex(columns=contracts).to_excel(f, '衍生品空头持仓价值截面')
            info_dict['衍生品空头累计开仓成本'].reindex(columns=contracts).to_excel(f, '衍生品空头累计开仓成本')
            info_dict['衍生品空头累计平仓价值'].reindex(columns=contracts).to_excel(f, '衍生品空头累计平仓价值')
            info_dict['衍生品空头累计行权收益'].reindex(columns=contracts).to_excel(f, '衍生品空头累计行权收益')
            info_dict['衍生品空头累计净损益'].reindex(columns=contracts).to_excel(f, '衍生品空头累计净损益')
            info_dict['衍生品空头剩余份数'].reindex(columns=contracts).to_excel(f, '衍生品空头剩余合约数')


if __name__ == '__main__':
    uri = 'clickhouse://default:Imsn0wfree@47.104.186.157:8123/system'

    transac_name = 'op_pred_arb_strategy'

    wh = WindHelper()
    config = Configs()

    PR = MockBackTest(
        uri, transac_name, 'trial_2_pred', db='mock_op_backtest',
        contract_2_person_rule={'MO\d{4}-[CP]-[0-9]+.CFE': 'll', }

    )
    PR.auto(wh, f'trail_2_{transac_name}.xlsx',

            start_with='2022-01-04', trade_type_mark={"卖开": 1, "卖平": -1,
                                                      "买开": 1, "买平": -1,
                                                      "买平今": -1, })
    print(1)

    pass

'CREATE TABLE mock_op_backtest.trial_1_put\n(\n    `order_dt` DateTime,\n    `contract_code` String,\n    `average_trade_price` Float64,\n    `order_time` String,\n    `volume` Int64,\n    `unexecuted_volume` Int64,\n    `buy_sell_indicator` String,\n    `open_close_indicator` String,\n    `profit_loss` Int64,\n    `commission` Float64,\n    `trade_number` String,\n    `order_status` String,\n    `detailed_status` String,\n    `remarks` String\n)\nENGINE = ReplacingMergeTree\nPARTITION BY substring(trade_number, 1, 5)\nORDER BY (order_dt, contract_code, order_time, trade_number,buy_sell_indicator,open_close_indicator)\nSETTINGS index_granularity = 8192'
