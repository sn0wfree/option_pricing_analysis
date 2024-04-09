
import pandas as pd
import datetime
from option_pricing_analysis.analysis.option_analysis_monitor import WindHelper,ReportAnalyst

def cal_profits_returns(person_holder):
    nv_df = pd.DataFrame(index=person_holder.index)
    returns = pd.DataFrame(index=person_holder.index)
    profit = pd.DataFrame(index=person_holder.index)

    periods = {'当日': 1, '近一周': 5, '近一月': 20}

if __name__ == '__main__':
    today_str = pd.to_datetime(datetime.datetime.today()).strftime('%Y%m%d')

    wh = WindHelper()

    PR = ReportAnalyst(
        report_file_path='E:\\prt\\pf_analysis\\pf_analysis\\optionanalysis\\report_file',
        contract_2_person_rule={'MO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'HO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'IO\d{4}-[CP]-[0-9]+.CFE': 'll',
                                'IH\d{4}.CFE': 'wj',
                                'IF\d{4}.CFE': 'wj',
                                'IM\d{4}.CFE': 'll',
                                'AG\d{4}.SHF': 'gr',
                                'AU\d{4}.SHF': 'wj',
                                'CU\d{4}.SHF': 'wj',
                                'CU\d{4}[CP]\d{5}.SHF': 'wj',
                                'AL\d{4}.SHF': 'gr'}

    )

    contracts = PR.reduced_contracts()

    quote = PR.get_quote_and_info(contracts, wh, start_with='2022-09-04')

    lastdel_multi = PR.get_info_last_delivery_multi(contracts, wh)

    info_dict = PR.parse_transactions_with_quote_v2(quote, lastdel_multi,
                                                    trade_type_mark={"卖开": 1, "卖平": -1,
                                                                     "买开": 1, "买平": -1,
                                                                     "买平今": -1, }

                                                    )

    person_holder, person_ls_summary_dict, merged_summary_dict, contract_summary_dict, info_dict = PR.group_by_summary(
        info_dict, return_data=True)

    # PR.summary_person_info(person_summary_dict, merged_summary_dict, info_dict, lastdel_multi, quote, )

    print(1)
    pass