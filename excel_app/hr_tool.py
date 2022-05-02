import os
import sys

import pandas as pd

# sys.path.append(f'C:\\Users\\Salim\\Dropbox\\Business opportunities\\HR_tool\\hr_tool_code\\')
path_list = os.getcwd().split('\\')
dir_level = len(path_list) - path_list.index('digital_hr') - 1
os.chdir(f"./{''.join(['../'] * dir_level)}")
sys.path.append(f"./{''.join(['../'] * dir_level)}")
import xlwings as xw
from excel_app.xls_utils import read_query, get_info_multiple_queries, get_ref_cells, get_sheets, save_project_dict, \
    read_project_dict
from utils.search_tools.general_search import query_google
from utils.search_tools.search_processing import li_enrich_gsearch_results, li_links_to_usernames
from utils.logging_tools import initialize_logger
from utils.config import get_config

config = get_config()
logger = initialize_logger(root=config['data']['path'], dir_name='digital_hr')

"""
Each method described below corresponds to a button
"""


# todo 6: add key words , hashtags to cand table (partially fixed)
# todo 7: add functionality to add candidate data based on linkedin id
# todo 8: make append mode functionality for candidate entries
# todo 9: make dropdown menu for existing projects. So that you can select existing or add a new one


# cur_dir = os.path.dirname(os.path.realpath(__file__))

# TODO: tmp turn-off linkedin capabilities


def search_candidates(from_excel=False):
    sheet_dict = get_sheets(from_excel, xls_name='excel_app/hr_tool.xlsm')

    # cleanup
    ref_cells = get_ref_cells(sheet_dict)
    # cand_sht.range(ref_cells['table_address']).expand().clear_contents()
    sheet_dict['exp_sht'].range('a1').expand().clear_contents()
    sheet_dict['edu_sht'].range('a1').expand().clear_contents()
    sheet_dict['skl_sht'].range('a1').expand().clear_contents()
    sheet_dict['cand_row_sht'].range('a1').expand().clear_contents()

    queries = read_query(sheet_dict['cand_sht'], ref_cells['query_params'])

    df = get_info_multiple_queries(queries, fun=query_google)
    df = li_enrich_gsearch_results(df)
    # todo: apply... one more time...

    # Add LI data
    if ref_cells['use_li_data']:
        usernames_to_parse = list(ppli.cache_checker(df['username']))

        max_requests = int(ref_cells['max_requests'])
        if max_requests > 0:
            usernames_to_parse = usernames_to_parse[:max_requests]

        # parse linked in and record to database
        ppli.li_parse_usernames(usernames_to_parse)

        db_names = ['Candidates', 'Experience', 'Education', 'Skills']
        cand_df, exp_df, edu_df, skl_df = ppli.select_data_by_users(db_names, df['username'])
        # cand_df, exp_df, edu_df = pli.enrich_cand_df(cand_df, exp_df, edu_df)  # todo: replace with new data processer (step 6)
        # todo: add link to the final data frame
        # todo: check  pli.enrich_cand_df function
        # cell_row = int(ref_cells['table_address'][1:])
        # for i, res in enumerate(json_res):
        #     print(json_res[res])
        #     cand_sht.range(f'b{i + cell_row}').value = str(json_res[res])
        sheet_dict['cand_row_sht'].range('a1').value = cand_df
        sheet_dict['exp_sht'].range('a1').value = exp_df
        sheet_dict['edu_sht'].range('a1').value = edu_df
        sheet_dict['skl_sht'].range('a1').value = skl_df
    else:

        first_col = df.pop('link')
        df.insert(0, 'link', first_col)
        sheet_dict['cand_row_sht'].range('a1').value = df


def record_project(from_excel, mode):
    sheets_dict = get_sheets(from_excel, xls_name='excel_app/hr_tool.xlsm')
    ref_cells = get_ref_cells(sheets_dict)
    project_desc = dict()
    # read info from sheets
    project_desc['project_params'] = sheets_dict['cand_sht'].range(ref_cells['project_params']).expand().options(
        pd.DataFrame).value
    project_desc['query_params'] = sheets_dict['cand_sht'].range(ref_cells['query_params']).expand().options(
        pd.DataFrame).value
    project_desc['li_candidates'] = sheets_dict['cand_row_sht'].range('a1').expand().options(pd.DataFrame).value
    table_row = ref_cells['table_address'][1:]
    nrows = len(project_desc['li_candidates'])

    # read recruiter data
    recruiter_df = sheets_dict['cand_sht'].range(f'x{table_row}:az{int(table_row) + nrows}').options(
        pd.DataFrame, index=False).value
    recruiter_df = recruiter_df.dropna(axis=1, how='all')
    recruiter_df['username'] = sheets_dict['cand_sht'].range(f'b{table_row}:b{int(table_row) + nrows}').options(
        pd.DataFrame, index=False).value
    recruiter_df['username'] = li_links_to_usernames(recruiter_df['username'])
    username_series = recruiter_df.pop('username')
    recruiter_df.insert(0, 'username', username_series)
    # todo: add project technical params to excel sheet

    project_desc['recruiter_data'] = recruiter_df
    project_desc['project_id'] = project_desc['project_params'].loc['project_id', 'value']
    if mode == 'create':
        project_desc['start_date'] = pd.to_datetime('now')
        project_desc['project_params'].loc['start_date', 'value'] = project_desc['start_date']
        project_desc['project_params'].loc['update_date', 'value'] = project_desc['start_date']
    elif mode == 'update':
        tmp_project_desc = read_project_dict(project_desc['project_id'])
        project_desc['start_date'] = tmp_project_desc['start_date']
        project_desc['update_date'] = pd.to_datetime('now')
        project_desc['project_params'].loc['start_date', 'value'] = project_desc['start_date']
        project_desc['project_params'].loc['update_date', 'value'] = project_desc['update_date']
    project_desc['status'] = 'active'

    # todo: add data checker for df project params, avoid ruining excel format
    sheets_dict['cand_sht'].range(ref_cells['project_params']).value = project_desc['project_params']

    save_project_dict(project_desc, mode=mode)

    return True


def load_project(from_excel):
    sheets_dict = get_sheets(from_excel, xls_name='excel_app/hr_tool.xlsm')
    ref_cells = get_ref_cells(sheets_dict)

    df = sheets_dict['cand_sht'].range(ref_cells['project_params']).expand().options(
        pd.DataFrame).value
    project_id = df.loc['project_id', 'value']

    project_desc = read_project_dict(project_id)

    # write project data to excel
    # todo: add data checker to avoid runing excel format, e.g. from old .p files
    sheets_dict['cand_sht'].range(ref_cells['project_params']).value = project_desc['project_params']
    sheets_dict['cand_sht'].range(ref_cells['query_params']).value = project_desc['query_params']
    sheets_dict['cand_row_sht'].range('a1').value = project_desc['li_candidates']

    recruiter_df = project_desc['li_candidates'][['username']].merge(project_desc['recruiter_data'],
                                                                     how='left', on='username', validate='1:1')
    recruiter_df.pop('username')  # remove username column to avoid duplication
    table_row = ref_cells['table_address'][1:]
    sheets_dict['cand_sht'].range(f'x{table_row}').expand().clear_contents()
    sheets_dict['cand_sht'].range(f'x{table_row}'). \
        options(pd.DataFrame, index=False).value = recruiter_df
    return True


def clear_project_contents(from_excel=True):
    logger.info('Delete project info')
    sheets_dict = get_sheets(from_excel, xls_name='excel_app/hr_tool.xlsm')
    ref_cells = get_ref_cells(sheets_dict)


    if sheets_dict['cand_row_sht'].range('a1').expand().value is None:
        nrows = 0
    else:
        nrows = max(len(sheets_dict['cand_row_sht'].range('a1').expand().value), 0)

    sheets_dict['cand_sht'].range('c7:c13').clear_contents()
    sheets_dict['cand_sht'].range('c15:h25').clear_contents()
    sheets_dict['exp_sht'].range('a1').expand().clear_contents()
    sheets_dict['edu_sht'].range('a1').expand().clear_contents()
    sheets_dict['skl_sht'].range('a1').expand().clear_contents()
    sheets_dict['cand_row_sht'].range('a1').expand().clear_contents()
    table_row = ref_cells['table_address'][1:]
    sheets_dict['cand_sht'].range(f'x{table_row}:z{int(table_row) + nrows}').clear_contents()

    return True


if __name__ == '__main__':
    xw.Book(r'excel_app/hr_tool.xlsm').set_mock_caller()
    search_candidates(from_excel=False)
    # record_project(from_excel=False, mode='create')
    # clear_project_contents(from_excel=True)
    # load_project(from_excel=False)
    logger.info('\n\n')
