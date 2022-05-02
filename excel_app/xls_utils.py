import os
import pickle
import logging as logger
import pandas as pd
import xlwings as xw
from utils.config import get_config
# from utils.aws_methods import save_obj, read_obj
from utils.python_tools import save_obj, read_obj

config = get_config()

def get_li_json(df, param):
    if param['use_li_offline']:
        json_res = pli.read_li_json(param['li_json_path'])
    else:
        json_res = pli.get_li_data(df.link.values)
    return json_res


def read_query(ws, cell_start='b4'):
    """
    Read query from Excel app file
    :param ws:
    :param cell_start:
    :return:
    """

    # query = "site:ru.linkedin.com/in/ "
    # "-inurl:people "
    # "-inurl:companies sberbank "
    # "-inurl:jobs quality assurance "
    # "-inurl:location russia "
    # "-inurl:Education"

    df = ws.range(cell_start).expand().options(pd.DataFrame).value

    for col in df.columns:
        df[col] = df.index + ' ' + df[col].fillna(' ') + ' '

    query_dict = dict(df.sum())

    return query_dict

def get_info_multiple_queries(queries, fun):
    dfs = []
    for q in queries:
        df = fun(queries[q])
        if df is not None:
            df['Search name'] = q
            dfs.append(df)
    df = pd.concat(dfs).reset_index()
    return df

async def async_get_info_multiple_queries(queries, fun):
    dfs = []
    for q in queries:
        df = await fun(queries[q])
        if df is not None:
            df['Search name'] = q
            dfs.append(df)
    df = pd.concat(dfs).reset_index()
    return df


def get_ref_cells(sheet_dict):
    ref_cells = sheet_dict['tech_sht'].range('b2').expand().options(dict).value
    return ref_cells


def get_sheets(from_excel, xls_name='excel_app/hr_tool.xlsm'):
    if from_excel:
        wb = xw.Book.caller()
    else:
        wb = xw.Book(xls_name)
    # define sheets
    sheet_dict = dict()

    sheet_dict['cand_sht'] = wb.sheets['Candidate mapping']
    # sheet_dict['cand_sht'].range('i14').value = f'{os.getcwd()}'
    sheet_dict['exp_sht'] = wb.sheets['li_experiences']
    sheet_dict['edu_sht'] = wb.sheets['li_educations']
    sheet_dict['skl_sht'] = wb.sheets['li_skills']
    sheet_dict['cand_row_sht'] = wb.sheets['li_candidates']
    sheet_dict['tech_sht'] = wb.sheets['Tech']
    return sheet_dict


def save_project_dict(project_desc, mode='create'):
    logger.info('Saving project data...')
    filename = f"{project_desc['project_id']}_{project_desc['start_date'].date()}.p"
    filepath = fr'excel_app/project_data/{filename}'
    if mode == 'create':
        if os.path.exists(filepath):
            # todo: print warnings to excel log
            logger.info('WARNING: Project file has not been written. It already exists!')
            return False

    save_obj(project_desc, filepath) # save to aws or to root folder
    # with open(filepath, 'wb') as f:
    #     print('WARNING: Project file has been written')
    #     pickle.dump(project_desc, f)

    return True


def read_project_dict(project_id, root=config['data']['path']):
    logger.info('Saving project data...')
    file_list = os.listdir(fr'{root}/excel_app/project_data/') # one could replace this with aws.list_dir
    project_files = [i for i in file_list if i.split('_')[0] == f"{project_id}"]

    if len(project_files) > 1:
        logger.info('WARNING: too many files for the same project')
        return IOError
    elif len(project_files)==1:
        filename = project_files[0]
    else:
        # todo: print warnings to excel log
        logger.info('WARNING: project file does not exist!')
        return IOError
    filepath: str = fr'/excel_app/project_data/{filename}'
    # if not os.path.exists(filepath):
    # file_list = aws.list_dir(filepath)
    # if len(file_list)>0: #check if file xists
    #     # todo: print warnings to excel log
    #     print('WARNING: project file does not exist!')
    #     return IOError

    # with open(filepath, 'rb') as f:
    #     project_desc = pickle.load(f)
    project_desc = read_obj(filepath)

    return project_desc