"""
General python utility functions
"""

import datetime
import gc
import json
import os
import pickle
import sqlite3
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import get_config

logger = getLogger()
config = get_config()
data_path = config['data']['path']


def read_obj(file_name, root=data_path):
    """
    load / read non-df object from .pkl, .p, .pickle files
    @param file_name:
    @param root:
    @return:
    """
    with open(root + file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_obj(bucket_desc, file_name, root=config['data']['path']):
    """
    Save non-df objects such as dicts
    @param bucket_desc:
    @param file_name:
    @return:
    """
    with open(root + file_name, 'wb') as f:
        pickle.dump(bucket_desc, f)
    return True


def read_file(file_name, root=config['data']['path'], encoding='cp1251', usecols=None, dtype=None, sep=',',
              sheet_name=0, skiprows=None, nrows=None, index_col=None, converters=None, quotechar='\"', decimal='.',
              df_name='df'):
    if file_name.endswith(".csv") | file_name.endswith(".zip") | file_name.endswith(".tsv"):
        df = pd.read_csv(root + file_name, encoding=encoding, dtype=dtype, sep=sep, decimal=decimal,
                         usecols=usecols, error_bad_lines=False, nrows=nrows, index_col=index_col, quotechar=quotechar
                         )
        # log.info('Loaded ' + file_name)
        return df
    elif file_name.endswith(".txt"):
        df = pd.read_csv(root + file_name, encoding=encoding, dtype=dtype, sep=sep,
                         usecols=usecols, error_bad_lines=False, nrows=nrows)
        # log.info('Loaded ' + file_name)
        return df
    elif file_name.endswith(".xlsx") | file_name.endswith(".xls"):
        df = pd.read_excel(root + file_name, dtype=dtype, sheet_name=sheet_name, skiprows=skiprows,
                           converters=converters)
        if usecols is not None:
            df = df[usecols]
        # log.info('Loaded ' + file_name)
        return df
    elif file_name.endswith('hd5'):
        return pd.read_hdf(root + file_name, key=df_name)
    elif file_name.endswith('.p') | file_name.endswith('.pkl'):
        return pd.read_pickle(root + file_name)
    elif file_name.endswith('.f'):
        return pd.read_feather(root + file_name, columns=usecols)
    else:
        file_name = file_name + '.f'
        return pd.read_feather(root + file_name, columns=usecols)


def save_file(df, file_name, root=config['data']['path'],
              enc='cp1251', index=False, sep=',', csv_copy=False, mode='a'):
    if file_name.endswith(".csv"):
        df.to_csv(root + file_name, encoding=enc, index=index, sep=sep)
        # log.info(file_name + ' saved')
        return True
    elif file_name.endswith(".xlsx") | file_name.endswith(".xls"):
        df.to_excel(root + file_name, encoding=enc, index=index)
        # log.info(file_name + ' saved')
        return True
    elif file_name.endswith(".hd5"):
        df.to_hdf(root + file_name, key='df', complevel=6, mode=mode)
        return True
    elif file_name.endswith(".pkl"):
        df.to_pickle(root + file_name)
        return True
    elif file_name.endswith(".f"):
        df.reset_index(drop=True).to_feather(root + file_name + '.f')
        return True
    elif file_name.endswith(".txt"):
        with open(root + file_name, "w") as text_file:
            text_file.write(df)
    else:
        # log.info(file_name + ' format is not supported, trying CSV')
        df.reset_index(drop=True).to_feather(root + file_name + '.f')
        if csv_copy:
            df.to_csv(root + file_name + '.csv', encoding=enc, index=index, sep=sep)
        # log.info(file_name + '.csv' + ' saved')
        return True


def read_chunks(file_name, chunksize, root=config['data']['path'], encoding='cp1251',
                usecols=None, dtype=None, sep=',',
                nrows=None, index_col=None, quotechar='\"', decimal='.'):
    if file_name.endswith(".csv") | file_name.endswith(".zip") | file_name.endswith(".tsv"):
        df = pd.read_csv(root + file_name, encoding=encoding, dtype=dtype, sep=sep, decimal=decimal,
                         usecols=usecols, error_bad_lines=False, nrows=nrows, index_col=index_col, quotechar=quotechar,
                         chunksize=chunksize
                         )
        # log.info('Loaded ' + file_name)
        return df
    elif file_name.endswith(".txt"):
        df = pd.read_csv(root + file_name, encoding=encoding, dtype=dtype, sep=sep,
                         usecols=usecols, error_bad_lines=False, nrows=nrows, chunksize=chunksize)
        return df


def append_file(df, df_name, file_name, df_desc='', root=config['data']['path']):
    # If file exists then append
    if os.path.exists(root + file_name + ''):
        with pd.HDFStore(root + file_name + '', complevel=6) as store:
            # Append data df
            store.put(value=df, key=df_name)
            # Update content df
            content = store['df']
            content.loc[df_name] = [df_name, df_desc]
            store.put(value=content, key='df')
    # Else create a new one
    else:
        with pd.HDFStore(root + file_name + '', complevel=6) as store:
            # Append data df
            store.put(value=df, key=df_name)
            # Create content df
            content = pd.DataFrame(columns=['df_name', 'df_desc'])
            content.loc[df_name] = [df_name, df_desc]
            store.put(value=content, key='df')


def delete_file(file_name, root=config['data']['path']):
    if os.path.exists(root + file_name):
        os.remove(root + file_name)
    elif os.path.exists(root + file_name + ''):
        os.remove(root + file_name + '')
    else:
        print('File does not exists')


def check_file_existence(file_name, root=data_path):
    """
    check that file exists
    :param file_name: path to file for check
    :param root: root data folder
    :return: bool whether file exists or not
    """
    return os.path.exists(root + file_name)


def read_files_in_folder(path_in_data, root=config['data']['path'], startswith='', endswith='',
                         encoding='cp1251', dtype=None, sep=',', file_name_column=False):
    df_list = []
    for f in os.listdir(root + path_in_data):
        print(f)
        if f.startswith(startswith) & f.endswith(endswith):
            if f.endswith(".csv") | f.endswith(".zip") | f.endswith(".txt") | f.endswith(".xlsx") | f.endswith(""):
                df = read_file(f, root=root + path_in_data + '/', encoding=encoding, dtype=dtype, sep=sep)
                if file_name_column:
                    df['file_name'] = f
                df = df.drop_duplicates()
                df_list.append(df)

    return pd.concat(df_list, sort=False, ignore_index=True)


def inn_to_string(float_inn, bad_inn_null=False):
    """
    :param float_inn: converts INN from float to string (includes necessary zeros)
    :param bad_inn_null: replace bad inns with nulls
    :return: INN in string format
    """

    # convert to string type
    float_inn = pd.to_numeric(float_inn, errors='coerce') / 1
    string_inn = float_inn.astype(str).str[:-2]

    # add '0' to get 10 letters
    for i in range(1, 10):
        string_inn = pd.Series(np.where(string_inn.str.len() == i, '0' * (10 - i) + string_inn.astype(str), string_inn))

    # add '0' to get 12 letters
    string_inn = pd.Series(np.where(string_inn.str.len() == 11, '0' + string_inn.astype(str), string_inn))

    # restore original index
    string_inn.index = float_inn.index

    if bad_inn_null:
        string_inn = np.where(string_inn.str[0:2] == '00', np.nan, string_inn)

    return string_inn


def cleanup_dataframe_list(dflist, garbageCollect=1):
    """
    Cleans up a list of dateframes

    Optional parameter to force run the garbage collector (default 1)
        Runs at the end of the list
    """

    for df in dflist:
        cleanup_dataframe(df, 0)

    if garbageCollect == 1:
        gc.collect()


def cleanup_dataframe(df, garbageCollect=1):
    """
    Cleans up a dataframe
    Optional parameter to force run the garbage collector (default 1)
    """

    try:
        if isinstance(df, pd.DataFrame):
            df.drop(df.columns.values, axis=1, inplace=True)
            df.drop(df.index.values, axis=0, inplace=True)
        elif isinstance(df, pd.Series):
            df.drop(df.index.values, axis=0, inplace=True)
    except:
        ...
        # log.error("Could not delete object")

    if garbageCollect == 1:
        gc.collect()


# def gini(flags, scores, weights=[1]):
#     """
#     :param flags: series with true values 0/1
#     :param scores: series with calculated score
#     :param weights: optional series with weights
#     :return: gini coefficient
#     """
#     selection = ~scores.isnull()
#     try:
#         if len(weights) > 1:
#             return 2 * metrics.roc_auc_score(flags[selection], scores[selection], sample_weight=weights[selection]) - 1
#         else:
#             return 2 * metrics.roc_auc_score(flags[selection], scores[selection]) - 1
#     except:
#         return None


# def gini_sd(y_true, y_pred, n_bootstrap=1000, lower_percentile = 0.05, upper_percentile = 0.95):
#     """
#     Get Gini and 10% CL
#     :param y_true:
#     :param y_pred:
#     :return:
#     """
#     gini_results = pd.DataFrame({'gini': '', 'lower': '', 'upper': ''}, index=[0])
#
#     selection = ~(y_true.isnull() | y_pred.isnull())
#     y_true = y_true[selection].reset_index(drop=True)
#     y_pred = y_pred[selection].reset_index(drop=True)
#
#     # Gini and its CI computation
#     rng_seed = 42  # control reproducibility
#     k = 0
#
#     gini_results.loc[k, 'gini'] = gini(y_true, y_pred)
#
#     bootstrapped_scores = []
#     rng = np.random.RandomState(rng_seed)
#
#     a = pd.DataFrame()
#     a['true'] = y_true
#     a['pred'] = y_pred
#
#     if ~(y_true.empty | y_pred.empty | ((y_true == 1).sum() == 0) | ((y_true == 0).sum() == 0)):
#         for i in range(n_bootstrap):
#             # bootstrap by sampling with replacement on the prediction indices
#             y = a.sample(replace=True, random_state=rng, frac=1).copy()
#             if len(np.unique(y['true'])) < 2:
#                 # We need at least one positive and one negative sample for ROC AUC
#                 # to be defined: reject the sample
#                 continue
#
#             score = gini(y['true'], y['pred'])
#             if score:
#                 bootstrapped_scores.append(score)
#             # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
#
#     if len(bootstrapped_scores) == 0:
#         gini_results.loc[k, 'lower'] = None
#         gini_results.loc[k, 'upper'] = None
#
#     else:
#         sorted_scores = np.array(bootstrapped_scores)
#         sorted_scores.sort()
#
#         confidence_lower = sorted_scores[int(lower_percentile * len(sorted_scores))]
#         confidence_upper = sorted_scores[int(upper_percentile * len(sorted_scores))]
#         gini_results.loc[k, 'lower'] = confidence_lower
#         gini_results.loc[k, 'upper'] = confidence_upper
#         gini_av = sorted_scores.mean()
#         gini_median = sorted_scores[int(0.5 * len(sorted_scores))]
#         gini_results.loc[k, 'median'] = gini_median
#         gini_results.loc[k, 'gini'] = gini_av
#
#     return gini_results
#
# def gini_avail_matrix(y_true, x_cols, seg_cols=None, n_bootstrap=1000):
#     """
#     :param y_true: series with true values 0/1
#     :param x_cols: series or df with explanatory variables
#     :param seg_cols: series or df with categorical variables for segmentation
#     :param n_bootstrap: number of samples for gini CI estimation
#     :return: df with Gini and availability for each variable and segment
#     """
#
#     # List of explanatory variables
#     if isinstance(x_cols, pd.Series):
#         x_cols = pd.DataFrame(x_cols)
#         x_cols.columns = ['factor']
#         x_cols_list = x_cols.columns
#     elif isinstance(x_cols, pd.DataFrame):
#         x_cols_list = x_cols.columns.to_list()
#     else:
#         raise TypeError('x_cols should be a DataFrame or Series')
#
#
#     # List of segments types
#     if seg_cols is None:
#         seg_cols_list = ['All']
#     elif isinstance(seg_cols, pd.DataFrame):
#         seg_cols_list = seg_cols.columns.to_list() + ['All']
#     elif isinstance(seg_cols, pd.Series):
#         seg_cols = pd.DataFrame(seg_cols)
#         seg_cols.columns = ['segment']
#         seg_cols_list = seg_cols.columns
#
#     # Number of possible segments
#     seg_number = 0
#     for seg in seg_cols_list:
#         seg_number += 1 if seg == 'All' else len(seg_cols[seg].unique())
#
#     # Skeleton
#     gini_mat = pd.DataFrame(np.zeros((seg_number, 6 * len(x_cols_list))),
#                             columns=[s + '_gini' for s in x_cols_list] + [s + '_gini_low' for s in x_cols_list] +
#                                     [s + '_gini_up' for s in x_cols_list] + [s + '_filled' for s in x_cols_list] +
#                                     [s + '_num_obs' for s in x_cols_list] + [s + '_num_defaults' for s in x_cols_list])
#     seg_index = 0
#     x_index = 0
#     n_obs = len(y_true.index)
#
#     seg_index_name = []
#     # Calculating Gini for each segment type
#     for seg_col in seg_cols_list:
#
#         if seg_col == 'All':  # each segment
#             for x_col in x_cols_list:  # each variable
#                 gini_res = gini_sd(y_true, x_cols[x_col], n_bootstrap=n_bootstrap)
#                 gini_mat.iloc[seg_index, x_index] = gini_res['gini'][0]
#                 gini_mat.iloc[seg_index, x_index + len(x_cols_list)] = gini_res['lower'][0]
#                 gini_mat.iloc[seg_index, x_index + 2 * len(x_cols_list)] = gini_res['upper'][0]
#                 gini_mat.iloc[seg_index, x_index + 3 * len(x_cols_list)] = (~x_cols[x_col].isnull()).sum() / n_obs
#                 gini_mat.iloc[seg_index, x_index + 4 * len(x_cols_list)] = (~x_cols[x_col].isnull()).sum()
#                 gini_mat.iloc[seg_index, x_index + 5 * len(x_cols_list)] = (y_true * (~x_cols[x_col].isnull())).sum()
#                 x_index += 1
#             x_index = 0
#             seg_index += 1
#             seg_index_name += [seg_col]
#         else:
#             for seg in seg_cols[seg_col].unique():  # each segment
#                 selection = seg_cols[seg_col] == seg
#                 for x_col in x_cols_list:  # each variable
#                     gini_res = gini_sd(y_true[selection], x_cols.loc[selection, x_col], n_bootstrap=n_bootstrap)
#                     gini_mat.iloc[seg_index, x_index] = gini_res['gini'][0]
#                     gini_mat.iloc[seg_index, x_index + len(x_cols_list)] = gini_res['lower'][0]
#                     gini_mat.iloc[seg_index, x_index + 2 * len(x_cols_list)] = gini_res['upper'][0]
#                     gini_mat.iloc[seg_index, x_index + 3 * len(x_cols_list)] = \
#                         (~x_cols.loc[selection, x_col].isnull()).sum() / selection.sum()
#                     gini_mat.iloc[seg_index, x_index + 4 * len(x_cols_list)] = \
#                         (~x_cols.loc[selection, x_col].isnull()).sum()
#                     gini_mat.iloc[seg_index, x_index + 5 * len(x_cols_list)] = \
#                         (y_true[selection] * (~x_cols.loc[selection, x_col].isnull())).sum()
#                     x_index += 1
#                 x_index = 0
#                 seg_index += 1
#                 seg_index_name += [seg_col + ": " + str(seg)]
#
#     gini_mat.rename(dict(zip(range(seg_index), seg_index_name)), axis=0, inplace=True)
#
#     return gini_mat
#

def merge_between_dates(a, b, a_start_col, a_end_col, b_date, other_key_cols={}, left_df='a'):
    """
    :param a: df with period start and end dates
    :param b: df with event date
    :param a_start_col: name of period start col
    :param a_end_col: name of period end col
    :param b_date: name of event date col
    :param other_key_cols: dict with col names for additional criteria {a_col: b_col}
    :param left_df: which df if on the left side of left join
    :return:
    """
    # Make the db in memory
    conn = sqlite3.connect(':memory:')
    a['a_key'] = a.index
    b['b_key'] = b.index
    # Write the tables
    a.to_sql('a', conn, index=False)
    b.to_sql('b', conn, index=False)

    qry = '''
        SELECT
             a.a_key ,
             b.b_key '''
    qry += '''
        FROM ''' + \
           left_df + \
           ' LEFT JOIN ' + \
           ('b ' if left_df == 'a' else 'a ') + \
           'ON a.' + a_start_col + '   <= b.' + b_date + '\n' + \
           'AND a.' + a_end_col + '   >= b.' + b_date + '\n'
    for a_field, b_field in other_key_cols.items():
        qry += 'AND a.' + a_field + ' = ' + 'b.' + b_field + '\n'

    merged_keys = pd.read_sql_query(qry, conn)
    merged_keys = merged_keys[['a_key', 'b_key']].fillna(-1)

    a = a.merge(merged_keys, on='a_key', how='left', validate='1:1')
    a['b_key'].fillna(-1, inplace=True)
    b = b.merge(merged_keys, on='b_key', how='left', validate='1:1')
    b['a_key'].fillna(-1, inplace=True)
    conn.close()
    return a, b


def bins_from_col(series, sep=', ', return_as='IntervalIndex'):
    """
    Extract bins as IntervalIndex from a series with strings in format "(0.2, 0.3]"
    :param series: series with bins in strings
    :param sep: separator between boundaries inside string
    :param return_as: return result as IntervalIndex or ndarray
    :return: Unique bins as IntervalIndex
    """

    # Remove 'nan' buckets as they are applied automatically during pd.cut
    series = series[series != 'nan']

    # Parse strings with bins
    bins = [elem[1:-1].split(sep) for elem in series.unique()]

    if return_as == 'IntervalIndex':
        # Turn bins into IntervalIndex
        intervals = [pd.Interval(left=float(elem[0]), right=float(elem[1])) for elem in bins]
        index = pd.Index(intervals)
    elif return_as == 'ndarray':
        # Turn bins into ndarray
        index = np.array(list(set([float(element) for sublist in bins for element in sublist])))
    else:
        raise

    return index


def write_dfs_to_excel(dfs, path, root=data_path):
    """
    Writes dict with multiple data frames into file
    :param dfs: dict with data frames
    :param path: excel file path
    :param root: root folder with data
    :return: True if succeeded
    """

    with pd.ExcelWriter(root + path, engine='xlsxwriter') as writer:
        for df_name in dfs:
            dfs[df_name].to_excel(writer, sheet_name=df_name[:31])

    return True


def read_dfs_from_excel(path, root=data_path):
    """
    Create dict with multiple data frames from Excel file
    :param path: excel file path
    :param root: root folder with data
    :param xlwings: whether xlwings should be used to read the file
    :return: dict with dfs
    """

    with pd.ExcelFile(root + path) as reader:
        sheet_names = reader.sheet_names
        dfs_dict = {sheet: pd.read_excel(reader, sheet, index_col=0,
                                         na_values="''", keep_default_na=False) for sheet in sheet_names}

    return dfs_dict


def log(*args):
    print(datetime.datetime.now().strftime('%H:%M:%S.%f')[:-4], *args)
    return True


def select_data(db_name, conn):
    """
    select all data from database
    :param db_name:
    :param db_path:
    :return:
    """
    db_df = pd.read_sql_query(f"SELECT * FROM {db_name}", conn)
    return db_df


def create_sql_table(table_name: str, table_cols: tuple, db):
    """

    :param table_name: name of the table in sql database
    :param table_cols: tuple of column names
    :param db: data base connection
    :return:
    """
    try:
        query = f'''CREATE TABLE {table_name} {table_cols}'''
        db.execute(query)
        db.commit()
        return True
    except Exception as msg:
        logger.info(f'Exception {msg}')
        return False


def create_sql_tables(db_structure: dict, db, check_existing=True):
    """

    :param table_names:
    :param table_cols:
    :param db:
    :param check_created:
    :return:
    """
    for table_name in db_structure:
        if check_existing:
            try:
                create_sql_table(table_name, db_structure[table_name], db=db)
                print(f'Table {table_name} is created')
            except:
                print(f'Table {table_name} already exists')
        else:
            create_sql_table(table_name, db_structure[table_name], db=db)
            print(f'Table {table_name} is created')

    return True


def select_last_data_by_id(table,
                           db,
                           ids=None,
                           id_col='username',
                           time_col='timestamp'):
    """
    select last unique records (by id) from a table
    :param table: table name in sql
    :param ids: list of ids to filter by. If none then all uniques will be selected
    :param id_col: id col name
    :param db: database connection
    :return:
    """

    # SM's version
    # query = f''' SELECT * FROM {table}  full_table
    #             INNER join (select {id_col}, max({time_col}) as last_{time_col} from  {table} group by {id_col}) deduplicated
    #              on full_table.{id_col} = deduplicated.{id_col} and full_table.{time_col}=deduplicated.{time_col}
    #             WHERE deduplicated.{id_col} IN {tuple(ids)}'''

    if ids is None:
        filter_id_script = ''
    # elif len(ids)==1:
    #     filter_id_script = f'WHERE deduplicated.{id_col} IN {tuple([ids[0],ids[0]])}'
    else:
        # filter_id_script = f'WHERE deduplicated.{id_col} IN {tuple(ids)}'
        df = pd.DataFrame({f'{id_col}': ids})
        df['timestamp'] = pd.to_datetime('now')
        df.to_sql('tmp_table_to_check', db, if_exists='replace', index=False)
        filter_id_script = f'inner join tmp_table_to_check t on t.{id_col} = deduplicated.{id_col}'


    query = f''' SELECT full_table.* FROM {table}  full_table
                INNER join (select {id_col}, max({time_col}) as last_{time_col} from  {table} group by {id_col}) deduplicated
                 on full_table.{id_col} = deduplicated.{id_col} and full_table.{time_col}=deduplicated.last_{time_col}
                {filter_id_script}'''

    # query = f'''
    #             SELECT  *,
    #                     MAX({time_col}) as last_{time_col}
    #             FROM {table}
    #             {filter_id_script}
    #             GROUP BY {id_col} '''

    df = pd.read_sql_query(query, db, parse_dates=True)

    if ids is not None:
        db.execute('drop table tmp_table_to_check')
    return df


def latest_file(path: str, pattern: str = "*"):
    path = Path(path)
    files = path.glob(pattern)
    return max(files, key=lambda x: x.stat().st_ctime)


def read_li_json(path):
    """
    Use li data offline from saved json
    :param path:
    :return:
    """
    with open(f'{path}', 'r') as f:
        profiles_dict = json.load(f)

    return profiles_dict