import pickle
from configparser import ConfigParser

import awswrangler as wr
import boto3
import pandas as pd
import smart_open as so

aws_config = ConfigParser()
aws_config.read(f'.aws_config.ini')

session = boto3.session.Session(aws_access_key_id=aws_config['default']['aws_access_key_id'],
                                aws_secret_access_key=aws_config['default']['aws_secret_access_key'])
s3 = session.resource('s3')


def list_dir(path, s3=s3, aws_config=aws_config):
    path_list = list(s3.Bucket(aws_config['default']['bucket']).objects.filter(Prefix=path))
    path_list = [i.key.split(f'{path}')[1].replace('/', '') for i in path_list]
    return path_list


def save_txt(obj, path, s3=s3, aws_config=aws_config):
    res = s3.Bucket(aws_config['default']['bucket']).put_object(Key=path, Body=obj)
    return res


def read_txt(path, s3=s3, aws_config=aws_config):
    res = s3.Object(key=path, bucket_name=aws_config['default']['bucket'])
    return res


def save_file(df, path, s3=s3, aws_config=aws_config, **kwargs):
    if path.endswith('.xls') or path.endswith('.xlsx'):
        wr.s3.to_excel(df=df, path=f's3://{aws_config["default"]["bucket"]}/{path}',
                       boto3_session=session)
    elif path.endswith('.csv'):
        wr.s3.to_csv(df=df, path=f's3://{aws_config["default"]["bucket"]}/{path}',
                     boto3_session=session)
    elif path.endswith('.p') or path.endswith('.pkl'):
        with so.open(f's3://{aws_config["default"]["bucket"]}/{path}', 'wb',
                     transport_params={'session': session}) as f:
            pickle.dump(df, f)
    elif path.endswith('.parquet'):
        wr.s3.to_parquet(df=df, path=f's3://{aws_config["default"]["bucket"]}/{path}',
                         boto3_session=session)
    else:
        path += '.parquet'
        wr.s3.to_parquet(df=df, path=f's3://{aws_config["default"]["bucket"]}/{path}',
                         boto3_session=session)
    return df


def read_file(path, s3=s3, aws_config=aws_config, **kwargs):
    if path.endswith('.xls') or path.endswith('.xlsx'):
        file = read_txt(path, s3=s3, aws_config=aws_config)
        df = pd.read_excel(file.get()['Body'].read(), engine='openpyxl', **kwargs)
    elif path.endswith('.csv') or path.endswith('.zip'):
        file = read_txt(path, s3=s3, aws_config=aws_config)
        df = pd.read_csv(file.get()['Body'].read(), **kwargs)
    elif path.endswith('.p') or path.endswith('.pkl'):
        file = read_txt(path, s3=s3, aws_config=aws_config)
        df = pd.read_pickle(file.get()['Body'].read(), **kwargs)
    elif path.endswith('.parquet'):
        file = read_txt(path, s3=s3, aws_config=aws_config)
        df = pd.read_parquet(file.get()['Body'].read(), **kwargs)
    else:
        path += '.parquet'
        file = read_txt(path, s3=s3, aws_config=aws_config)
        df = pd.read_parquet(file.get()['Body'].read(), **kwargs)
    return df


def save_obj(obj, path, session=session, aws_config=aws_config):
    with so.open(f's3://{aws_config["default"]["bucket"]}/{path}', 'wb', transport_params={'session': session}) as f:
        pickle.dump(obj, f)
    return True


def read_obj(path, session=session, aws_config=aws_config):
    with so.open(f's3://{aws_config["default"]["bucket"]}/{path}', 'rb', transport_params={'session': session}) as f:
        obj = pickle.load(f)
    return obj


def read_parquet(path, session=session, aws_config=aws_config):
    if not path.endswith('.parquet'):
        path += '.parquet'
    df = wr.s3.read_parquet(path=f's3://{aws_config["default"]["bucket"]}/{path}',
                            boto3_session=session)

    return df


def to_parquet(df, path, session=session, aws_config=aws_config):
    if not path.endswith('.parquet'):
        path += '.parquet'
    wr.s3.to_parquet(df=df, path=f's3://{aws_config["default"]["bucket"]}/{path}',
                     boto3_session=session)

    return True

# df = pd.DataFrame({"id": [1, 2], "value": ["foo", "boo"]})
