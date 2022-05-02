import logging
import datetime
import subprocess
import inspect
import os
from configparser import ConfigParser


#TODO
def initialize_logger(file_path=None, reset_log=False, root='./', dir_name='logs'):
    """
    Initializes logger and writes launch information into
    :param file_path: Optional, default is None, Path to logger file
    :param reset_log: Optional, default is False, If True existing log is erased - opened with filemode 'w'
    :param root: Optional, default is //moscow/itfs/ow_modeling_drive/SME_data/, root path
    :param dir_name: Optional, defualt is 'ow_sales_platform', name of folder containing cloned git repository
    :return: logger instance
    """
    # Get path to save_file
    if file_path is None:
        frame = inspect.stack()[1]
        calling_module = frame[0].f_code.co_filename
        calling_module = os.path.realpath(calling_module)
        calling_module = str(calling_module)
        start_pos = calling_module.find(dir_name)
        file_path = calling_module[start_pos:]
        file_path = file_path.replace(dir_name, 'logs')
        file_path = file_path.replace('.py', '.txt')

    # Initiate filemode
    if reset_log:
        mode = 'w'
    else:
        mode = 'a'

    # Remove old handlers if exist
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    # Initiate new handlers
    if '<input>' in file_path:
        handlers = [logging.StreamHandler()]
    if '<input>' not in file_path:
        full_path = root + fr'/{file_path}'
        # Create logging directory
        try:
            full_path = full_path.replace('/','\\')
            dir_path = full_path[:full_path.rfind('\\')]
            os.makedirs(dir_path)
        except OSError:
            pass
        handlers = [logging.FileHandler(full_path, mode=mode), logging.StreamHandler()]

    # Specify logging config
    logging.basicConfig(
        format='%(asctime)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers
    )

    # Write launch params
    # Writing initialization time
    logger.info(f'##################################################')
    logger.info(f'Logger initialized')

    # Writing user signature
    user_login = os.getlogin()
    logger.info(f'Launch signature: {user_login}')

    # Writing actual git commit at the time of script launch
    last_commit = subprocess.check_output(['git', 'describe', '--always']).strip()
    logger.info(f'Previous git commit: {last_commit}')
    logger.info(f'##################################################')

    return logger
