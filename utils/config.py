from configparser import ConfigParser




def get_config():
    config = ConfigParser()
    config.read(f'./config.ini')

    return config


