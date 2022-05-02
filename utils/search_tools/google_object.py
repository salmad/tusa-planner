from configparser import ConfigParser
import os
import pandas as pd
from googleapiclient.discovery import build  # Import the library

cur_dir = os.path.dirname(os.path.realpath(__file__))
config = ConfigParser()
config.read(f'{cur_dir}/config.ini')


def google_query(query, **kwargs):


    api_key = config['google'][f'api_key']
    cse_id = config['google'][f'cse_id']
    # api_key = 'AIzaSyAvI-8eDDqvUxoWJrF-c52areOGFnH_8O8'
    # cse_id = '6b5e5f883c7125df4'
    query_service = build("customsearch",
                          "v1", developerKey=api_key)
    query_results = query_service.cse().list(q=query,  # Query
                                             cx=cse_id,  # CSE ID
                                             **kwargs
                                             ).execute()
    return query_results['items']




def get_title(res):
    row = res['pagemap']['metatags'][0]['og:title']
    items_split = row.split('-')
    name = items_split[0]
    if len(items_split) > 2:
        position = items_split[1]
    else:
        position = '_'
    employer = items_split[-1].replace(' | LinkedIn', '')
    return {'name': name,
            'position': position,
            'employer': employer}


def main(query="site:ru.linkedin.com/in/ "
               "-inurl:people "
               "-inurl:companies msd "
               "-inurl:jobs quality assurance "
               "-inurl:location russia "
               "-inurl:Education",
         num=10
         ):

    """
    Get top n URLs from google search
    :param query:
    :return:
    """
    my_results = google_query(query, num=num)

    df = pd.DataFrame()
    for result in my_results:
        # my_results_list.append(result['link'])
        # print(f"Link: {result['link']}")
        parse_dict = get_title(result)
        df.loc[result['link'], 'name'] = parse_dict['name']
        df.loc[result['link'], 'position'] = parse_dict['position']
        df.loc[result['link'], 'employer'] = parse_dict['employer']
        df.loc[result['link'], 'comment'] = result['snippet']
        # print(f"Name: {parse_dict['name']}, Postion: {parse_dict['position']}, Employer: {parse_dict['employer']}")

        df = df.reset_index()
        df = df.rename(columns={'index': 'link'})
        df = df.set_index('link')

    return df




if __name__ == '__main__':
    query = '''site:ru.linkedin.com/in/ 
    -inurl:people 
    -inurl:companies msd 
    -inurl:jobs quality assurance 
    -inurl:location russia 
    -inurl:Education'''
    df = main()
