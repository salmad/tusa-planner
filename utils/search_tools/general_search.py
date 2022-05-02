import time

import numpy as np
import pandas as pd
from scraper_api import ScraperAPIClient
from search_engine_parser.core.engines.bing import Search as BingSearch
from search_engine_parser.core.engines.google import Search as GoogleSearch


def query_google_v1(query='', page=1):
    gsearch = GoogleSearch()
    gresults = gsearch.search(query, page)

    df = pd.DataFrame(gresults.results)
    if 'links' not in df.columns:
        return None
    if df['links'].str.contains('q=').sum() > 0:
        df['link'] = df['links'].str.split('q=', expand=True)[1].str.split('&', expand=True)[0]
    else:
        df['link'] = 'ERROR in parsing'

    df['rank'] = df.index
    first_col = df.pop('link')
    df.insert(0, 'link', first_col)

    return df


def query_google_v2(query, page=1):
    parse_url = f'https://www.google.com/search?q={query}&{(page - 1) * 10}'
    client = ScraperAPIClient('2aeab1c89e10f286d5ddd1a334ae6aad')
    res = client.get(url=parse_url).text

    total_results = res['search_information']['total_results']
    org_res = res['organic_results']
    df = pd.DataFrame(org_res)

    df = df.rename(columns={'title': 'titles', 'snippet': 'descriptions', 'position': 'order'})

    if 'link' not in df.columns:
        return None
    else:
        df['link'] = 'ERROR in parsing'
    df['rank'] = df.index
    first_col = df.pop('link')
    df.insert(0, 'link', first_col)

    return df


def query_google(query='', page=1, use_v1=True):
    try:
        if use_v1:
            df = query_google_v1(query, page)
        else:
            df = query_google_v2(query, page)
    except Exception as msg:
        print(f'query_google: {msg}')
        if use_v1:
            df = query_google_v2(query, page)
        else:
            df = query_google_v1(query, page)

    return df


async def async_query_google_v2(query, page=1):
    parse_url = f'https://www.google.com/search?q={query}&{(page - 1) * 10}'
    client = ScraperAPIClient('2aeab1c89e10f286d5ddd1a334ae6aad')
    res = client.get(url=parse_url, autoparse=True).json()

    total_results = res['search_information']['total_results']  # might be useful
    org_res = res['organic_results']
    df = pd.DataFrame(org_res)

    df = df.rename(columns={'title': 'titles', 'snippet': 'descriptions', 'position': 'order'})

    df['rank'] = df.index
    first_col = df.pop('link')
    df.insert(0, 'link', first_col)

    return df


async def async_query_google_v1(query='', page=3):
    gsearch = GoogleSearch()

    dfs = []
    if_break = False
    for p in range(page):
        if if_break:
            break
        else:
            gresults = await gsearch.async_search(query, page=p+1)
            df = pd.DataFrame(gresults.results)
            if 'links' in df.columns:
                df['rank'] = df.index+(p)*10
                dfs.append(df)
            else:
                if_break = True # no more results to parse or error # todo: differentiate errors and not enough results
    if dfs ==[]:
        return None
    df = pd.concat(dfs)
    if 'links' not in df.columns:
        return None
    if df['links'].str.contains('q=').sum() > 0:
        df['link'] = df['links'].str.split('q=', expand=True)[1].str.split('&', expand=True)[0]
    else:
        df['link'] = 'ERROR in parsing'

    first_col = df.pop('link')
    df.insert(0, 'link', first_col)

    return df


async def async_query_google(query='', page=10, use_v1=True):
    try:
        if use_v1:
            df = await async_query_google_v1(query, page)
        else:
            df = await async_query_google_v2(query, page)
    except Exception as msg:
        print(f'query_google: {msg}')
        if use_v1:
            df = await async_query_google_v2(query, page)
        else:
            df = await async_query_google_v1(query, page)

    return df


def query_bing(query='', page=1):
    bsearch = BingSearch()
    bresults = bsearch.search(query, page)

    df = pd.DataFrame(bresults.results)
    if 'links' not in df.columns:
        return None
    if df['links'].str.contains('q=').sum() > 0:
        df['link'] = df['links'].str.split('q=', expand=True)[1].str.split('&', expand=True)[0]
    else:
        df['link'] = 'ERROR in parsing'

    df['rank'] = df.index
    first_col = df.pop('link')
    df.insert(0, 'link', first_col)

    return df


async def async_query_bing(query='', page=1):
    bsearch = BingSearch()
    bresults = await bsearch.search(query, page)

    df = pd.DataFrame(bresults.results)
    if 'links' not in df.columns:
        return None
    if df['links'].str.contains('q=').sum() > 0:
        df['link'] = df['links'].str.split('q=', expand=True)[1].str.split('&', expand=True)[0]
    else:
        df['link'] = 'ERROR in parsing'

    df['rank'] = df.index
    first_col = df.pop('link')
    df.insert(0, 'link', first_col)

    return df


def get_info_multiple_queries(queries, fun):
    dfs = []
    for q in queries:
        r_sleep = np.random.uniform(2, 5)
        df = fun(queries[q])
        time.sleep(r_sleep)
        if df is not None:
            df['Search name'] = q
            dfs.append(df)
    df = pd.concat(dfs).reset_index()
    return df

async def async_get_info_multiple_queries(queries, fun):
    dfs = []
    for i, q in enumerate(queries):
        print('search', i)
        if i < 15:
            r_sleep = np.random.uniform(1, 3)
            df = await fun(queries[q])
            if (len(queries) > 1) & (i < len(queries) - 1)&(i%10==0):
                time.sleep(r_sleep)
            if df is not None:
                df['Search name'] = q
                dfs.append(df)
        else:
            print('async_get_info_multiple: skipping searches - too many, reduce number of queries', i)
    df = pd.concat(dfs).reset_index()
    return df


def query_constructor(experience='',
                      company='',
                      job='',
                      license='',
                      location='',
                      education='',
                      name='',
                      skills='',
                      search_country='ru'):
    """
    Returns linkedin query for google using key words
    :param experience:
    :param company:
    :param job:
    :param license:
    :param location:
    :param education:
    :param name:
    :param skills:
    :param search_country:
    :return: txt query for google
    """

    query = f"""
                site:{search_country}.linkedin.com/in/
               -inurl:people {name}
               -inurl:experiences {experience}
               -inurl:companies {company}
               -inurl:jobs {job}
               -inurl:licenses {license}
               -inurl:location {location}
               -inurl:Education {education}
               -inurl:skills {skills}
                """
    query = query.replace('\n', ' ').strip()
    return query


if __name__ == '__main__':
    # query = 'linkedin "andrey-kozlov-37136161" at'
    query = query_constructor(company='sberbank')
    # df = query_bing(query)
    search_args = (query, 1, False)
    gsearch = GoogleSearch()
    # ysearch = YahooSearch()
    # yresults = ysearch.search(*search_args)
    # print(yresults.results)
    # ysearch = YandexSearch()
    gresults = gsearch.search(*search_args)

    # yresults = ysearch.search(*search_args)
    # print(yresults.results)
    # # bresults = ysearch.search(*search_args)
    # a = {
    #     "Google": gresults,
    #     "Yahoo": yresults,
    #     "Bing": bresults
    # }
    #
    # # pretty print the result from each engine
    # for k, v in a.items():
    #     print(f"-------------{k}------------")
    #     for result in v:
    #         pprint.pprint(result)
    #
    # # print first title from google search
    # print(gresults["titles"][0])
    # # print 10th link from yahoo search
    # print(yresults["links"][9])
    # # print 6th description from bing search
    # print(bresults["descriptions"][5])
    #
    # # print first result containing links, descriptions and title
    # print(gresults[1])
