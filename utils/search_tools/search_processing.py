import pandas as pd


# Linkedin google search results processing

def li_link_to_username(link):
    return link.split('/in/')[1].split('/')[0]


def li_links_to_usernames(links: pd.Series):
    return links.str.split('/in/').str[1].str.split('/').str[0]


def li_username_to_link(username):
    return 'https://www.linkedin.com/in/' + str(username)


def li_enrich_gsearch_results(df):
    """
    Parse google search results on linkedin, extract relevant info
    :param df:
    :return:
    """

    # todo: test no google search results case
    if df is None:
        df_dict = {}
        for i in ['name', 'position', 'titles', 'employer', 'descriptions',
                  'comment', 'score', 'username', 'search_names']:
            df_dict[i] = None
        return pd.DataFrame({df_dict})


    df['name'] = df['titles'].str.split(' – ', expand=True)[0].str.split(' - ', expand=True)[0]
    if df['titles'].str.contains(' - ').sum() > 0:
        df['position'] = df['titles'].str.split(' - ', expand=True)[1]

    if df['titles'].str.contains(' – ').sum() > 0:
        df['position'] = df['position'].fillna(df['titles'].str.split(' – ', expand=True)[1])
    if df['titles'].str.contains(' – ').sum() + df['titles'].str.contains(' - ').sum() == 0:
        df['position'] = 'ERROR in parsing'

    try:
        df['employer'] = df['titles'].str.split(' - ', expand=True)[2].str.replace(' \| LinkedIn', '')
    except:
        df['employer'] = None

    if df['descriptions'].str.contains(' at ').sum() > 0:
        df['employer'] = df['employer'].fillna(
            df['descriptions'].str.split(' at ', expand=True)[1].str.split('.', expand=True)[0])
    df['employer'] = df['employer'].str.replace('\.\.\.', '')
    df['comment'] = df['descriptions']

    df['username'] = li_links_to_usernames(df['link'])

    df['search_names'] = df.groupby('username')['Search name'].transform(lambda x: ';'.join(x))
    df['score'] = (100-df['rank'])
    df['score'] = df.groupby('username')['score'].transform(sum)
    df = df.sort_values('score', ascending=False).drop_duplicates('username', keep='first').copy()
    first_col = df.pop('link')
    df.insert(0, 'link', first_col)
    return df



# other website search results processing
