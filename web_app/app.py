import uuid
from datetime import timedelta
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException
from starlette.responses import FileResponse
from starlette.responses import RedirectResponse

from utils.database import db, conn
# from utils.search_tools.general_search import async_query_google, async_get_info_multiple_queries, query_constructor
# from utils.search_tools.search_processing import li_enrich_gsearch_results
from web_app.datamodels import User, Event, UpdateAttendee

user_list = ['salim', 'jean', 'karina']

app = FastAPI(debug=True)
SECRET = 'df9070e49ffee6fd61a7ebe1c6e2b5e8e960122e4ce6fafd'
# To obtain a suitable secret key you can run | import os; print(os.urandom(24).hex())
manager = LoginManager(SECRET, token_url="/auth/token", use_cookie=True)
manager.cookie_name = "session_access"

app.mount("/static", StaticFiles(directory="web_app/static"), name="static")
templates = Jinja2Templates(directory="web_app/templates")


@manager.user_loader()
def load_user(email: str):  # could also be an asynchronous function

    # login = email.split('@')[0]
    login = email
    if login.lower() in user_list:
        return login
    else:
        return None


class NotAuthenticatedException(Exception):
    pass


# these two argument are mandatory
def exc_handler(request, exc):
    return RedirectResponse(url='/login')


manager.not_authenticated_exception = NotAuthenticatedException
# You also have to add an exception handler to your app instance
app.add_exception_handler(NotAuthenticatedException, exc_handler)
app.add_exception_handler(401, exc_handler)


# app.add_exception_handler(405, exc_handler)

@app.post('/redirect_login')
async def redirect_login(request: Request):
    return RedirectResponse('/login', status_code=401)


@app.post('/auth/token')
async def login(data: OAuth2PasswordRequestForm = Depends()):
    print('from auth/token', data)
    email = data.username.lower()
    password = data.username.lower()

    print(email)
    user = load_user(email)  # we are using the same function to retrieve the user

    log_dict = {'code': user, 'page_name': 'login_auth', 'request': 1,
                'params': f'{email}_{password}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    if not user:
        # RedirectResponse('/login', status_code=401)
        raise InvalidCredentialsException
        # raise NotAuthenticatedException  # you can also use your own HTTPException
    elif password != user:
        # RedirectResponse('/login', status_code=401)
        raise InvalidCredentialsException
        # raise InvalidCredentialsException

    access_token = manager.create_access_token(
        data=dict(sub=email), expires=timedelta(hours=24)
    )
    resp = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    manager.set_cookie(resp, access_token)
    return resp


@app.get('/')
async def event_board(request: Request, user=Depends(manager)):
    code = user.lower()
    log_dict = {'code': code, 'page_name': 'event_board', 'request': 1,
                'params': ''}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    query = f"""
                select eb.*, at.attendance_status from event_board eb
                left outer join 
                (select * from attendees at where at.attendee='{user}') at
                on at.event_id= eb.id
            """
    query = query.replace('\n', '').strip()
    events_df = pd.read_sql_query(query, db)
    search = {'code': user}
    rename_dict = {'name': 'Event name',
                   'event_type': 'Event category',
                   'event_date': 'Event date',
                   'venue': 'Venue',
                   'description': 'Description',
                   'added_user': 'Creator',
                   'timestamp': 'timestamp',
                   'attendance_status': 'Going?'}

    return templates.TemplateResponse("search_params.html",
                                      context={'request': request, 'search': search,
                                               'cand_list': events_df,
                                               'rename_dict': rename_dict})


@app.get('/create_event')
async def create_event(request: Request, user=Depends(manager)):
    code = user

    # todo - add other request params
    log_dict = {'code': code, 'page_name': 'create_event', 'request': 1,
                'params': ''}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    events_df = pd.read_sql_query(f"select * from event_board eb", db)
    search = {'code': user}
    rename_dict = {'name': 'Event name',
                   'event_type': 'Event category',
                   'event_date': 'Event date',
                   'venue': 'Venue',
                   'description': 'Description',
                   'username': 'Linkedin profile',
                   'link': 'link',
                   'score': 'Score',
                   'recruiter_comment': 'Recruiter comment',
                   'exists': 'In project'}

    return templates.TemplateResponse("add_event.html",
                                      context={'request': request, 'search': search,
                                               'rename_dict': rename_dict})


@app.get('/update_event/{event_id}')
async def update_event(event_id: str, request: Request, user=Depends(manager)):
    code = user

    # todo - add other request params
    log_dict = {'code': code, 'page_name': 'update_event', 'request': 1,
                'params': f'{event_id}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    events_df = pd.read_sql_query(f"select * from event_board eb where eb.id='{event_id}'", db)
    search = {'code': user}

    events_df = events_df.to_dict()
    for i in events_df.keys():
        # if df[i][0] is not None:
        search[i] = events_df[i][0]

    search['post_type'] = 'update'
    search['event_date_formatted'] = f"{str(search['event_date'].date())}T{str(search['event_date'].time())}"
    rename_dict = {'name': 'Event name',
                   'event_type': 'Event category',
                   'event_date': 'Event date',
                   'venue': 'Venue',
                   'description': 'Description'}

    return templates.TemplateResponse("add_event.html",
                                      context={'request': request, 'search': search,
                                               'rename_dict': rename_dict})

@app.post('/change_event/{event_id}')
async def change_event(event_id:str,request: Request, user=Depends(manager)):
    changed_event = {**await request.form()}
    print(changed_event)
    container = changed_event
    code = user
    log_dict = {'code': code, 'page_name': 'event_board', 'request': 2,
                'params': f'changed_event_{changed_event["event_name"]}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    table_name='event_board'
    print('only update')
    query = f'''update {table_name} eb
                            set event_type='{changed_event['event_type']}',
                             event_date='{changed_event['event_date']}',
                             venue='{changed_event['venue']}',
                             name='{changed_event['event_name']}',
                             description='{changed_event['description']}',
                            timestamp='{pd.Timestamp.now()}'
                            where id = '{event_id}' 
                            '''
    query = query.replace('\n', '').strip()
    db.execute(query)

    resp = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return resp


@app.post('/')
async def add_event(request: Request, user=Depends(manager)):
    added_event = {**await request.form()}
    print(added_event)
    container = added_event
    code = user
    log_dict = {'code': code, 'page_name': 'event_board', 'request': 2,
                'params': f'added_event_{added_event["event_name"]}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    event = {
        'id': [uuid.uuid4()],
        'name': [added_event['event_name']],
        'event_type': [added_event['event_type']],
        'event_date': [added_event['event_date']],
        'venue': [added_event['venue']],
        'timestamp': [pd.to_datetime('now', utc=True)],
        'added_user': [user],
        'description': [added_event['description']]
    }
    event = pd.DataFrame.from_dict(event)
    table_name = 'event_board'
    event.to_sql(table_name, db, if_exists='append', index=False)

    resp = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return resp


@app.post('/update_attendee')
async def update_attendee(update_attendee: UpdateAttendee, user=Depends(manager)):
    # todo we need to add more identifiers here

    # Log page clicks
    log_dict = {'code': update_attendee.attendee, 'page_name': 'update_attendee', 'request': 2,
                'params': f'update_attendee_to_{update_attendee.event_id}_{update_attendee.attendance_status}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)
    print(update_attendee.dict())

    # df = pd.DataFrame(rec_comment.dict(), index=[pd.to_datetime('now')] )
    table_name = 'attendees'
    query = f'''select * from {table_name} at
                where at.attendee='{update_attendee.attendee}'
                and at.event_id = '{update_attendee.event_id}' 
             '''
    query = query.replace('\n', '').strip()
    # Todo check if candidate exists
    df = pd.read_sql_query(query, conn)

    # todo: prohibit one user from accessing projects of another user
    if df.empty:
        print('update_attendee: adding attendee, because such record does not exist')
        # todo add row if username within project does not exist
        await log_object_to_db(update_attendee.dict(), table_name, db)
    else:
        # todo update if exists
        # todo think here if needed update of other fields (they might change with time)
        print('only update')
        query = f'''update {table_name} rt
                        set attendance_status='{update_attendee.attendance_status}',
                        timestamp='{pd.to_datetime('now')}'
                        where attendee='{update_attendee.attendee}'
                        and event_id = '{update_attendee.event_id}' 
                        '''
        query = query.replace('\n', '').strip()
        db.execute(query)

    return update_attendee


@app.get('/events/{event_id}')
async def get_event(event_id: str, request: Request, user=Depends(manager)):
    # TODO make search link from search ID (rather than showing search ID it self)

    code = user
    # Log page clicks
    log_dict = {'code': code, 'page_name': 'event', 'request': 1, 'params': f'event_{event_id}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    table_name = 'event_board'

    event_by_id = pd.read_sql_query(
        f'''select  *
            from  {table_name} 
            where id='{event_id}'
            order by timestamp desc''', conn, parse_dates=True)

    query = f"""
                 select at.attendee, at.attendance_status from event_board eb
                 left outer join attendees at
                 on at.event_id= eb.id
                 where eb.id='{event_id}' 
             """

    attendees = pd.read_sql_query(query, conn, parse_dates=True)

    # todo: create a user table

    # todo: rename dict

    # todo: reorder columns in proper manner
    search = {}
    search['code'] = code
    search['id'] = event_id

    rename_dict = {'name': 'Event name',
                   'event_type': 'Event category',
                   'event_date': 'Event date',
                   'venue': 'Venue',
                   'description': 'Description'}

    # cand_list.pop('index')

    return templates.TemplateResponse("event.html",
                                      context={'request': request, 'search': search,
                                               'attendees': attendees, 'event_info': event_by_id,
                                               'rename_dict': rename_dict, 'cand_list': ''})


# async def update_attendee(rec_comment: Rec_comment, user=Depends(manager)):
#     # todo we need to add more identifiers here
#
#     # Log page clicks
#     log_dict = {'code': rec_comment.code, 'page_name': 'add_cand', 'request': 2,
#                 'params': f'add_to_project_{rec_comment.project_name}'}
#     await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)
#     print(rec_comment.dict())
#
#     # df = pd.DataFrame(rec_comment.dict(), index=[pd.to_datetime('now')] )
#     table_name = 'recruiter_list'
#     query = f'''select * from {table_name} rt
#                 where rt.username='{rec_comment.username}'
#                 and rt.project_name = '{rec_comment.project_name}'
#              '''
#     query = query.replace('\n', '').strip()
#     # Todo check if candidate exists
#     df = pd.read_sql_query(query, conn)
#
#     # todo: prohibit one user from accessing projects of another user
#     if df.empty:
#         print('add_candidate: adding, because such record does not exist')
#         # todo add row if username within project does not exist
#         await log_object_to_db(rec_comment.dict(), table_name, db)
#     else:
#         # todo update if exists
#         # todo think here if needed update of other fields (they might change with time)
#         print('only update')
#         query = f'''update {table_name} rt
#                         set comment='{rec_comment.comment}',
#                         timestamp='{pd.to_datetime('now')}'
#                         where username='{rec_comment.username}'
#                         and project_name = '{rec_comment.project_name}'
#                         '''
#         query = query.replace('\n', '').strip()
#         db.execute(query)
#
#     return rec_comment

# async def index(request: Request, user=Depends(manager), search_index: Optional[str] = ''):
#     search = Event()
#     search = dict(search)
#
#     code = user
#     search['code'] = code
#     print(user)
#     # Log page clicks
#     log_dict = {'code': code, 'page_name': 'event_board', 'request': 1,
#                 'params': f'project:{search["project_name"]}, search:{search_index}'}
#     await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)
#
#     if search_index != '':
#         print(search_index)
#         df = pd.read_sql_query(f"select * from log_searches ls where ls.index = '{search_index}'", conn)
#         df = df.to_dict()
#         for i in search:
#             # if df[i][0] is not None:
#             search[i] = df[i][0]
#         cand_list, rename_dict = await search_cand_button_action(search_index, search)
#         activity_tabs = {'search': 'active', 'cand_sheet': 'active'}
#
#         return templates.TemplateResponse("search_params.html",
#                                           context={'request': request, 'search': search,
#                                                    'cand_list': cand_list,
#                                                    'rename_dict': rename_dict, 'activity_tabs': activity_tabs})
#
#     rename_dict = {'name': 'Name (auto-detect)',
#                    'employer': 'Employer (auto-detect)',
#                    'position': 'Position (auto-detect)',
#                    'descriptions': 'Comment from google',
#                    'search_names': 'Found in searches',
#                    'username': 'Linkedin profile',
#                    'link': 'link',
#                    'score': 'Score',
#                    'recruiter_comment': 'Recruiter comment',
#                    'exists': 'In project'}
#     cand_list = pd.DataFrame()
#     return templates.TemplateResponse("search_params.html",
#                                       context={'request': request, 'search': search,
#                                                'cand_list': cand_list,
#                                                'rename_dict': rename_dict})


@app.post('/')
async def cand_sheet(request: Request, user=Depends(manager)):
    search = Search(**await request.form())  # contains search params
    search = search.dict()
    print(search)
    search['code'] = user
    log_id = await log_object_to_db(search, table_name='log_searches', db=db)

    cand_list, rename_dict = await search_cand_button_action(log_id, search)
    # cand_list = cand_list.rename(columns=rename_dict)
    # cand_list = cand_list.drop(['search_id'], axis=1)

    # cand_list.to_excel('saved_file.xlsx')
    # return render_template('cand_sheet.html', cand_list=df)
    return templates.TemplateResponse("search_params.html",
                                      context={'request': request, 'search': search,
                                               'cand_list': cand_list,
                                               'rename_dict': rename_dict})


async def search_cand_button_action(log_id, search):
    rename_dict = {'name': 'Name (auto-detect)',
                   'employer': 'Employer (auto-detect)',
                   'position': 'Position (auto-detect)',
                   'descriptions': 'Comment from google',
                   'search_names': 'Found in searches',
                   'username': 'Linkedin profile',
                   'link': 'link',
                   'score': 'Score',
                   'recruiter_comment': 'Recruiter comment',
                   'exists': 'In project'}
    queries = {}
    queries[f'{log_id}'] = query_constructor(experience=search['experience'],
                                             company=search['company'],
                                             job=search['job'],
                                             skills=search['skill'],
                                             license=search['license'],
                                             location=search['location'])
    # get google search results
    df = await async_get_info_multiple_queries(queries, fun=async_query_google)
    df = li_enrich_gsearch_results(df)
    first_col = df.pop('link')
    df.insert(0, 'link', first_col)
    # df['username'] = "<a href=" + df['link'] + ">" + df['username']+ "</a>"
    # print(df['test'])
    df = df[['username', 'link', 'name', 'employer', 'position', 'descriptions', 'search_names', 'score']].copy()
    print(df)
    table_name = 'recruiter_list'
    recruiter_df = pd.read_sql_query(f'''select * from {table_name} tsr
                                where tsr.project_name='{search['project_name']}'
                            ''', db, parse_dates=True)
    # # delete data related to user
    # pd.read_sql_query(f'''DELETE FROM {tmp_table} WHERE condition
    #                                 where code = "{search['code']}" ''')
    df['search_id'] = log_id
    df['recruiter_comment'] = ''
    # df.to_sql('tmp_search_results', db, if_exists='append')
    df = df.merge(recruiter_df[['comment', 'username']], on=['username'], how='left', validate='1:1')
    df.loc[~df['comment'].isna(), 'recruiter_comment'] = df['comment']
    df['exists'] = False
    df.loc[~df['comment'].isna(), 'exists'] = True
    df = df.drop('comment', axis=1)
    df = df[['recruiter_comment', 'exists', 'name', 'employer', 'position', 'username', 'link', 'descriptions',
             'search_names', 'score']].copy()
    cand_list = df.sort_values('score', ascending=False).copy()
    cand_list = cand_list.reset_index(drop=True)
    return cand_list, rename_dict


async def log_object_to_db(search, table_name, db):
    log_id = f'{uuid.uuid4()}'
    log_df = pd.DataFrame(search, index=[log_id]).reset_index()
    log_df['timestamp'] = pd.to_datetime('now', utc=True)
    print(log_df)
    try:
        log_df.to_sql(table_name, db, if_exists='append', index=False)
        print(
            f'WARNING FROM log_object_to_db: could not save initial log_df to {table_name} table, converting to string')
    except:
        log_df.applymap(str).to_sql(table_name, db, if_exists='append', index=False)

    return log_id


@app.get('/download_cand_sheet/{project_name}')
async def download_cand_sheet(project_name: str, request: Request, user=Depends(manager)):
    code = user
    if code is None:
        code = ''
    # Log page clicks
    log_dict = {'code': code, 'page_name': 'download_xlsx', 'request': 1, 'params': f'project_{project_name}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    table_name = 'recruiter_list'
    recruiter_df = pd.read_sql_query(f'''select * from {table_name} tsr
                                    where tsr.project_name='{project_name}'
                                ''', db, parse_dates=True)

    recruiter_df['link'] = 'https://www.linkedin.com/in/' + recruiter_df['username']
    recruiter_df.to_excel(f'saved_file.xlsx')

    return FileResponse(f'saved_file.xlsx', media_type='text/csv', filename=f'{project_name}.xlsx')


@app.get('/projects/{project_name}')
async def get_project(project_name: str, request: Request, user=Depends(manager)):
    # TODO make search link from search ID (rather than showing search ID it self)

    code = user
    # Log page clicks
    log_dict = {'code': code, 'page_name': 'project', 'request': 1, 'params': f'project_{project_name}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    table_name = 'recruiter_list'

    cand_list = pd.read_sql_query(
        f'''select  *
                                        from  {table_name} 
                                        where project_name='{project_name}'
                                        order by timestamp desc''', db, parse_dates=True)

    cand_list['link'] = 'https://www.linkedin.com/in/' + cand_list['username']

    # todo: rename dict

    # todo: reorder columns in proper manner
    search = {}
    search['code'] = code
    search['project_name'] = project_name

    cand_list = cand_list[
        ['name', 'employer', 'position', 'comment', 'code', 'timestamp', 'descriptions', 'search_names', 'link']]
    rename_dict = {'name': 'Name',
                   'employer': 'Employer',
                   'code': 'Recruiter who added',
                   'position': 'Position',
                   'descriptions': 'Comment from google',
                   'search_names': 'Found in searches',
                   'link': 'Link',
                   'score': 'Score',
                   'comment': 'Recruiter comment',
                   'timestamp': 'Timestamp'}

    # cand_list.pop('index')

    return templates.TemplateResponse("event.html",
                                      context={'request': request, 'search': search,
                                               'cand_list': cand_list,
                                               'rename_dict': rename_dict})


@app.get('/projects')
async def get_projects(request: Request, user=Depends(manager)):
    code = user
    log_dict = {'code': code, 'page_name': 'projects', 'request': 1}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    table_name = 'recruiter_list'

    projects = pd.read_sql_query(
        f'''select  tsr.project_name, count(tsr.index) n_candidates, max(tsr.timestamp) as last_update 
                            from  {table_name} as tsr
                            where tsr.project_name in (select distinct project_name {table_name} 
                                                        where code='{code}' ) 
                            group by tsr.project_name
                            order by max(tsr.timestamp) desc
                                        ''', db, parse_dates=True)

    search = {}
    search['code'] = code

    return templates.TemplateResponse("my_projects.html",
                                      context={'request': request, 'search': search, 'projects': projects})


@app.get('/login')
async def get_login(request: Request):
    log_dict = {'code': None, 'page_name': 'login', 'request': 1,
                'params': None}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)
    error_text = None
    return templates.TemplateResponse("simple-login.html", context={'request': request, 'error_text': error_text})


@app.post('/login')
async def get_login(request: Request):
    request_params = {**await request.form()}

    try:
        code = request_params['username'].split('@')[0]
    except:
        code = ''
    log_dict = {'code': code, 'page_name': 'login_post', 'request': 1,
                'params': f'{request_params}'}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    error_text = 'Enter correct login and password!\n For help - info@smeanalytica.com'
    return templates.TemplateResponse("simple-login.html", context={'request': request, 'error_text': error_text})


# @app.post("/login")
# async def login(request: Request, login: str = Form(...)):
#     print(f'from /login {login}')
#     # todo: add google login, add password
#     if login.lower() not in user_list:
#         login = 'Please, enter a correct code!'
#         return templates.TemplateResponse("simple-login.html", context={'request': request, 'login': login})
#
#     else:
#         return RedirectResponse(url=f'/?code={login}')


@app.get('/signup')
def get_signup(request: Request):
    return templates.TemplateResponse("signup.html", context={'request': request})


@app.get('/forgot_password')
def get_forgot_password(request: Request):
    return templates.TemplateResponse("forgot-password.html", context={'request': request})


@app.get('/searches')
async def get_my_searches(request: Request, user=Depends(manager)):
    code = user
    # log page clicks
    log_dict = {'code': code, 'page_name': 'my_searches', 'request': 1}
    await log_object_to_db(log_dict, table_name='log_page_clicks', db=db)

    query = f'''select * from log_searches ls
                where ls.code='{code}'
                order by ls.timestamp desc'''

    df = pd.read_sql_query(query, con=db)
    # todo add project name
    df = df.drop_duplicates(['company', 'experience', 'job', 'skill', 'location', 'license', 'code'], keep="first")
    # df.drop_duplicates([])
    print(df)
    search = {}
    search['code'] = code

    return templates.TemplateResponse("my_searches.html",
                                      context={'request': request, 'search': search, 'searches': df})


if __name__ == "__main__":
    uvicorn.run(app)

    # uvicorn.run(app, host="0.0.0.0", port=8000)
