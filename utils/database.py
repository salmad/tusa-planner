from utils.config import get_config
import pandas as pd
import uuid

config = get_config()
# db_path = config['data']['db_path']
from sqlalchemy import create_engine

# postgresql instead of postgres
new_URI = 'postgresql://ywierfvrcrvojs:143304102db922db05981e8353f2cfe201bf3cfe6eb0627a84cf74c62a064db5@ec2-3-223-213-207.compute-1.amazonaws.com:5432/d9scupe8kmhosv'
db = create_engine(new_URI)

username = 'ywierfvrcrvojs'
password = '143304102db922db05981e8353f2cfe201bf3cfe6eb0627a84cf74c62a064db5'
database = 'd9scupe8kmhosv'
hostname = 'ec2-3-223-213-207.compute-1.amazonaws.com'

# conn = sqlite3.connect(db_path)
# db = sqlite3.connect(db_path, isolation_level=None, detect_types=sqlite3.PARSE_COLNAMES)

conn = db.connect().connection
# db = psycopg2.connect(user=username,
#                       password=password,
#                       host = hostname,
#                       database=database,
#                       port=5432)


# Initiate tables
# Event board
# event = {
#     'id': [uuid.uuid4(), uuid.uuid4()],
#     'name': ['Karaoke with Debs', 'Thirsty thursday'],
#     'event_type': ['Karaoke', 'Corporate'],
#     'event_date': [pd.to_datetime('13/05/2022, 7pm'), pd.to_datetime('05/05/2022, 7pm')],
#     'venue': ["Cash studio", "Jeffo's"],
#     'timestamp': [pd.Timestamp.now(), pd.Timestamp.now()],
#     'added_user': ['Salim', 'Karina'],
#     'description': ['', '']
# }
# event = pd.DataFrame.from_dict(event)
# table_name = 'event_board'
# event.to_sql(table_name, db, if_exists='append', index=False)

# Attendies
attendees = {
    'index':[uuid.uuid4().hex],
    'event_id': [uuid.uuid4().hex],
    'attendee':['Salim'],
    'attendance_status':['Yes'],
    'timestamp':[pd.to_datetime('now',utc=True)]
}
attendees = pd.DataFrame.from_dict(attendees)
table_name = 'attendees'
attendees.to_sql(table_name, db, if_exists='append', index=False)
