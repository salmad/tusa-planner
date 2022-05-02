from typing import Optional
import uuid
import datetime
from pydantic.main import BaseModel
import pandas as pd




class Event(BaseModel):
    id = uuid.UUID
    name: str
    event_type: Optional[str] =''
    event_date: Optional[datetime.datetime]
    venue: Optional[str] = 'TBD'
    description: Optional[str] =''

class User(BaseModel):
    id: uuid.UUID

    name: str

    class Config:
        orm_mode = True

class Venue(BaseModel):
    id = uuid.UUID
    name: str
    address: Optional[str] =''
    venue_url: Optional[datetime.datetime]
    added_user: User
    description: Optional[str] =''
    class Config:
        orm_mode = True

class UpdateAttendee(BaseModel):
    event_id: str
    attendee: str
    attendance_status: str
    class Config:
        orm_mode = True




class Rec_comment(BaseModel):
    code: str #todo: rename to user(recruiter)
    project_name: str
    username: str
    name: str
    employer:str
    position:str
    descriptions: str
    search_names:str
    score: float
    comment: str


class Project(BaseModel):
    project_id: Optional[int] = ''
    user_id: Optional[int] = ''
    experience: Optional[str] = ''
    job: Optional[str] = ''
    skill: Optional[str] = ''
    location: Optional[str] = 'Russia'
    license: Optional[str] = ''
    project_name: Optional[str] = ''


# creating indices

# query=f'''
# CREATE INDEX rec_index ON recruiter_list
# (
#         project_name DESC,
#         timestamp DESC,
#         index DESC
# );
# '''
# query = query.replace('\n','').strip()
#
# db.execute(query)