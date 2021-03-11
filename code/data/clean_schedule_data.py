import pandas as pd
import numpy as np
import os
import datetime

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
schedule_data_dir = PROJECT_DIR + '/data/raw/Schedule Data/'

# Output files
processed_schedule_data_file = PROJECT_DIR + '/data/clean/schedule_data.csv'

##### Cleaning Functions
def clean_calendar_data():
    filenames = os.listdir(schedule_data_dir)
    filenames = [schedule_data_dir + name for name in filenames if '.xlsx' in name]
    df = pd.DataFrame()
    for filename in filenames:
        mdf = pd.read_excel(filename)
        mdf = process_month_df(mdf)
        df = df.append(mdf,ignore_index=True)

    return(df)

def process_month_df(tdf):
    date_year = tdf.columns[0]
    month = date_year.split(' ')[0]
    month_num = datetime.datetime.strptime(month,'%B').month
    year = int(date_year.split(' ')[1])

    tdf.rename(columns={tdf.columns[0]:'JudgeName'},inplace=True)
    tdf = tdf.melt(id_vars='JudgeName',var_name='StartDay',value_name='FullAssignment')
    tdf['TempWeek'] =  tdf['StartDay'].apply(lambda x: datetime.date(year,month_num,x).isocalendar()[1])
    tdf['Week'] = tdf['TempWeek'].apply(lambda x: str(x)+'-'+str(year))
    tdf.drop(columns=['TempWeek','StartDay'],inplace=True)

    tdf['FullAssignment'] = tdf['FullAssignment'].fillna('na')

    tdf['Assignment1'] = tdf['FullAssignment'].apply(lambda x: x.split('+')[0])
    tdf['Assignment2'] = tdf['FullAssignment'].apply(get_half)
    tdf['Assignment3'] = tdf['Assignment2'].apply(get_half)
    tdf['Assignment4'] = tdf['Assignment3'].apply(get_half)
    tdf['Assignment5'] = tdf['Assignment4'].apply(get_half)

    tdf['Assignment2'] = tdf['Assignment2'].apply(lambda x: x.split('+')[0])
    tdf['Assignment3'] = tdf['Assignment3'].apply(lambda x: x.split('+')[0])
    tdf['Assignment4'] = tdf['Assignment4'].apply(lambda x: x.split('+')[0])
    tdf['Assignment5'] = tdf['Assignment5'].apply(lambda x: x.split('+')[0])
    tdf = tdf.melt(id_vars=['JudgeName','FullAssignment','Week'],value_name='Assignment')
    tdf = tdf.loc[(tdf.FullAssignment == 'na') | (tdf.Assignment != 'na'),:].drop(columns=['variable'])

    tdf['JudgeName'] = tdf['JudgeName'].str.upper()
    tdf['JudgeName'] = tdf['JudgeName'].str.replace('[^A-Za-z\s]','')
    tdf['JudgeName'] = tdf['JudgeName'].str.strip()
    tdf.loc[tdf.JudgeName == 'COOPER', 'JudgeName'] = 'COOPER TW'
    return(tdf)

def get_half(assignment):
    split_assignment = assignment.split('+')
    if len(split_assignment) > 1:
        return '+'.join(split_assignment[1:])
    else:
        return 'na'

def main():
    df = clean_calendar_data()
    df.to_csv(processed_schedule_data_file,index=False)

main()
