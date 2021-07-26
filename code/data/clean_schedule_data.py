import pandas as pd
import numpy as np
import os
import datetime
import re

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
schedule_data_dir = PROJECT_DIR + '/data/raw/Schedule Data/'
county_file = PROJECT_DIR + '/data/raw/county_list.csv'

# Output files
processed_weekly_schedule_file = PROJECT_DIR + '/data/processed/weekly_schedule_data.csv'
processed_daily_schedule_file = PROJECT_DIR + '/data/processed/daily_schedule_data.csv'

class ParserHelper:
    def __init__(self):
        self.date_regex = re.compile('([\d]+)')
        self.plus_regex = re.compile('[+]+')
        self.numbers_no_letters_regex = re.compile('^[\d\W,]+$')
        self.circuit_regex = re.compile('([\d]+st|[\d]+nd|[\d]+rd|[\d]+th) Cir\.?')
        cdf = pd.read_csv(county_file)
        self.assignments = '|'.join(cdf.County.unique()[::-1]) #flipping order so that Chesterfield is before Chester
        self.assignments += '|([\d]+st|[\d]+nd|[\d]+rd|[\d]+th) Cir\.?'

    def is_single_assignment_nd(self,assignment):
        plus_match = self.plus_regex.search(assignment)
        split_str = assignment.split()
        dates = [substr for substr in split_str if
                        self.numbers_no_letters_regex.search(substr) is not None]
        return (plus_match is None and len(dates) == 0)

    def is_single_assignment_wd(self,assignment):
        plus_match = self.plus_regex.search(assignment)
        split_str = assignment.split()
        dates = [substr for substr in split_str if
                        self.numbers_no_letters_regex.search(substr) is not None]
        return (plus_match is None and len(dates) > 0)

    def is_multiple_assignment_nd(self,assignment):
        plus_match = self.plus_regex.search(assignment)
        split_ass = assignment.split('+')
        with_dates = [ass for ass in split_ass if self.is_single_assignment_wd(ass)]
        return (plus_match is not None and len(with_dates) == 0)

    def is_multiple_assignment_sd(self,assignment):
        plus_match = self.plus_regex.search(assignment)
        split_ass = assignment.split('+')
        with_dates = [ass for ass in split_ass if self.is_single_assignment_wd(ass)]
        return (plus_match is not None and len(with_dates) < len(split_ass))

    def is_multiple_assignment_ad(self,assignment):
        plus_match = self.plus_regex.search(assignment)
        split_ass = assignment.split('+')
        with_dates = [ass for ass in split_ass if self.is_single_assignment_wd(ass)]
        return (plus_match is not None and len(with_dates) == len(split_ass))

    def process_single_assignment_nd(self,row):
        year, week, start_day = row['ISOWeek']
        assignment_type = 'Missing' if row['FullAssignment'] == 'na' else 'Single'
        days = [(year,week,start_day+i) for i in range(5)]
        dates = [datetime.date.fromisocalendar(year,week,day) for year,week,day in days]
        new_rows = [{'JudgeName':row['JudgeName'],'FullAssignment':row['FullAssignment'],
           'Assignment':row['FullAssignment'],'Date':date,'AssignmentType':assignment_type,
           'StartDate':dates[0]}
                    for date in dates]
        return(new_rows)

    def process_single_assignment_wd(self,row):
        year, week, start_day = row['ISOWeek']
        start_date = datetime.date.fromisocalendar(year,week,start_day)
        year = start_date.year
        month = start_date.month
        split_str = row['FullAssignment'].split()
        days = [substr for substr in split_str if
                        self.numbers_no_letters_regex.search(substr) is not None][0]
        days = self.date_regex.findall(days)
        dates = [datetime.date(year,month,int(day)) for day in days]
        dates = self.adjust_month(dates,start_date)
        all_days = [(year,week,start_day+i) for i in range(5)]
        all_dates = [datetime.date.fromisocalendar(year,week,day) for year,week,day in all_days]
        assigned_dates = dates
        assignment = self.remove_dates(row['FullAssignment'])
        new_rows = [{'JudgeName':row['JudgeName'],'FullAssignment':row['FullAssignment'],
           'Assignment':assignment,'Date':date,'AssignmentType':'SingleWD','StartDate':start_date}
                    for date in dates]

        remaining_dates = [date for date in all_dates if date not in assigned_dates]
        remaining_dates = self.adjust_month(remaining_dates,start_date)
        new_rows += [{'JudgeName':row['JudgeName'],'FullAssignment':row['FullAssignment'],
           'Assignment':'na','Date':date,'AssignmentType':'SingleWD','StartDate':start_date}
                        for date in remaining_dates]
        return(new_rows)

    def process_multiple_assignment_nd(self,row):
        year, week, start_day = row['ISOWeek']
        start_date = datetime.date.fromisocalendar(year,week,start_day)
        year = start_date.year
        month = start_date.month
        assignments = row['FullAssignment'].split('+')
        all_days = [(year,week,start_day+i) for i in range(5)]
        all_dates = [datetime.date.fromisocalendar(year,week,day) for year,week,day in all_days]
        all_dates = self.adjust_month(all_dates,start_date)
        new_rows = []
        for assignment in assignments:
            assignment_rows = [{'JudgeName':row['JudgeName'],'FullAssignment':row['FullAssignment'],
               'Assignment':assignment,'Date':date,'AssignmentType':'MultipleND','StartDate':start_date}
                        for date in all_dates]
            new_rows += assignment_rows

        return(new_rows)

    def process_multiple_assignment_sd(self,row):
        year, week, start_day = row['ISOWeek']
        start_date = datetime.date.fromisocalendar(year,week,start_day)
        year = start_date.year
        month = start_date.month
        assignments = row['FullAssignment'].split('+')
        undated_assignments = [ass for ass in assignments if self.is_single_assignment_nd(ass)]
        dated_assignments = [ass for ass in assignments if self.is_single_assignment_wd(ass)]
        all_days = [(year,week,start_day+i) for i in range(5)]
        all_dates = [datetime.date.fromisocalendar(year,week,day) for year,week,day in all_days]
        assigned_dates = []
        new_rows = []
        assignment_type = 'MultipleSD' if len(undated_assignments) == 1 else 'MultipleSDA'
        for assignment in dated_assignments:
            split_str = assignment.split()
            days = [substr for substr in split_str if
                            self.numbers_no_letters_regex.search(substr) is not None][0]
            days = self.date_regex.findall(days)
            dates = [datetime.date(year,month,int(day)) for day in days]
            dates = self.adjust_month(dates,start_date)
            assignment = self.remove_dates(assignment)
            assignment_rows = [{'JudgeName':row['JudgeName'],'FullAssignment':row['FullAssignment'],
               'Assignment':assignment,'Date':date,'AssignmentType':assignment_type,'StartDate':start_date}
                        for date in dates]
            new_rows += assignment_rows
            assigned_dates += dates

        remaining_dates = [date for date in all_dates if date not in assigned_dates]
        for assignment in undated_assignments:
            main_assignment = self.remove_dates(assignment)
            remaining_dates = self.adjust_month(remaining_dates,start_date)
            new_rows += [{'JudgeName':row['JudgeName'],'FullAssignment':row['FullAssignment'],
               'Assignment':main_assignment,'Date':date,'AssignmentType':assignment_type,'StartDate':start_date}
                            for date in remaining_dates]
        return(new_rows)

    def process_multiple_assignment_ad(self,row):
        year, week, start_day = row['ISOWeek']
        start_date = datetime.date.fromisocalendar(year,week,start_day)
        year = start_date.year
        month = start_date.month
        all_days = [(year,week,start_day+i) for i in range(5)]
        all_dates = [datetime.date.fromisocalendar(year,week,day) for year,week,day in all_days]
        assigned_dates = []
        assignments = row['FullAssignment'].split('+')
        new_rows = []
        for assignment in assignments:
            split_str = assignment.split()
            days = [substr for substr in split_str if
                            self.numbers_no_letters_regex.search(substr) is not None][0]
            days = self.date_regex.findall(days)
            dates = [datetime.date(year,month,int(day)) for day in days]
            dates = self.adjust_month(dates,start_date)
            assignment = self.remove_dates(assignment)
            assignment_rows = [{'JudgeName':row['JudgeName'],'FullAssignment':row['FullAssignment'],
               'Assignment':assignment,'Date':date,'AssignmentType':'MultipleAD','StartDate':start_date}
                        for date in dates]
            new_rows += assignment_rows
            assigned_dates += dates

        remaining_dates = [date for date in all_dates if date not in assigned_dates]
        remaining_dates = self.adjust_month(remaining_dates,start_date)
        new_rows += [{'JudgeName':row['JudgeName'],'FullAssignment':row['FullAssignment'],
           'Assignment':'na','Date':date,'AssignmentType':'MultipleAD','StartDate':start_date}
                        for date in remaining_dates]
        return(new_rows)

    def adjust_month(self,dates,start_date):
        threshold = datetime.timedelta(days=-1)
        time_deltas = [date - start_date for date in dates]
        in_next_month = [delta < threshold for delta in time_deltas]
        for i, cond in enumerate(in_next_month):
            if cond:
                dates[i] = self.add_month(dates[i])
        return dates

    def add_month(self,date):
        new_month = (date.month % 12) + 1
        if new_month == 1:
            new_date = datetime.date(date.year+1,new_month,date.day)
        else:
            new_date = datetime.date(date.year,new_month,date.day)
        return new_date

    def remove_dates(self,assignment):
        split_str = assignment.split()
        non_dates = [substr for substr in split_str if
                        self.numbers_no_letters_regex.search(substr) is None]
        return ' '.join(non_dates)

    def get_assignment(self,assignment):
        match = re.search(self.assignments,assignment,flags=re.IGNORECASE)
        if match is not None:
            return match.group(0)
        else:
            return assignment

    def get_circuit(self,assignment):
        if 'Cir' in assignment:
            circuit = self.date_regex.findall(assignment)[0]
            return circuit
        else:
            return 'na'

##### Cleaning Functions
def create_calendar_data(daily=False):
    filenames = os.listdir(schedule_data_dir)
    filenames = [schedule_data_dir + name for name in filenames if '.xlsx' in name]
    df = pd.DataFrame()
    for filename in filenames:
        mdf = pd.read_excel(filename)
        cols = [col for col in mdf.columns if 'Unnamed' not in str(col)]
        mdf = mdf.loc[:,cols]
        if daily:
            mdf = process_month_df_for_daily(mdf)
        else:
            mdf = process_month_df_for_weekly(mdf)
        df = df.append(mdf,ignore_index=True)
    df = add_day_weights(df)
    return(df)

def process_month_df_minimal(tdf):
    date_year = tdf.columns[0]
    month = date_year.split(' ')[0]
    month_num = datetime.datetime.strptime(month,'%B').month
    year = int(date_year.split(' ')[1])

    tdf.rename(columns={tdf.columns[0]:'JudgeName'},inplace=True)
    tdf = tdf.melt(id_vars='JudgeName',var_name='StartDay',value_name='FullAssignment')
    tdf['TempWeek'] =  tdf['StartDay'].apply(lambda x: datetime.date(year,month_num,x).isocalendar()[1])
    tdf['Week'] = tdf['TempWeek'].apply(lambda x: str(x)+'-'+str(year))
    tdf.drop(columns=['TempWeek'],inplace=True)

    tdf['FullAssignment'] = tdf['FullAssignment'].fillna('na')

    tdf['JudgeName'] = tdf['JudgeName'].str.upper()
    tdf['JudgeName'] = tdf['JudgeName'].str.replace('[^A-Za-z\s]','',regex=True)
    tdf['JudgeName'] = tdf['JudgeName'].str.strip()
    tdf.loc[tdf.JudgeName == 'COOPER', 'JudgeName'] = 'COOPER TW'
    judges_to_drop = ['BERGDORF','COUCH','DREW','GIER','PEEPLES','SIMMONS','WATTS','YOUNG']
    tdf = tdf.loc[~tdf.JudgeName.isin(judges_to_drop),:]
    return(tdf)

def process_month_df_for_weekly(tdf):
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
    tdf['JudgeName'] = tdf['JudgeName'].str.replace('[^A-Za-z\s]','',regex=True)
    tdf['JudgeName'] = tdf['JudgeName'].str.strip()
    tdf.loc[tdf.JudgeName == 'COOPER', 'JudgeName'] = 'COOPER TW'
    judges_to_drop = ['BERGDORF','COUCH','DREW','GIER','PEEPLES','SIMMONS','WATTS','YOUNG']
    tdf = tdf.loc[~tdf.JudgeName.isin(judges_to_drop),:]
    return(tdf)

def process_month_df_for_daily(tdf):
    date_year = tdf.columns[0]
    month = date_year.split(' ')[0]
    month_num = datetime.datetime.strptime(month,'%B').month
    year = int(date_year.split(' ')[1])

    tdf.rename(columns={tdf.columns[0]:'JudgeName'},inplace=True)
    tdf = tdf.melt(id_vars='JudgeName',var_name='StartDay',value_name='FullAssignment')
    tdf['StartDay'] = tdf['StartDay'].apply(get_monday,args=(month_num,year,))
    tdf['ISOWeek'] =  tdf['StartDay'].apply(lambda x: datetime.date(year,month_num,x).isocalendar())
    tdf['Week'] = tdf['StartDay'].apply(lambda x: datetime.date(year,month_num,x))
    tdf['FullAssignment'] = tdf['FullAssignment'].fillna('na')
    ph = ParserHelper()
    all_assignments = []
    for idx,row in tdf.iterrows():
        full_assignment = row['FullAssignment']
        if ph.is_single_assignment_nd(full_assignment):
            daily_assignments = ph.process_single_assignment_nd(row)
        elif ph.is_single_assignment_wd(full_assignment):
            daily_assignments = ph.process_single_assignment_wd(row)
        elif ph.is_multiple_assignment_nd(full_assignment):
            daily_assignments = ph.process_multiple_assignment_nd(row)
        elif ph.is_multiple_assignment_sd(full_assignment):
            daily_assignments = ph.process_multiple_assignment_sd(row)
        elif ph.is_multiple_assignment_ad(full_assignment):
            daily_assignments = ph.process_multiple_assignment_ad(row)
        else:
            print('Error, unidentified row \n')
            print(row)
            break
        all_assignments += daily_assignments

    tdf = pd.DataFrame(all_assignments)

    tdf['Assignment'] = tdf['Assignment'].str.strip()
    tdf['WorkType'] = tdf.Assignment.str.replace(ph.assignments,'',regex=True)
    tdf['WorkType'] = tdf.WorkType.str.replace('\(Sitting .*$','',regex=True)
    tdf['WorkType'] = tdf.WorkType.str.strip()
    tdf.loc[tdf.WorkType == 'Cap.PCR','WorkType'] = 'Capital PCR'
    tdf['Week'] = tdf.StartDate.apply(lambda x: str(x.isocalendar()[1]) + '-' + str(x.isocalendar()[0]))
    tdf['Date'] = tdf.Date.astype(str)

    tdf['JudgeName'] = tdf['JudgeName'].str.upper()
    tdf['JudgeName'] = tdf['JudgeName'].str.replace('[^A-Za-z\s]','',regex=True)
    tdf['JudgeName'] = tdf['JudgeName'].str.strip()
    tdf.loc[tdf.JudgeName == 'COOPER', 'JudgeName'] = 'COOPER TW'
    judges_to_drop = ['BERGDORF','COUCH','DREW','GIER','PEEPLES','SIMMONS','WATTS','YOUNG']
    tdf = tdf.loc[~tdf.JudgeName.isin(judges_to_drop),:]

    cty = pd.read_csv(county_file)
    tdf['County'] = tdf.Assignment.apply(ph.get_assignment)
    tdf = tdf.merge(cty,on='County',how='left')
    tdf['Circuit'] = tdf.Circuit.astype('Int64').astype(str)
    tdf.loc[tdf.Circuit == '<NA>','Circuit'] = tdf.loc[tdf.Circuit == '<NA>','County'].apply(ph.get_circuit)
    return(tdf)

def get_half(assignment):
    split_assignment = assignment.split('+')
    if len(split_assignment) > 1:
        return '+'.join(split_assignment[1:])
    else:
        return 'na'

def get_monday(day,month,year):
    iso_date = datetime.date(year,month,day).isocalendar()
    year, week, day = iso_date
    monday_date = datetime.date.fromisocalendar(year,week,1)
    return(monday_date.day)

def add_day_weights(cdf):
    day_weights = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    day_weights['Days'] = 1/day_weights['N']
    day_weights.drop(columns=['N'],inplace=True)
    cdf = cdf.merge(day_weights,on=['JudgeName','Date'])
    return cdf

def get_home_county(cdf):
    cdf = cdf.loc[cdf.County != 'na']
    counts = cdf.groupby(['JudgeID','County','Circuit']).size().reset_index(name='N')
    home = counts.sort_values(['JudgeID','N'],ascending=False).groupby('JudgeID').head(1)

def main():
    # df = create_calendar_data(daily=False)
    # df.to_csv(processed_weekly_schedule_file,index=False)
    ddf = create_calendar_data(daily=True)
    ddf.to_csv(processed_daily_schedule_file,index=False)


main()
