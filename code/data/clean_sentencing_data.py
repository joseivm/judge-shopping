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
processed_daily_schedule_file = PROJECT_DIR + '/data/processed/daily_schedule_data.csv'
schedule_data_file = PROJECT_DIR + '/data/processed/weekly_schedule_data.csv'
hester_data_file = PROJECT_DIR + '/data/raw/Sentencing Data/Hester_Data.csv'
expected_min_sentece_file = PROJECT_DIR + '/data/raw/Sentencing Data/expected_min_sentence.dta'
home_circuit_file = PROJECT_DIR + '/data/raw/JudgeNumber_ResidentCircuit.xlsx'

# Output files
processed_sentencing_data_file = PROJECT_DIR + '/data/processed/sentencing_data.csv'
judge_name_id_mapping_file = PROJECT_DIR + '/data/processed/judge_name_id_mapping.csv'

clean_column_names = {'date':'Date','county':'County','circuit':'Circuit','judge':'JudgeID',
'statute':'Statute','offdescr':'OffenseDescription','sgc_offcode':'OffenseCode',
'counts':'Counts','offser':'OffenseSeriousness','sentence':'Sentence','datedisp':'Date',
'realsent':'Sentence','trial':'Trial','incarc':'Incarceration','age':'Age','black':'Black',
'crimhist':'CriminalHistory','expmin':'ExpMinSentence'
}
holidays = ['2000-09-04','2000-10-09','2000-11-10','2000-11-23','2000-12-24',
'2000-12-25','2000-12-26','2001-01-01','2001-01-15','2001-02-19','2001-05-10',
'2001-05-28','2001-07-04','2000-07-04']

##### Cleaning Functions #####
def clean_sentencing_data():
    sdf = pd.read_csv(hester_data_file)
    sdf.rename(columns=clean_column_names,inplace=True)
    sdf = sdf.loc[sdf.JudgeID != 1,:]
    sdf['JudgeID'] = sdf['JudgeID'].apply(lambda x: 'Judge '+str(x-1))
    sdf['Incarceration'] = sdf['Incarceration'].replace({'yes':1,'no':0})
    sdf['Date'] = pd.to_datetime(sdf['Date'],format="%m/%d/%y")
    sdf['Sentence'] = sdf['Sentence'].round(1).astype('float32')
    sdf = add_offense_type(sdf)
    sdf = clean_statutes(sdf)
    sdf = add_sex(sdf)
    sdf = sdf.reset_index().rename(columns={'index':'EventID'})

    cols_to_keep = ['Date','County','Circuit','Counts','OffenseSeriousness','OffenseCode',
    'OffenseType','Sentence','Statute','JudgeID','Trial','Incarceration','Age',
    'Black','EventID','CriminalHistory','Sex']
    sdf = sdf[cols_to_keep]
    sdf = add_exp_min_sentence(sdf)
    sdf = add_home_circuit(sdf)
    sdf['Circuit'] = sdf.Circuit.astype(str)
    sdf = add_judge_names(sdf)
    sdf['Plea'] = (sdf['Trial']+1)%2
    sdf = impute_missing_dates(sdf)
    sdf.loc[sdf.Date.notna(),'Week'] = sdf.loc[sdf.Date.notna(),'Date'].apply(get_week)
    sdf = add_work_type(sdf)
    return(sdf)

def add_judge_names(sdf):
    cdf = pd.read_csv(schedule_data_file)
    counties = '|'.join(sdf.County.unique())
    cdf['County'] = cdf['Assignment'].apply(find_string,args=(counties,))
    cdf.to_csv('data/processed/temp_schedule_data.csv',index=False)

    merge_cols = ['County','Week']
    overlap = pd.merge(sdf[['JudgeID','County','Week']],cdf[['JudgeName','County','Week']],on=merge_cols)
    counts = overlap.groupby(['JudgeName','JudgeID']).size().reset_index(name='N')

    max_indices = counts.groupby('JudgeID')['N'].transform(max) == counts['N']
    mapping = counts.loc[max_indices,['JudgeName','JudgeID']]
    mapping.to_csv(judge_name_id_mapping_file,index=False)
    sdf = sdf.merge(mapping,on='JudgeID',how='left')
    return(sdf)

def find_string(string,regex):
    match = re.search(regex,string,flags=re.IGNORECASE)
    if match is not None:
        return match.group(0)
    else:
        return 'na'

def get_week(date):
    week = date.isocalendar()[1]
    year = date.isocalendar()[0]
    return(str(week) + '-' +str(year))

def add_exp_min_sentence(df):
    merge_cols = ['Date','County','Circuit','Counts','OffenseSeriousness','OffenseCode',
    'OffenseType','Sentence','Statute']
    edf = clean_exp_min_data()
    edf.drop(columns=['jud_no'],inplace=True)
    df = pd.merge(df,edf,on=merge_cols,how='left')
    df.drop_duplicates(inplace=True)
    return(df)

def clean_exp_min_data():
    edf = pd.read_stata(expected_min_sentece_file)
    edf = add_offense_type(edf)
    edf.rename(columns=clean_column_names,inplace=True)
    edf['Sentence'] = edf['Sentence'].round(1).astype('float32')
    edf.loc[edf.Statute == '','Statute'] = np.nan
    cols = ['Date','County','Circuit','Counts','OffenseSeriousness','OffenseCode',
    'OffenseType','Sentence','Statute','ExpMinSentence','jud_no']
    edf.drop_duplicates(inplace=True)
    return(edf[cols])

def add_home_circuit(df):
    merge_cols = ['Date','County','Circuit','Counts','OffenseSeriousness','OffenseCode',
    'OffenseType','Sentence','Statute']
    edf = clean_exp_min_data()
    mdf = pd.merge(df,edf,on=merge_cols,how='left')
    mapping = mdf.groupby(['JudgeID','jud_no']).size().reset_index(name='N').sort_values('N',ascending=False).groupby('JudgeID').head(1)
    df = df.merge(mapping[['JudgeID','jud_no']],on='JudgeID')

    home_circuits = pd.read_excel(home_circuit_file)
    home_circuits.rename(columns={'JudgeID':'jud_no'},inplace=True)
    df = df.merge(home_circuits[['jud_no','HomeCircuit']],on='jud_no')
    df.drop(columns=['jud_no'],inplace=True)
    return df

def add_offense_type(df):
    df['OffenseType'] = "Other"
    df.loc[df.of_hom == 1, 'OffenseType'] = 'Homicide'
    df.loc[df.of_rape == 1, 'OffenseType'] = 'Rape'
    df.loc[df.of_rob == 1, 'OffenseType'] = 'Robbery'
    df.loc[df.of_asslt == 1, 'OffenseType'] = 'Assault'
    df.loc[df.of_burg == 1, 'OffenseType'] = 'Burglary'
    df.loc[df.of_dstrb == 1, 'OffenseType'] = 'Drug Distribution'
    df.loc[df.of_possn == 1, 'OffenseType'] = 'Drug Possesion'
    df.loc[df.of_theft == 1, 'OffenseType'] = 'Theft'
    df.loc[df.of_fraud == 1, 'OffenseType'] = 'Fraud'
    df.loc[df.of_other == 1, 'OffenseType'] = 'Other'
    return(df)

def clean_statutes(df):
    df.loc[df.Statute == '12/21/90','Statute'] = '12-21-2790'
    df.loc[df.Statute == '12/21/92','Statute'] = '12-21-2792'
    return(df)

def add_sex(df):
    df['Sex'] = 'Missing'
    df.loc[df.male == 1, 'Sex'] = 'Male'
    df.loc[df.male == 0, 'Sex'] = 'Female'
    return(df)

###### Imputation #####
def impute_missing_dates(sdf):
    sdf = impute_missing_dates_county_GS(sdf)
    sdf = impute_missing_dates_county_nonGS(sdf)
    sdf = impute_missing_dates_circuit_GS(sdf)
    sdf = impute_missing_dates_circuit_nonGS(sdf)
    sdf = impute_missing_dates_based_on_sentencing(sdf)
    sdf = impute_missing_dates_non_matching(sdf)
    return sdf

def impute_missing_dates_county_GS(sdf):
    cdf = load_calendar_data()

    pdf = cdf.merge(sdf[['EventID','JudgeID','County','Date']],on=['JudgeID','County','Date'],how='left')
    pdf = pdf.loc[(pdf.EventID.notna()) | (pdf.WorkType == 'GS'),:]

    mdf = sdf.loc[sdf.Date.isna(),['JudgeID','County','EventID']]
    mdf = mdf.merge(pdf[['JudgeID','County']],on=['JudgeID','County'])
    mdf = mdf.drop_duplicates('EventID')

    groups = mdf.groupby(['JudgeID','County'])
    imputed_observations = []
    for name, group in groups:
        judge_id, county = name
        event_ids = group['EventID'].to_numpy()
        possible_dates = pdf.loc[(pdf.JudgeID == judge_id) & (pdf.County == county),'Date'].unique()
        assignments = np.array_split(event_ids,len(possible_dates))
        for event_ids, date in zip(assignments,possible_dates):
            obs = [{'EventID':event_id,'ImputedDate':date,'Imputation':'CountyGS'} for event_id in event_ids]
            imputed_observations += obs

    idf = pd.DataFrame(imputed_observations)
    sdf = sdf.merge(idf,on=['EventID'],how='left')
    sdf.loc[sdf.Date.notna(),'Imputation'] = 'None'
    sdf.loc[sdf.Imputation == 'CountyGS','Date'] = sdf.loc[sdf.Imputation == 'CountyGS','ImputedDate']
    return sdf

def impute_missing_dates_county_nonGS(sdf):
    cdf = load_calendar_data()

    pdf = cdf.merge(sdf[['EventID','JudgeID','County','Date']],on=['JudgeID','County','Date'],how='left')

    mdf = sdf.loc[sdf.Date.isna(),['JudgeID','County','EventID']]
    mdf = mdf.merge(pdf[['JudgeID','County']],on=['JudgeID','County'])
    mdf = mdf.drop_duplicates('EventID')

    groups = mdf.groupby(['JudgeID','County'])
    imputed_observations = []
    for name, group in groups:
        judge_id, county = name
        event_ids = group['EventID'].to_numpy()
        possible_dates = pdf.loc[(pdf.JudgeID == judge_id) & (pdf.County == county),'Date'].unique()
        assignments = np.array_split(event_ids,len(possible_dates))
        for event_ids, date in zip(assignments,possible_dates):
            obs = [{'EventID':event_id,'ImputedDateNonGS':date} for event_id in event_ids]
            imputed_observations += obs

    idf = pd.DataFrame(imputed_observations)
    sdf = sdf.merge(idf,on=['EventID'],how='left')
    sdf.loc[sdf.ImputedDateNonGS.notna(), 'Imputation'] = 'CountyNonGS'
    sdf.loc[sdf.Imputation == 'CountyNonGS','Date'] = sdf.loc[sdf.Imputation == 'CountyNonGS','ImputedDateNonGS']
    sdf.loc[sdf.Imputation == 'CountyNonGS','ImputedDate'] = sdf.loc[sdf.Imputation == 'CountyNonGS','ImputedDateNonGS']
    sdf.drop(columns=['ImputedDateNonGS'],inplace=True)
    return sdf

def impute_missing_dates_circuit_GS(sdf):
    cdf = load_calendar_data()

    pdf = cdf.merge(sdf[['EventID','JudgeID','Circuit','Date']],on=['JudgeID','Circuit','Date'],how='left')
    pdf = pdf.loc[((pdf.EventID.notna()) | (pdf.WorkType == 'GS')) |
                   (pdf.Assignment.str.contains('Cir',na=False) & pdf.WorkType.str.contains('GS',na=False)),:]

    mdf = sdf.loc[sdf.Date.isna(),['JudgeID','Circuit','EventID']]
    mdf = mdf.merge(pdf[['JudgeID','Circuit']],on=['JudgeID','Circuit'])
    mdf = mdf.drop_duplicates('EventID')

    groups = mdf.groupby(['JudgeID','Circuit'])
    imputed_observations = []
    for name, group in groups:
        judge_id, circuit = name
        event_ids = group['EventID']
        possible_dates = pdf.loc[(pdf.JudgeID == judge_id) & (pdf.Circuit == circuit),'Date'].unique()
        assignments = np.array_split(event_ids,len(possible_dates))
        for event_ids, date in zip(assignments,possible_dates):
            events = [{'EventID':event_id,'TempImputedDate':date} for event_id in event_ids]
            imputed_observations += events

    idf = pd.DataFrame(imputed_observations)
    sdf = sdf.merge(idf,on=['EventID'],how='left')
    sdf.loc[sdf.TempImputedDate.notna(), 'Imputation'] = 'CircuitGS'
    sdf.loc[sdf.Imputation == 'CircuitGS','Date'] = sdf.loc[sdf.Imputation == 'CircuitGS','TempImputedDate']
    sdf.loc[sdf.Imputation == 'CircuitGS','ImputedDate'] = sdf.loc[sdf.Imputation == 'CircuitGS','TempImputedDate']
    sdf.drop(columns=['TempImputedDate'],inplace=True)
    return sdf

def impute_missing_dates_circuit_nonGS(sdf):
    cdf = load_calendar_data()

    pdf = cdf.merge(sdf[['EventID','JudgeID','Circuit','Date']],on=['JudgeID','Circuit','Date'],how='left')

    mdf = sdf.loc[sdf.Date.isna(),['JudgeID','Circuit','EventID']]
    mdf = mdf.merge(pdf[['JudgeID','Circuit']],on=['JudgeID','Circuit'])
    mdf = mdf.drop_duplicates('EventID')

    groups = mdf.groupby(['JudgeID','Circuit'])
    imputed_observations = []
    for name, group in groups:
        judge_id, circuit = name
        event_ids = group['EventID']
        possible_dates = pdf.loc[(pdf.JudgeID == judge_id) & (pdf.Circuit == circuit),'Date'].unique()
        assignments = np.array_split(event_ids,len(possible_dates))
        for event_ids, date in zip(assignments,possible_dates):
            events = [{'EventID':event_id,'TempImputedDate':date} for event_id in event_ids]
            imputed_observations += events

    idf = pd.DataFrame(imputed_observations)
    sdf = sdf.merge(idf,on=['EventID'],how='left')
    sdf.loc[sdf.TempImputedDate.notna(), 'Imputation'] = 'CircuitNonGS'
    sdf.loc[sdf.Imputation == 'CircuitNonGS','Date'] = sdf.loc[sdf.Imputation == 'CircuitNonGS','TempImputedDate']
    sdf.loc[sdf.Imputation == 'CircuitNonGS','ImputedDate'] = sdf.loc[sdf.Imputation == 'CircuitNonGS','TempImputedDate']
    sdf.drop(columns=['TempImputedDate'],inplace=True)
    return sdf

def impute_missing_dates_based_on_sentencing(sdf):
    cdf = load_calendar_data()

    pdf = sdf.loc[sdf.Imputation == 'None',['JudgeID','County','Date']].drop_duplicates()

    mdf = sdf.loc[sdf.Date.isna(),['JudgeID','County','EventID']]
    mdf = mdf.merge(pdf[['JudgeID','County']],on=['JudgeID','County'])
    mdf = mdf.drop_duplicates('EventID')

    groups = mdf.groupby(['JudgeID','County'])
    imputed_observations = []
    for name, group in groups:
        judge_id, county = name
        event_ids = group['EventID'].to_numpy()
        possible_dates = pdf.loc[(pdf.JudgeID == judge_id) & (pdf.County == county),'Date'].unique()
        assignments = np.array_split(event_ids,len(possible_dates))
        for event_ids, date in zip(assignments,possible_dates):
            obs = [{'EventID':event_id,'TempImputedDate':date} for event_id in event_ids]
            imputed_observations += obs

    idf = pd.DataFrame(imputed_observations)
    sdf = sdf.merge(idf,on=['EventID'],how='left')
    sdf.loc[sdf.TempImputedDate.notna(), 'Imputation'] = 'SentencingData'
    sdf.loc[sdf.Imputation == 'SentencingData','Date'] = sdf.loc[sdf.Imputation == 'SentencingData','TempImputedDate']
    sdf.loc[sdf.Imputation == 'SentencingData','ImputedDate'] = sdf.loc[sdf.Imputation == 'SentencingData','TempImputedDate']
    sdf.drop(columns=['TempImputedDate'],inplace=True)
    return sdf

def impute_missing_dates_non_matching(sdf):
    cdf = load_calendar_data()

    pdf = cdf.merge(sdf[['EventID','JudgeID','County','Date']],on=['JudgeID','County','Date'],how='left')
    pdf = pdf.loc[(pdf.EventID.notna()) | (pdf.WorkType == 'GS'),:]

    mdf = sdf.loc[sdf.Date.isna(),['JudgeID','EventID']]
    mdf = mdf.merge(pdf[['JudgeID']],on=['JudgeID'])
    mdf = mdf.drop_duplicates('EventID')

    groups = mdf.groupby(['JudgeID'])
    imputed_observations = []
    for judge_id, group in groups:
        event_ids = group['EventID'].to_numpy()
        possible_dates = pdf.loc[(pdf.JudgeID == judge_id),'Date'].unique()
        assignments = np.array_split(event_ids,len(possible_dates))
        for event_ids, date in zip(assignments,possible_dates):
            obs = [{'EventID':event_id,'TempImputedDate':date} for event_id in event_ids]
            imputed_observations += obs

    idf = pd.DataFrame(imputed_observations)
    sdf = sdf.merge(idf,on=['EventID'],how='left')
    sdf.loc[sdf.TempImputedDate.notna(),'Imputation'] = 'NonMatching'
    sdf.loc[sdf.Imputation == 'NonMatching','Date'] = sdf.loc[sdf.Imputation == 'NonMatching','TempImputedDate']
    sdf.loc[sdf.Imputation == 'NonMatching','ImputedDate'] = sdf.loc[sdf.Imputation == 'NonMatching','TempImputedDate']
    sdf.drop(columns=['TempImputedDate'],inplace=True)
    return sdf

def add_work_type(sdf):
    cdf = load_calendar_data()
    cdf_cols = ['JudgeID','Date','County','WorkType']
    sdf = sdf.merge(cdf[cdf_cols],on=['JudgeID','Date','County'],how='left')
    sdf.loc[sdf.WorkType.isna(),'WorkType'] = 'Disagree'
    return sdf

def load_calendar_data():
    cdf = pd.read_csv(processed_daily_schedule_file)
    mapping = pd.read_csv(judge_name_id_mapping_file)
    cdf = cdf.merge(mapping,on='JudgeName')
    cdf = cdf.loc[~cdf.Date.isin(holidays),:]
    cdf['Date'] = pd.to_datetime(cdf['Date'])
    return cdf

def main():
    sdf = clean_sentencing_data()
    sdf.to_csv(processed_sentencing_data_file,index=False)

main()
