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
schedule_data_file = PROJECT_DIR + '/data/processed/weekly_schedule_data.csv'
hester_data_file = PROJECT_DIR + '/data/raw/Sentencing Data/Hester_Data.csv'
expected_min_sentece_file = PROJECT_DIR + '/data/raw/Sentencing Data/expected_min_sentence.dta'
home_circuit_file = PROJECT_DIR + '/data/raw/JudgeNumber_ResidentCircuit.xlsx'

# Output files
processed_sentencing_data_file = PROJECT_DIR + '/data/raw/sentencing_data.csv'
judge_name_id_mapping_file = PROJECT_DIR + '/data/processed/judge_name_id_mapping.csv'

clean_column_names = {'date':'Date','county':'County','circuit':'Circuit','judge':'JudgeID',
'statute':'Statute','offdescr':'OffenseDescription','sgc_offcode':'OffenseCode',
'counts':'Counts','offser':'OffenseSeriousness','sentence':'Sentence','datedisp':'Date',
'realsent':'Sentence','trial':'Trial','incarc':'Incarceration','age':'Age','black':'Black',
'crimhist':'CriminalHistory','expmin':'ExpMinSentence'
}

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
    'Black','EventID','CriminalHistory']
    sdf = sdf[cols_to_keep]
    sdf = add_exp_min_sentence(sdf)
    sdf = add_home_circuit(sdf)
    sdf.loc[sdf.Date.notna(),'Week'] = sdf.loc[sdf.Date.notna(),'Date'].apply(get_week)
    sdf.to_csv('data/processed/temp_sentencing_data.csv',index=False)
    sdf = add_judge_names(sdf)
    sdf['Plea'] = (sdf['Trial']+1)%2
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

def main():
    sdf = clean_sentencing_data()
    sdf.to_csv(processed_sentencing_data_file,index=False)

main()
