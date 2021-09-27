import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from patsy import dmatrices
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
processed_daily_schedule_file = PROJECT_DIR + '/data/processed/daily_schedule_data.csv'
processed_sentencing_data_file = PROJECT_DIR + '/data/processed/sentencing_data.csv'
judge_name_id_mapping_file = PROJECT_DIR + '/data/processed/judge_name_id_mapping.csv'
county_file = PROJECT_DIR + '/data/raw/county_list.csv'

# Output files
FIGURE_DIR = PROJECT_DIR + '/output/figures/Exploration/'
TABLE_DIR = PROJECT_DIR + '/output/tables/Exploration/'

holidays = ['2000-09-04','2000-10-09','2000-11-10','2000-11-23','2000-12-24',
'2000-12-25','2000-12-26','2001-01-01','2001-01-15','2001-02-19','2001-05-10',
'2001-05-28','2001-07-04','2000-07-04']


##### Data loading/imputation #####
def load_sentencing_data():
    sdf = pd.read_csv(processed_sentencing_data_file)
    sdf['Circuit'] = sdf.Circuit.astype(str)
    return sdf

def load_calendar_data():
    cdf = pd.read_csv(processed_daily_schedule_file)
    mapping = pd.read_csv(judge_name_id_mapping_file)
    cdf = cdf.merge(mapping,on='JudgeName')
    cdf = cdf.loc[~cdf.Date.isin(holidays),:]
    return cdf

def make_regression_data(group):
    # group = ['JudgeID','County']
    sdf = load_sentencing_data()
    sdf = sdf.loc[sdf.WorkType == 'GS',:]
    counts = sdf.groupby(group)[['Plea','Trial']].sum().reset_index()

    cdf = load_calendar_data()
    cdf = cdf.loc[cdf.WorkType == 'GS',:]
    days = cdf.groupby(group)['Days'].sum().reset_index()
    counts = counts.merge(days,on=group)
    # counts = counts.set_index(group)
    return(counts)

def make_nlopt_data(group):
    df = make_regression_data(['JudgeID','County'])
    encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
    dummy_matrix = encoder.fit_transform(df[group].to_numpy().reshape(-1,1))

    days = df['Days'].to_numpy()
    day_matrix = dummy_matrix * -days[:, np.newaxis]
    data = np.hstack((day_matrix,df[['Plea','Trial']].to_numpy()))
    return data, encoder

def obj(x,A):
    result = A.dot(x)
    return np.square(result).sum()

def nl_reg(group,lb):
    data, encoder = make_nlopt_data(group)
    ncols = data.shape[1]
    bounds = [(lb,1) for i in range(ncols-2)]
    bounds += [(0,None),(0,None)]

    x0 = np.ones(ncols)*0.85
    x0[-2] = 0.1
    x0[-1] = 5

    res = minimize(obj,x0,args=(data),bounds=bounds)
    param_names = np.append(encoder.categories_[0],['BetaP','BetaT'])
    df = pd.DataFrame({str(lb):res.x,'Parameter':param_names})
    return df

for group in ['JudgeID','County']:
    dfs = []
    lbs = [0,0.5,0.6,0.65,0.7,0.75]
    for lb in lbs:
        tdf = nl_reg(group,lb)
        dfs.append(tdf)

    rdf = reduce(lambda x,y: pd.merge(x,y,on='Parameter'),dfs)
    cols = ['Parameter'] + [str(lb) for lb in lbs]
    filename = TABLE_DIR + 'nlreg_{}.tex'.format(group)
    rdf[cols].to_latex(filename,index=False,float_format='%.2f')
