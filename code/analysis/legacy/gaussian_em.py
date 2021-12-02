import torch
import pandas as pd
import numpy as np
import os
import datetime
import re
import pickle
import math
from numpy import linalg as LA
from scipy.stats import norm

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
processed_daily_schedule_file = PROJECT_DIR + '/data/processed/daily_schedule_data.csv'
processed_sentencing_data_file = PROJECT_DIR + '/data/raw/sentencing_data.csv'
judge_name_id_mapping_file = PROJECT_DIR + '/data/processed/judge_name_id_mapping.csv'

# Output files
processed_sentencing_data_file = PROJECT_DIR + '/data/processed/sentencing_data.csv'
optimization_data_folder = PROJECT_DIR + '/data/optimization/'

holidays = ['2000-09-04','2000-10-09','2000-11-10','2000-11-23','2000-12-24',
'2000-12-25','2000-12-26','2001-01-01','2001-01-15','2001-02-19','2001-05-10',
'2001-05-28','2001-07-04','2000-07-04']

##### Data loading/cleaning #####
def load_sentencing_data():
    sdf = pd.read_csv(processed_sentencing_data_file)
    sdf = sdf.loc[sdf.Date.notna(),:]
    sdf['Plea'] = (sdf['Trial']+1)%2
    return sdf

def load_calendar_data():
    cdf = pd.read_csv(processed_daily_schedule_file)
    mapping = pd.read_csv(judge_name_id_mapping_file)
    cdf = cdf.merge(mapping,on='JudgeName')
    cdf = cdf.loc[~cdf.Date.isin(holidays),:]
    return cdf

def get_clean_day_pleas():
    sdf = load_sentencing_data()
    sdf = remove_conflicting_days(sdf)
    sdf = remove_non_GS_days(sdf)
    sdf = remove_multi_county_days(sdf)
    sdf = remove_multi_assignment_days(sdf)
    sdf = remove_judges_with_few_clean_days(sdf)
    counts = sdf.groupby(['JudgeID','County','Date'])[['Plea']].sum().reset_index()
    counts.loc[counts.Plea > 17,'Plea'] = 17
    return counts['Plea'].to_numpy()

def remove_conflicting_days(sdf):
    cdf = load_calendar_data()
    fdf = sdf.merge(cdf,on=['Date','JudgeName'])
    fdf['Circuit_y'] = fdf.Circuit_y.replace({'na':np.nan})
    fdf['Circuit_y'] = fdf.Circuit_y.astype('float').astype('Int64')
    fdf.loc[fdf.County_x != fdf.County_y,'MatchType'] = 'Disagree'
    fdf.loc[fdf.Circuit_x == fdf.Circuit_y,'MatchType'] = 'Circuit'
    fdf.loc[fdf.County_x == fdf.County_y,'MatchType'] = 'Agree'
    fdf.sort_values('MatchType',inplace=True)
    fdf = fdf.drop_duplicates('EventID',keep='first')
    fdf = fdf.loc[fdf.MatchType == 'Disagree',['Date','JudgeName','MatchType']].drop_duplicates(['Date','JudgeName'])
    sdf = sdf.merge(fdf,on=['Date','JudgeName'],how='left')
    sdf = sdf.loc[sdf.MatchType.isna(),:]
    return sdf

def remove_non_GS_days(sdf):
    cdf = load_calendar_data()
    gs_days = cdf.loc[cdf.Assignment.str.contains('GS$',regex=True),['JudgeName','Date']]
    sdf = sdf.merge(gs_days,on=['JudgeName','Date'])
    return sdf

def remove_multi_county_days(sdf):
    judge_events = sdf.groupby(['JudgeID','Date','County']).size().reset_index(name='N')
    counties_by_day = judge_events.groupby(['JudgeID','Date']).size().reset_index(name='N')
    single_county_days = counties_by_day.loc[counties_by_day.N == 1,['JudgeID','Date']]
    sdf = sdf.merge(single_county_days,on=['JudgeID','Date'])
    return sdf

def remove_multi_assignment_days(sdf):
    cdf = load_calendar_data()
    day_assignments = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    single_assignment_days = day_assignments.loc[day_assignments.N == 1,['JudgeName','Date']]
    sdf = sdf.merge(single_assignment_days,on=['JudgeName','Date'])
    return sdf

def remove_judges_with_few_clean_days(sdf,threshold=10):
    judge_clean_days = sdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    good_judges = judge_clean_days.loc[judge_clean_days.N >= threshold,['JudgeName','Date']]
    sdf = sdf.merge(good_judges,on=['JudgeName','Date'])
    return sdf

def get_judge_days():
    cdf = load_calendar_data()
    day_weights = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    day_weights['Days'] = 1/day_weights['N']
    cdf = cdf.merge(day_weights,on=['JudgeName','Date'])
    judge_days = cdf['Days'].sum()
    return judge_days

##### Estimation/Optimization #####
def gaussian_em(init_mu_x,init_sigma_x,init_mu_d,init_sigma_d,tolerance=0.05):
    pleas = get_clean_day_pleas()

    prev_mu_x, prev_sigma_x = (0,0)
    prev_mu_d, prev_sigma_d = (0,0)

    mu_x, sigma_x = (init_mu_x, init_sigma_x)
    mu_d, sigma_d = (init_mu_d, init_sigma_d)
    prev_theta = np.array([prev_mu_x,prev_sigma_x,prev_mu_d,prev_sigma_d])
    theta = np.array([mu_x,sigma_x,mu_d,sigma_d])

    iters = 1
    vals = []
    while (LA.norm(theta-prev_theta,np.inf) > tolerance):
        prev_mu_x, prev_sigma_x = (mu_x, sigma_x)
        prev_mu_d, prev_sigma_d = (mu_d, sigma_d)
        prev_theta = np.array([prev_mu_x,prev_sigma_x,prev_mu_d,prev_sigma_d])

        mu_x = update_mu(prev_mu_x, prev_sigma_x, pleas)
        sigma_x = update_sigma(prev_mu_x, prev_sigma_x, pleas)

        mu_d = update_mu(prev_mu_d, prev_sigma_d, pleas)
        sigma_d = update_sigma(prev_mu_d, prev_sigma_d, pleas)

        theta = np.array([mu_x,sigma_x,mu_d,sigma_d])

        x_1 = [calculate_x_1(prev_mu_x, prev_sigma_x, s) for s in pleas]
        x_2 = [calculate_x_2(prev_mu_x, prev_sigma_x, s) for s in pleas]
        d_1 = [calculate_x_1(prev_mu_d, prev_sigma_d, s) for s in pleas]
        d_2 = [calculate_x_2(prev_mu_d, prev_sigma_d, s) for s in pleas]
        vals.append({'X1':np.mean(x_1),'X2':np.mean(x_2),'D1':np.mean(d_1),
                    'D2':np.mean(d_2),'Iteration':iters})
        iters += 1
        print(theta)
        if iters == 20:
            break
    data = pd.DataFrame(vals)
    filename = optimization_data_folder + 'gaussian_em_data.csv'
    data.to_csv(filename,index=False)
    return theta

def update_mu(mu,sigma,pleas):
    n = len(pleas)
    x_1 = [calculate_x_1(mu, sigma, s) for s in pleas]
    summation = (x_1 + pleas).sum()
    new_mu = summation/(2*n)
    return new_mu

def update_sigma(mu,sigma,pleas):
    n = len(pleas)
    x_2 = [calculate_x_2(mu, sigma, s) for s in pleas]
    x_1 = [calculate_x_1(mu, sigma, s) for s in pleas]
    first_term = (x_2 + pleas**2).sum()/(2*n)
    second_term = ((x_1 + pleas).sum()/(2*n))**2
    new_sigma = np.sqrt(first_term-second_term)
    return new_sigma

def calculate_x_1(mu,sigma,s):
    y = (s-mu)/sigma
    first_term = sigma/(1-norm.cdf(y))
    second_term = norm.pdf(y)
    return mu + first_term*second_term

def calculate_x_2(mu,sigma,s):
    y = (s-mu)/sigma
    first_term = ((mu*sigma)/(1-norm.cdf(y)))*norm.pdf(y)
    second_term = ((sigma**2)/(1-norm.cdf(y)))*norm.pdf(y)*s
    return mu**2 + sigma**2 + first_term + second_term

def estimate_mu_t(mu_p):
    counts = get_judge_county_event_counts()
    assigned_days = get_day_assignments()
    df = counts.merge(assigned_days,on=['JudgeName','County'])
    df['PleaDays'] = df['Plea']/mu_p
    df = df[df.Trial >=2]
    mu_t = df.Trial.sum()/(df.Days.sum()-df.PleaDays.sum())
    return mu_t

def make_mu_t_data(mu_p):
    counts = get_judge_county_event_counts()
    assigned_days = get_day_assignments()
    df = counts.merge(assigned_days,on=['JudgeName','County'])
    df['PleaDays'] = df['Plea']/mu_p
    df['TrialDays'] = df['Days']-df['PleaDays']
    df['MuT'] = df['Trial']/df['TrialDays']
    return df

def get_judge_county_event_counts():
    sdf = load_sentencing_data()
    sdf = trial_capacity_sample(sdf)
    counts = sdf.groupby(['JudgeName','JudgeID','County'])[['Plea','Trial']].sum().reset_index()
    return counts

def trial_capacity_sample(sdf):
    daily_sentences = sdf.groupby(['Date','JudgeName','County']).size().reset_index(name="N")
    outlier_jcs = daily_sentences.loc[daily_sentences['N'] >= 35,['JudgeName','County']].drop_duplicates()
    outlier_jcs['Outlier'] = True
    trial_jcs = sdf.loc[sdf.Trial == 1,['JudgeName','County']].drop_duplicates()

    sample = trial_jcs.merge(outlier_jcs,on=['JudgeName','County'],how='left')
    sample = sample.loc[~(sample.Outlier == True),['JudgeName','County']].drop_duplicates()

    sdf = sdf.merge(sample,on=['JudgeName','County'])
    return sdf

def get_day_assignments():
    cdf = load_calendar_data()
    day_weights = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    day_weights['Days'] = 1/day_weights['N']
    cdf = cdf.merge(day_weights,on=['JudgeName','Date'])
    assigned_days = cdf.groupby(['JudgeName','County'])['Days'].sum().reset_index()
    return assigned_days

def ad_hoc_algorithm(init_mu_t,init_mu_p,tolerance=0.05,opt_iter=100):
    sdf = load_sentencing_data()
    pleas = get_clean_day_pleas()
    judge_days = get_judge_days()
    prev_mu_t = 0
    prev_mu_p = 0
    mu_t = init_mu_t
    mu_p = init_mu_p
    iters = 0
    while (abs(mu_t - prev_mu_t) > tolerance) or (abs(mu_p - prev_mu_p) > tolerance):
        prev_mu_p = mu_p
        prev_mu_t = mu_t
        mu_p = optimize_mu_p(prev_mu_t,prev_mu_p,sdf,judge_days,pleas,opt_iter,0.02)
        mu_t = estimate_mu_t(mu_p)

        NLL_data = make_NLL_data(prev_mu_t,mu_p)
        NLL_filename = optimization_data_folder + 'opt_data_{}.csv'.format(iters)
        NLL_data.to_csv(NLL_filename,index=False)

        mu_t_data = make_mu_t_data(mu_p)
        mu_t_filename = optimization_data_folder + 'mut_data_{}.csv'.format(iters)
        mu_t_data.to_csv(mu_t_filename,index=False)

        iters += 1
        print('mu_p: {}, mu_t: {}\n'.format(mu_p,mu_t))
    return({'mu_p':mu_p,'mu_t':mu_t})
