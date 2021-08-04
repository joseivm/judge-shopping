import torch
import pandas as pd
import numpy as np
import os
import datetime
import re
import pickle
import math

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# Input files/dirs
processed_daily_schedule_file = PROJECT_DIR + '/data/processed/daily_schedule_data.csv'
processed_sentencing_data_file = PROJECT_DIR + '/data/raw/sentencing_data.csv'
judge_name_id_mapping_file = PROJECT_DIR + '/data/processed/judge_name_id_mapping.csv'

# Output files
optimization_data_folder = PROJECT_DIR + '/data/optimization/'

holidays = ['2000-09-04','2000-10-09','2000-11-10','2000-11-23','2000-12-24',
'2000-12-25','2000-12-26','2001-01-01','2001-01-15','2001-02-19','2001-05-10',
'2001-05-28','2001-07-04','2000-07-04']

##### Data Loading/Cleaning #####
def load_sentencing_data(impute=False):
    sdf = pd.read_csv(processed_sentencing_data_file)
    sdf['Circuit'] = sdf.Circuit.astype(str)
    if impute:
        sdf = impute_missing_dates(sdf)
        sdf = add_work_type(sdf)
    else:
        sdf = sdf.loc[sdf.Date.notna(),:]
    return sdf

def load_calendar_data():
    cdf = pd.read_csv(processed_daily_schedule_file)
    mapping = pd.read_csv(judge_name_id_mapping_file)
    cdf = cdf.merge(mapping,on='JudgeName')
    cdf = cdf.loc[~cdf.Date.isin(holidays),:]
    return cdf

def impute_missing_dates(sdf):
    sdf = impute_missing_dates_county_GS(sdf)
    sdf = impute_missing_dates_county_nonGS(sdf)
    sdf = impute_missing_dates_circuit_GS(sdf)
    sdf = impute_missing_dates_circuit_nonGS(sdf)
    sdf = impute_missing_dates_non_matching(sdf)
    return sdf

def impute_missing_dates_county_GS(sdf):
    cdf = load_calendar_data()
    # cdf['Date'] = pd.to_datetime(cdf['Date'])

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
    # cdf['Date'] = pd.to_datetime(cdf['Date'])

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
    # cdf['Date'] = pd.to_datetime(cdf['Date'])

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
    # cdf['Date'] = pd.to_datetime(cdf['Date'])

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

def impute_missing_dates_non_matching(sdf):
    cdf = load_calendar_data()
    # cdf['Date'] = pd.to_datetime(cdf['Date'])

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

##### Clean Day Filtering #####
def get_clean_day_pleas(impute=False):
    sdf = load_sentencing_data(impute)
    sdf = remove_conflicting_days(sdf)
    sdf = remove_non_GS_days(sdf)
    sdf = remove_multi_county_days(sdf)
    sdf = remove_multi_assignment_days(sdf)
    sdf = remove_judges_with_few_clean_days(sdf)
    counts = sdf.groupby(['JudgeID','County','Date'])[['Plea']].sum().reset_index()
    counts.loc[counts.Plea > 17,'Plea'] = 17
    return counts['Plea']

def remove_non_missing_judges(sdf):
    all_judges = sdf.JudgeID.unique()
    missing_judges = sdf.loc[sdf.Date.isna(),'JudgeID'].unique()
    sdf = sdf.loc[~sdf.JudgeID.isin(missing_judges),:]
    return sdf

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
    gs_days = cdf.loc[cdf.WorkType == 'GS',['JudgeName','Date']]
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

##### Samples #####
def get_judge_days(GS=False):
    cdf = load_calendar_data()
    if GS:
        cdf = cdf.loc[cdf.WorkType == 'GS',:]
    judge_days = cdf['Days'].sum()
    return judge_days

def get_judge_county_event_counts(GS=False):
    sdf = load_sentencing_data()
    sdf = trial_capacity_sample(sdf)
    if GS:
        sdf = sdf.loc[sdf.WorkType == 'GS',:]
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

def get_day_assignments(GS=False):
    cdf = load_calendar_data()
    if GS:
        cdf = cdf.loc[cdf.WorkType == 'GS',:]
    assigned_days = cdf.groupby(['JudgeName','County'])['Days'].sum().reset_index()
    return assigned_days

##### Estimation/Optimization Functions #####
def calculate_NLL(pleas,theta,mu_p):
    NLL = 0
    for s in pleas:
        theta_term = (theta**s)*(1/math.factorial(s))
        theta_sum = 1
        for i in range(0,s):
            theta_sum -= math.pow(mu_p,i)*math.exp(-mu_p)/math.factorial(i)

        mu_term = math.pow(mu_p,s)*math.exp(-mu_p)/math.factorial(s)
        mu_sum = 1
        for j in range(0,s+1):
            mu_sum -= (theta**j)*math.exp(-theta)/math.factorial(j)

        NLL += -math.log(theta_term*theta_sum+mu_term*mu_sum)
    return NLL

def make_NLL_data(mu_t,opt_mu_p):
    sdf = load_sentencing_data()
    pleas = get_clean_day_pleas()
    judge_days = get_judge_days()
    num_trials = sdf['Trial'].sum()
    trial_days = num_trials/mu_t
    plea_days = judge_days - trial_days
    theta = sdf['Plea'].sum()/plea_days
    NLLS = []
    mu_ps = np.linspace(9,14,num=200)
    for mu_p in mu_ps:
        NLL = calculate_NLL(pleas,theta,mu_p)
        NLLS.append(NLL)

    brute_force_min = mu_ps[np.argmin(NLLS)]
    data = pd.DataFrame({'MuP':mu_ps,'NLL':NLLS,'MuT':mu_t,'OptMuP':opt_mu_p,'BFMuP':brute_force_min})
    return data

def optimize_mu_p(mu_t,mu_p,sdf,judge_days,pleas,iters=100,lr=0.02,GS=False):
    if GS:
        sdf = sdf.loc[sdf.WorkType == 'GS']
    num_trials = sdf['Trial'].sum()
    trial_days = num_trials/mu_t
    plea_days = judge_days - trial_days
    theta = sdf['Plea'].sum()/plea_days
    mu_p = torch.tensor([mu_p],requires_grad=True)
    print("Trials: {}, Trial Days: {}, Pleas: {}, Plea Days: {}, Theta: {}".format(num_trials,trial_days,sdf.Plea.sum(),plea_days,theta))
    optimizer = torch.optim.AdamW([mu_p],lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,20,0.01)
    min_grad = math.inf
    min_NLL = math.inf
    best_mu = 0
    for k in range(iters):
        optimizer.zero_grad()
        NLL = 0
        for s in pleas:
            theta_term = (theta**s)*(1/math.factorial(s))
            theta_sum = 1
            for i in range(1,s):
                theta_sum -= torch.pow(mu_p,i)*torch.exp(-mu_p)/math.factorial(i)

            mu_term = torch.pow(mu_p,s)*torch.exp(-mu_p)/math.factorial(s)
            mu_sum = 1
            for j in range(1,s+1):
                mu_sum -= (theta**j)*math.exp(-theta)/math.factorial(j)

            NLL += -torch.log(theta_term*theta_sum+mu_term*mu_sum)
        NLL.backward()
        optimizer.step()
        # scheduler.step()
        if NLL <= min_NLL:
            best_mu = mu_p.detach().clone()
            min_NLL = NLL
            min_grad = mu_p.grad.abs()

    # return {'mu_p':best_mu.item(),'grad':min_grad,'k':k}
    print(min_grad)
    return best_mu.item()

def estimate_mu_t(mu_p,GS=False):
    counts = get_judge_county_event_counts(GS)
    assigned_days = get_day_assignments(GS)
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

##### Debugging Functions #####

##### Main Function #####
def ad_hoc_algorithm(init_mu_t,init_mu_p,tolerance=0.05,opt_iter=100,GS=False):
    sdf = load_sentencing_data()
    pleas = get_clean_day_pleas()
    judge_days = get_judge_days(GS)
    prev_mu_t = 0
    prev_mu_p = 0
    mu_t = init_mu_t
    mu_p = init_mu_p
    iters = 1
    param_values = [{'MuP':mu_p,'MuT':mu_t,'Iteration':0}]
    while (abs(mu_t - prev_mu_t) > tolerance) or (abs(mu_p - prev_mu_p) > tolerance):
        prev_mu_p = mu_p
        prev_mu_t = mu_t
        mu_p = optimize_mu_p(prev_mu_t,prev_mu_p,sdf,judge_days,pleas,opt_iter,0.02,GS)
        mu_t = estimate_mu_t(mu_p,GS)

        # NLL_data = make_NLL_data(prev_mu_t,prev_mu_p)
        # NLL_filename = optimization_data_folder + 'opt_data_{}.csv'.format(iters)
        # NLL_data.to_csv(NLL_filename,index=False)
        #
        # mu_t_data = make_mu_t_data(prev_mu_p)
        # mu_t_filename = optimization_data_folder + 'mut_data_{}.csv'.format(iters)
        # mu_t_data.to_csv(mu_t_filename,index=False)

        param_values.append({'MuP':mu_p,'MuT':mu_t,'Iteration':iters})
        iters += 1

        print('mu_p: {}, mu_t: {}\n'.format(mu_p,mu_t))
    param_df = pd.DataFrame(param_values)
    pdf_filename = optimization_data_folder + 'ad_hoc_param_data.csv'
    param_df.to_csv(pdf_filename,index=False,float_format='%.3f')
    return(param_df)
