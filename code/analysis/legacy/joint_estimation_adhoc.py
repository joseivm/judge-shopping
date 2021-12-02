import torch
import pandas as pd
import numpy as np
import os
import datetime
import re
import pickle
import math
import matplotlib.pyplot as plt

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
optimization_figures_folder = PROJECT_DIR + '/output/figures/Optimization/'
experiment_dir = PROJECT_DIR + '/output/experiments/'

holidays = ['2000-09-04','2000-10-09','2000-11-10','2000-11-23','2000-12-24',
'2000-12-25','2000-12-26','2001-01-01','2001-01-15','2001-02-19','2001-05-10',
'2001-05-28','2001-07-04','2000-07-04']

##### Data Loading/Cleaning #####
def load_sentencing_data():
    sdf = pd.read_csv(processed_sentencing_data_file)
    sdf['Circuit'] = sdf.Circuit.astype(str)
    sdf = impute_missing_dates(sdf)
    sdf = add_work_type(sdf)
    return sdf

def load_calendar_data():
    cdf = pd.read_csv(processed_daily_schedule_file)
    mapping = pd.read_csv(judge_name_id_mapping_file)
    cdf = cdf.merge(mapping,on='JudgeName')
    cdf = cdf.loc[~cdf.Date.isin(holidays),:]
    return cdf

##### Samples #####
### Theta sample ###
def get_theta_judge_days(sample):
    cdf = load_calendar_data()
    if sample == 'GS':
        cdf = cdf.loc[cdf.WorkType == 'GS',:]
    elif sample == 'Clean':
        cdf = get_clean_day_calendar_data(cdf)
    elif sample == 'GSExp':
        sdf = load_sentencing_data()
        sentencing_event_days = sdf.groupby(['JudgeID','Date']).size().reset_index(name='N')
        cdf = cdf.merge(sentencing_event_days,on=['JudgeID','Date'],how='outer')
        cdf.loc[cdf.Days.isna(),'Days'] = 1
        cdf = cdf.loc[(cdf.WorkType == 'GS') | (cdf.N.notna()),:]

    return cdf

def get_theta_pleas(sample):
    sdf = load_sentencing_data()
    if sample == 'GS':
        sdf = sdf.loc[sdf.WorkType == 'GS',:]
    elif sample == 'Clean':
        sdf = get_clean_day_sentencing_data(sdf)
    elif sample == 'GSExp':
        pass
    return sdf.Plea.sum()

def get_theta_trials(sample):
    sdf = load_sentencing_data()
    if sample == 'GS':
        sdf = sdf.loc[sdf.WorkType == 'GS',:]
    elif sample == 'Clean':
        sdf = get_clean_day_sentencing_data(sdf)
    elif sample == 'GSExp':
        pass
    return sdf.Trial.sum()

def get_clean_day_calendar_data(cdf):
    cdf = remove_conflicting_days(cdf)
    cdf = remove_multi_county_days(cdf)
    cdf = remove_multi_assignment_days(cdf)
    cdf = remove_judges_with_few_clean_days_from_calendar(cdf)
    return cdf

def remove_judges_with_few_clean_days_from_calendar(cdf):
    sdf = load_sentencing_data()
    clean_days = get_clean_day_sentencing_data(sdf)
    clean_judges = clean_days.JudgeID.unique()
    cdf = cdf.loc[cdf.JudgeID.isin(clean_judges),:]
    return cdf

### mu_t sample ###
def get_judge_county_event_counts(sample):
    sdf = load_sentencing_data()
    sdf = trial_capacity_sample(sdf)
    if sample == 'GS':
        sdf = sdf.loc[sdf.WorkType == 'GS',:]
    elif sample == 'Clean':
        sdf = get_clean_day_sentencing_data(sdf)
    elif sample == 'GSExp':
        pass
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

def get_day_assignments(sample):
    cdf = load_calendar_data()
    if sample == 'GS':
        cdf = cdf.loc[cdf.WorkType == 'GS',:]
    elif sample == 'Clean':
        cdf = get_clean_day_calendar_data(cdf)
    elif sample == 'GSExp':
        sdf = load_sentencing_data()
        sentencing_event_days = sdf.groupby(['JudgeID','Date']).size().reset_index(name='N')
        cdf = cdf.merge(sentencing_event_days,on=['JudgeID','Date'],how='outer')
        cdf.loc[cdf.Days.isna(),'Days'] = 1
        cdf = cdf.loc[(cdf.WorkType == 'GS') | (cdf.N.notna()),:]
    assigned_days = cdf.groupby(['JudgeName','County'])['Days'].sum().reset_index()
    return assigned_days

def get_clean_day_sentencing_data(sdf):
    sdf = remove_conflicting_days(sdf)
    sdf = sdf.loc[sdf.WorkType == 'GS',:]
    sdf = remove_multi_county_days(sdf)
    sdf = remove_multi_assignment_days(sdf)
    sdf = remove_judges_with_few_clean_days(sdf)
    return sdf

### MLE sample ###
def get_MLE_pleas(sample,topcode=150):
    sdf = load_sentencing_data()
    if sample == 'GS':
        sdf = sdf.loc[sdf.WorkType == 'GS',:]
    elif sample == 'Clean':
        sdf = get_clean_day_sentencing_data(sdf)
    elif sample == 'GSExp':
        pass
    counts = sdf.groupby(['JudgeID','County','Date'])[['Plea']].sum().reset_index()
    counts.loc[counts.Plea > topcode,'Plea'] = topcode
    return counts['Plea']

def remove_conflicting_days(df):
    cdf = load_calendar_data()
    sdf = load_sentencing_data()
    fdf = sdf.merge(cdf,on=['Date','JudgeName'])
    fdf['Circuit_y'] = fdf.Circuit_y.replace({'na':np.nan})
    fdf['Circuit_y'] = fdf.Circuit_y.astype('float').astype('Int64')
    fdf.loc[fdf.County_x != fdf.County_y,'MatchType'] = 'Disagree'
    fdf.loc[fdf.Circuit_x == fdf.Circuit_y,'MatchType'] = 'Circuit'
    fdf.loc[fdf.County_x == fdf.County_y,'MatchType'] = 'Agree'
    fdf.sort_values('MatchType',inplace=True)
    fdf = fdf.drop_duplicates('EventID',keep='first')
    conflicting_days = fdf.loc[fdf.MatchType == 'Disagree',['Date','JudgeName','MatchType']].drop_duplicates(['Date','JudgeName'])
    df = df.merge(conflicting_days,on=['Date','JudgeName'],how='left')
    df = df.loc[df.MatchType.isna(),:]
    df.drop(columns=['MatchType'],inplace=True)
    return df

def remove_non_GS_days(sdf):
    sdf = sdf.loc[sdf.WorkType == 'GS',:]
    # cdf = load_calendar_data()
    # gs_days = cdf.loc[cdf.WorkType == 'GS',['JudgeName','Date']]
    # sdf = sdf.merge(gs_days,on=['JudgeName','Date'])
    return sdf

def remove_multi_county_days(df):
    sdf = load_sentencing_data()
    judge_events = sdf.groupby(['JudgeID','Date','County']).size().reset_index(name='N')
    counties_by_day = judge_events.groupby(['JudgeID','Date']).size().reset_index(name='N')
    multi_county_days = counties_by_day.loc[counties_by_day.N > 1,['JudgeID','Date','N']]
    df = df.merge(multi_county_days,on=['JudgeID','Date'],how='left')
    df = df.loc[df.N.isna(),:]
    df.drop(columns=['N'],inplace=True)
    return df

def remove_multi_assignment_days(df):
    cdf = load_calendar_data()
    day_assignments = cdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    single_assignment_days = day_assignments.loc[day_assignments.N == 1,['JudgeName','Date']]
    df = df.merge(single_assignment_days,on=['JudgeName','Date'])
    return df

def remove_judges_with_few_clean_days(sdf,threshold=10):
    judge_clean_days = sdf.groupby(['JudgeName','Date']).size().reset_index(name='N')
    good_judges = judge_clean_days.loc[judge_clean_days.N >= threshold,['JudgeName','Date']]
    sdf = sdf.merge(good_judges,on=['JudgeName','Date'])
    return sdf

##### Estimation/Optimization Functions #####
def calculate_NLL(pleas,theta,mu_p):
    eps = 10**(-8)
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

        likelihood = max(theta_term*theta_sum+mu_term*mu_sum,eps)
        NLL += -math.log(likelihood)
    return NLL

def plot_NLL_optimization(opt_mu_p,theta,pleas,iters,out_dir,all=False):
    NLLS = []
    if all:
        start_mu_p = 2
        end_mu_p = 100
    else:
        start_mu_p = opt_mu_p - 10
        end_mu_p = opt_mu_p + 10
    mu_ps = np.linspace(start_mu_p,end_mu_p,num=500)
    for mu_p in mu_ps:
        NLL = calculate_NLL(pleas,theta,mu_p)
        NLLS.append(NLL)

    brute_force_min = mu_ps[np.argmin(NLLS)]
    filename = out_dir + 'mu_p_opt_{}.png'.format(iters)
    plt.figure()
    plt.plot(mu_ps,NLLS)
    plt.axvline(x=brute_force_min,color='g')
    plt.axvline(x=opt_mu_p,color='r')
    # plt.show()
    plt.savefig(filename, dpi=120)
    plt.close('all')

def optimize_mu_p_brute_force(mu_p,theta,pleas,size=500,window=10):
    NLLS = []
    start_mu_p = max(mu_p - window,0)
    end_mu_p = mu_p + window
    mu_ps = np.linspace(start_mu_p,end_mu_p,num=size)
    for mu_p in mu_ps:
        NLL = calculate_NLL(pleas,theta,mu_p)
        NLLS.append(NLL)

    brute_force_min = mu_ps[np.argmin(NLLS)]
    return brute_force_min

def optimize_mu_p(mu_t,mu_p,sdf,judge_days,pleas,iters=100,lr=0.02,GS=False):
    if GS:
        sdf = sdf.loc[sdf.WorkType == 'GS']
    num_trials = sdf['Trial'].sum()
    trial_days = num_trials/mu_t
    plea_days = judge_days - trial_days
    theta = sdf['Plea'].sum()/plea_days
    mu_p = torch.tensor([mu_p],requires_grad=True)
    # print("Trials: {}, Trial Days: {}, Pleas: {}, Plea Days: {}, Theta: {}".format(num_trials,trial_days,sdf.Plea.sum(),plea_days,theta))
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
            for i in range(0,s):
                theta_sum -= torch.pow(mu_p,i)*torch.exp(-mu_p)/math.factorial(i)

            mu_term = torch.pow(mu_p,s)*torch.exp(-mu_p)/math.factorial(s)
            mu_sum = 1
            for j in range(0,s+1):
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

def estimate_mu_t(mu_p,sample):
    counts = get_judge_county_event_counts(sample)
    assigned_days = get_day_assignments(sample)
    df = counts.merge(assigned_days,on=['JudgeName','County'])
    df['PleaDays'] = df['Plea']/mu_p
    df = df[df.Trial >=2]
    mu_t = df.Trial.sum()/(df.Days.sum()-df.PleaDays.sum())
    return mu_t

##### Debugging Functions #####

##### Main Function #####
def run_experiment(name,sample):
    exp_dir_name = experiment_dir+name+'/'
    hist_filename = exp_dir_name + 'plea_hist.png'
    os.mkdir(exp_dir_name)

    # MLE Plea Hist
    pleas = get_MLE_pleas(sample)
    plt.figure()
    plt.hist(pleas,bins=50)
    plt.savefig(hist_filename)
    plt.close('all')

    # Theta Stats
    judge_days = get_theta_judge_days(sample)
    theta_pleas = get_theta_pleas(sample)
    theta_trials = get_theta_trials(sample)

    df = pd.DataFrame({'Quantity':['Pleas','Trials','Judge Days'],'Value':[theta_pleas,theta_trials,judge_days]})
    filename = exp_dir_name + 'summary_stats.tex'
    df.to_latex(filename,index=False)

    # Mu_t stats

    init_mu_t = 0.14
    init_mu_p = 10.7

    trial_days = theta_trials/init_mu_t
    plea_days = judge_days - trial_days
    theta = theta_pleas/plea_days

    plot_NLL_optimization(init_mu_p,theta,pleas,iters=0,all=True,out_dir=exp_dir_name)
    ad_hoc_algorithm(init_mu_t,init_mu_p,exp_dir_name,sample,plea_topcode=150,tolerance=0.1)

def ad_hoc_algorithm(init_mu_t,init_mu_p,out_dir,sample,tolerance=0.1,size=500,plea_topcode=60):
    mle_pleas = get_MLE_pleas(sample,topcode=plea_topcode)
    judge_days = get_theta_judge_days(sample)
    theta_pleas = get_theta_pleas(sample)
    theta_trials = get_theta_trials(sample)

    prev_mu_t = 0
    prev_mu_p = 0
    mu_t = init_mu_t
    mu_p = init_mu_p
    iters = 1
    param_values = [{'MuP':mu_p,'MuT':mu_t,'Iteration':0}]
    while (abs(mu_t - prev_mu_t) > tolerance) or (abs(mu_p - prev_mu_p) > tolerance):
        prev_mu_p = mu_p
        prev_mu_t = mu_t

        trial_days = theta_trials/prev_mu_t
        plea_days = judge_days - trial_days
        theta = theta_pleas/plea_days

        window = max(50-iters*10,5)
        curr_size = min(size+iters*20,1000)

        mu_p = optimize_mu_p_brute_force(prev_mu_p,theta,mle_pleas,size=curr_size,window=window)
        mu_t = estimate_mu_t(mu_p,sample)

        plot_NLL_optimization(prev_mu_p,theta,mle_pleas,iters,out_dir=out_dir)

        param_values.append({'MuP':mu_p,'MuT':mu_t,'Iteration':iters})
        iters += 1

        print('mu_p: {}, mu_t: {}, theta: {}\n'.format(mu_p,mu_t,theta))
    param_df = pd.DataFrame(param_values)
    pdf_filename = optimization_data_folder + 'ad_hoc_param_data.csv'
    param_df.to_csv(pdf_filename,index=False,float_format='%.3f')
    return(param_df)
