import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from patsy import dmatrices

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

##### Baseline method for capacity estimation #####
def regression_analysis():
    results = []
    for group in ['JudgeID','County']:
        df = make_regression_data(group)
        df = df.loc[df.Trial > 0,:]
        y, X = dmatrices('Days ~ Plea + Trial - 1',data=df,return_type='dataframe')
        model = OLS(y,X).fit()
        residuals = model.resid.reset_index(name='Residual')
        residuals = residuals.merge(df[['Plea']],left_on=group,right_index=True)
        residuals.sort_values('Plea',ascending=False,inplace=True)

        # Residuals plot
        plt.figure()
        plt.bar(residuals[group],residuals.Residual)
        plt.xticks(rotation=90)
        plt.ylabel('Residuals')
        plt.grid(axis='y')
        filename = FIGURE_DIR + 'resid_plot_{}.png'.format(group)
        plt.savefig(filename, dpi=120)

        # True vs fitted
        plt.figure()
        plt.plot(model.fittedvalues,y,'o')
        plt.ylabel('True values')
        plt.xlabel('Fitted values')
        filename = FIGURE_DIR + 'true_vs_fitted_{}.png'.format(group)
        plt.savefig(filename)

        # Lambda plots
        df['TrialDays'] = df['Trial']*model.params['Trial']
        df['PleaDays'] = df['Days']-df['TrialDays']
        df['Lambda'] = df['Plea']/df['PleaDays']
        df.sort_values('Plea',ascending=False,inplace=True)
        # plt.figure()
        # plt.bar(df.index,df.Lambda)
        # plt.ylabel('Lambda')
        # plt.grid(axis='y')
        # plt.xticks(rotation=90)
        filename = TABLE_DIR + 'lambda_table_{}.tex'.format(group)
        df.to_latex(filename,float_format="%.2f")

        results.append({'Model':group,'Plea':model.params['Plea'],'$Trial$':model.params['Trial'],
            '$R^2$':model.rsquared,'$R^2$ Adj':model.rsquared_adj})

    results = pd.DataFrame(results)
    filename = TABLE_DIR + 'regression_results.tex'
    results.to_latex(filename,index=False,float_format='%.2f',escape=False)
    plt.close('all')

def make_regression_data(group):
    sdf = load_sentencing_data()
    sdf = sdf.loc[sdf.WorkType == 'GS',:]
    counts = sdf.groupby(group)[['Plea','Trial']].sum().reset_index()

    cdf = load_calendar_data()
    cdf = cdf.loc[cdf.WorkType == 'GS',:]
    days = cdf.groupby(group)['Days'].sum().reset_index()
    counts = counts.merge(days,on=group)
    counts = counts.set_index(group)
    return(counts)

def estimate_service_rate_model(df):
    y, X = dmatrices('Days ~ Plea + Trial - 1',data=df,return_type='dataframe')
    model = OLS(y,X).fit()
    return model

##### Busy County Analysis #####
def make_county_data():
    sdf = load_sentencing_data()
    group = 'County'
    sdf = sdf.loc[sdf.WorkType == 'GS',:]
    counts = sdf.groupby(group)[['Plea','Trial']].sum().reset_index()

    cdf = load_calendar_data()
    cdf = cdf.loc[cdf.WorkType == 'GS',:]
    days = cdf.groupby(group)['Days'].sum().reset_index()
    counts = counts.merge(days,on=group)

    cols = ['Plea','Trial','Days']
    for col in cols:
        rank_col = col+'Ranking'
        col_ranking = counts[[col,'County']].sort_values(col,ascending=False)
        col_ranking[rank_col] = range(1,counts.shape[0]+1)
        counts = counts.merge(col_ranking[[rank_col,'County']],on='County')

    counts['OverallScore'] = counts['PleaRanking']*counts['TrialRanking']*counts['DaysRanking']
    counts.sort_values('OverallScore',inplace=True)
    counts.index = range(1,counts.shape[0]+1)
    cols = ['County','Plea','Trial','Days','OverallScore']
    counts = counts[cols]
    filename = TABLE_DIR + 'county_rankings.tex'
    counts.to_latex(filename,float_format='%.2f')
    return(counts)

def county_bar_charts():
    df = make_county_data()
    plea_service_rate = 0.17
    trial_service_rate = 4.23
    df['WorkDays'] = df['Plea']*0.17 + df['Trial']*4.23
    df['Utilization'] = df['WorkDays']/df['Days']

    plt.figure()
    plt.bar(df.County,df.Plea)
    plt.ylabel('Pleas')
    plt.xticks(rotation=90)
    filename = FIGURE_DIR + 'county_pleas.png'
    plt.savefig(filename)

    plt.figure()
    plt.bar(df.County,df.Trial)
    plt.ylabel('Trials')
    plt.xticks(rotation=90)
    filename = FIGURE_DIR + 'county_trials.png'
    plt.savefig(filename)

    plt.figure()
    plt.bar(df.County,df.Days)
    plt.ylabel('GS Days')
    plt.xticks(rotation=90)
    filename = FIGURE_DIR + 'county_days.png'
    plt.savefig(filename)

    plt.figure()
    plt.bar(df.County,df.Utilization)
    plt.ylabel('Utilization')
    plt.xticks(rotation=90)
    filename = FIGURE_DIR + 'county_utilization.png'
    plt.savefig(filename)

    plt.close('all')

def cdf_table():
    df = make_county_data()
    df['PleaShare'] = df.Plea.cumsum()/df.Plea.sum()
    df['TrialShare'] = df.Trial.cumsum()/df.Trial.sum()
    df['GSDayShare'] = df.Days.cumsum()/df.Days.sum()
    df.index = range(1,df.shape[0]+1)

    filename = TABLE_DIR + 'county_CDF_table.tex'
    df.to_latex(filename,float_format='%.2f')

def busy_vs_idle_plea_hists():
    df = make_county_data()
    sdf = load_sentencing_data()
    busy = df.loc[1:23,'County']
    idle = df.loc[24:,'County']

    daily_counts = sdf.groupby(['County','Date'])['Plea'].sum().reset_index()
    bdf = daily_counts.loc[daily_counts.County.isin(busy),:]
    idf = daily_counts.loc[daily_counts.County.isin(idle),:]

    fig, axes = plt.subplots(1,2,figsize=(8,5))

    axes[0].hist(bdf.Plea,bins=50)
    plea_mean = bdf.Plea.mean()
    axes[0].axvline(x=plea_mean,color='r')
    x_ticks = np.append(axes[0].get_xticks(),plea_mean)
    axes[0].set_xticks(x_ticks)
    axes[0].set_title('Daily Pleas for Busy Counties')
    axes[0].tick_params(labelrotation=90)

    axes[1].hist(idf.Plea,bins=50)
    plea_mean = idf.Plea.mean()
    axes[1].axvline(x=plea_mean,color='r')
    x_ticks = np.append(axes[1].get_xticks(),plea_mean)
    axes[1].set_xticks(x_ticks)
    axes[1].set_title('Daily Pleas for Non Busy Counties')
    axes[1].tick_params(labelrotation=90)
    plt.tight_layout()
    filename = FIGURE_DIR + 'busy_vs_idle_plea_hists.png'
    plt.savefig(filename)
    plt.close('all')

def busy_vs_idle_lambda_hists():
    df = make_regression_data('County')
    odf = make_county_data()

    df = df.merge(odf[['County','OverallScore']],right_on='County',left_index=True)
    df.sort_values('OverallScore',inplace=True)
    df.index = range(1,df.shape[0]+1)

    y, X = dmatrices('Days ~ Plea + Trial - 1',data=df,return_type='dataframe')
    model = OLS(y,X).fit()

    df['TrialDays'] = df['Trial']*model.params['Trial']
    df['PleaDays'] = df['Days']-df['TrialDays']
    df['Lambda'] = df['Plea']/df['PleaDays']

    bdf = df.loc[1:23,:]
    idf = df.loc[24:,:]

    fig, axes = plt.subplots(1,2,figsize=(8,5))

    axes[0].hist(bdf.Lambda)
    plea_mean = bdf.Lambda.mean()
    axes[0].axvline(x=plea_mean,color='r')
    x_ticks = np.append(axes[0].get_xticks(),plea_mean)
    axes[0].set_xticks(x_ticks)
    axes[0].set_title('Lambdas for Busy Counties')
    axes[0].tick_params(labelrotation=90)

    axes[1].hist(idf.Lambda)
    plea_mean = idf.Lambda.mean()
    axes[1].axvline(x=plea_mean,color='r')
    x_ticks = np.append(axes[1].get_xticks(),plea_mean)
    axes[1].set_xticks(x_ticks)
    axes[1].set_title('Lambdas for Non Busy Counties')
    axes[1].tick_params(labelrotation=90)
    plt.tight_layout()
    filename = FIGURE_DIR + 'lambda_hists.png'
    plt.savefig(filename)
    plt.close('all')

##### Busy Judge Analysis ####
def make_judge_data():
    sdf = load_sentencing_data()
    group = 'JudgeID'
    sdf = sdf.loc[sdf.WorkType == 'GS',:]
    counts = sdf.groupby(group)[['Plea','Trial']].sum().reset_index()

    cdf = load_calendar_data()
    cdf = cdf.loc[cdf.WorkType == 'GS',:]
    days = cdf.groupby(group)['Days'].sum().reset_index()
    counts = counts.merge(days,on=group)

    cols = ['Plea','Trial','Days']
    for col in cols:
        rank_col = col+'Ranking'
        col_ranking = counts[[col,'JudgeID']].sort_values(col,ascending=False)
        col_ranking[rank_col] = range(1,counts.shape[0]+1)
        counts = counts.merge(col_ranking[[rank_col,'JudgeID']],on='JudgeID')

    counts['OverallScore'] = counts['PleaRanking']*counts['TrialRanking']*counts['DaysRanking']
    counts.sort_values('OverallScore',inplace=True)
    counts.index = range(1,counts.shape[0]+1)
    cols = ['JudgeID','Plea','Trial','Days','OverallScore']
    counts = counts[cols]
    # filename = TABLE_DIR + 'judge_rankings.tex'
    # counts.to_latex(filename,float_format='%.2f')
    return(counts)

def judge_bar_charts():
    df = make_judge_data()
    plea_service_rate = 0.17
    trial_service_rate = 4.23
    df['WorkDays'] = df['Plea']*0.17 + df['Trial']*4.23
    df['Utilization'] = df['WorkDays']/df['Days']

    plt.figure()
    plt.bar(df.JudgeID,df.Plea)
    plt.ylabel('Pleas')
    plt.xticks(rotation=90)
    filename = FIGURE_DIR + 'judge_pleas.png'
    plt.savefig(filename)

    plt.figure()
    plt.bar(df.JudgeID,df.Trial)
    plt.ylabel('Trials')
    plt.xticks(rotation=90)
    filename = FIGURE_DIR + 'judge_trials.png'
    plt.savefig(filename)

    plt.figure()
    plt.bar(df.JudgeID,df.Days)
    plt.ylabel('GS Days')
    plt.xticks(rotation=90)
    filename = FIGURE_DIR + 'judge_days.png'
    plt.savefig(filename)

    plt.figure()
    plt.bar(df.JudgeID,df.Utilization)
    plt.ylabel('Utilization')
    plt.xticks(rotation=90)
    filename = FIGURE_DIR + 'judge_utilization.png'
    plt.savefig(filename)

    plt.close('all')

def cdf_table_judge():
    df = make_judge_data()
    df['PleaShare'] = df.Plea.cumsum()/df.Plea.sum()
    df['TrialShare'] = df.Trial.cumsum()/df.Trial.sum()
    df['GSDayShare'] = df.Days.cumsum()/df.Days.sum()
    df.index = range(1,df.shape[0]+1)

    filename = TABLE_DIR + 'judge_CDF_table.tex'
    df.to_latex(filename,float_format='%.2f')

def busy_vs_idle_plea_hists_judge():
    df = make_judge_data()
    sdf = load_sentencing_data()
    busy = df.loc[1:23,'JudgeID']
    idle = df.loc[24:,'JudgeID']

    daily_counts = sdf.groupby(['JudgeID','Date'])['Plea'].sum().reset_index()
    bdf = daily_counts.loc[daily_counts.JudgeID.isin(busy),:]
    idf = daily_counts.loc[daily_counts.JudgeID.isin(idle),:]

    fig, axes = plt.subplots(1,2,figsize=(8,5))

    axes[0].hist(bdf.Plea,bins=50)
    plea_mean = bdf.Plea.mean()
    axes[0].axvline(x=plea_mean,color='r')
    x_ticks = np.append(axes[0].get_xticks(),plea_mean)
    axes[0].set_xticks(x_ticks)
    axes[0].set_title('Daily Pleas for Busy Judge')
    axes[0].tick_params(labelrotation=90)

    axes[1].hist(idf.Plea,bins=50)
    plea_mean = idf.Plea.mean()
    axes[1].axvline(x=plea_mean,color='r')
    x_ticks = np.append(axes[1].get_xticks(),plea_mean)
    axes[1].set_xticks(x_ticks)
    axes[1].set_title('Daily Pleas for Non Busy Judge')
    axes[1].tick_params(labelrotation=90)
    plt.tight_layout()
    filename = FIGURE_DIR + 'busy_vs_idle_judge_plea_hists.png'
    plt.savefig(filename)
    plt.close('all')

def busy_vs_idle_lambda_hists():
    df = make_regression_data('JudgeID')
    odf = make_judge_data()

    df = df.merge(odf[['JudgeID','OverallScore']],right_on='JudgeID',left_index=True)
    df.sort_values('OverallScore',inplace=True)
    df.index = range(1,df.shape[0]+1)

    y, X = dmatrices('Days ~ Plea + Trial',data=df,return_type='dataframe')
    model = OLS(y,X).fit()

    df['TrialDays'] = df['Trial']*model.params['Trial']
    df['PleaDays'] = df['Days']-df['TrialDays']
    df['Lambda'] = df['Plea']/df['PleaDays']

    bdf = df.loc[1:23,:]
    idf = df.loc[24:,:]

    fig, axes = plt.subplots(1,2,figsize=(8,5))

    axes[0].hist(bdf.Lambda)
    plea_mean = bdf.Lambda.mean()
    axes[0].axvline(x=plea_mean,color='r')
    x_ticks = np.append(axes[0].get_xticks(),plea_mean)
    axes[0].set_xticks(x_ticks)
    axes[0].set_title('Lambdas for Busy Counties')
    axes[0].tick_params(labelrotation=90)

    axes[1].hist(idf.Lambda)
    plea_mean = idf.Lambda.mean()
    axes[1].axvline(x=plea_mean,color='r')
    x_ticks = np.append(axes[1].get_xticks(),plea_mean)
    axes[1].set_xticks(x_ticks)
    axes[1].set_title('Lambdas for Non Busy Counties')
    axes[1].tick_params(labelrotation=90)
    plt.tight_layout()
    filename = FIGURE_DIR + 'lambda_hists.png'
    plt.savefig(filename)
    plt.close('all')

##### Clean Day Analysis #####
def clean_day_analysis():
    for group in ['JudgeID','County']:
        sdf = load_sentencing_data()
        cdf = get_clean_day_sentencing_data(sdf)

        clean_days = cdf.groupby([group,'Date'])['Plea'].sum().reset_index()
        counts = clean_days.groupby(group).size().reset_index(name='N')
        counts.sort_values('N',ascending=False,inplace=True)
        counts['Share'] = counts.N/counts.N.sum()
        counts['CDF'] = counts.Share.cumsum()
        counts.index = range(1,counts.shape[0]+1)
        filename = TABLE_DIR + 'clean_day_dist_{}.tex'.format(group)
        counts.to_latex(filename,float_format="%.2f")

        plt.figure(figsize=(9,5))
        plt.xticks(rotation=90)
        plt.bar(counts[group],counts.Share)

        plt.ylabel('Share')
        plt.grid(axis='y')
        plt.title('Distribution of clean days across {}'.format(group))
        filename = FIGURE_DIR + 'clean_day_dist_{}.png'.format(group)
        plt.savefig(filename, dpi=120)
        plt.close('all')

def restriction_analysis():
    df = []
    sdf = load_sentencing_data()
    plea_mean = sdf.groupby(['JudgeID','Date'])['Plea'].sum().mean()
    df.append({'Restriction':'None','Average Pleas Per Day':plea_mean})

    sdf = remove_conflicting_days(sdf)
    plea_mean = sdf.groupby(['JudgeID','Date'])['Plea'].sum().mean()
    df.append({'Restriction':'No conflicting days','Average Pleas Per Day':plea_mean})

    sdf = sdf.loc[sdf.WorkType == 'GS',:]
    plea_mean = sdf.groupby(['JudgeID','Date'])['Plea'].sum().mean()
    df.append({'Restriction':'Only GS days','Average Pleas Per Day':plea_mean})

    sdf = remove_multi_county_days(sdf)
    plea_mean = sdf.groupby(['JudgeID','Date'])['Plea'].sum().mean()
    df.append({'Restriction':'No mutli-county days','Average Pleas Per Day':plea_mean})

    sdf = remove_multi_assignment_days(sdf)
    plea_mean = sdf.groupby(['JudgeID','Date'])['Plea'].sum().mean()
    df.append({'Restriction':'No mutli-assignment days','Average Pleas Per Day':plea_mean})

    sdf = remove_days_with_few_sentencing_events(sdf)
    plea_mean = sdf.groupby(['JudgeID','Date'])['Plea'].sum().mean()
    df.append({'Restriction':'No days with less than 10 events','Average Pleas Per Day':plea_mean})

    df = pd.DataFrame(df)
    filename = TABLE_DIR + 'restriction_table.tex'
    df.to_latex(filename,index=False,float_format='%.2f')

def get_clean_day_sentencing_data(sdf):
    sdf = remove_conflicting_days(sdf)
    sdf = sdf.loc[sdf.WorkType == 'GS',:]
    sdf = remove_multi_county_days(sdf)
    sdf = remove_multi_assignment_days(sdf)
    sdf = remove_days_with_few_sentencing_events(sdf)
    return sdf

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

def remove_trial_judge_county_combinations(df):
    sdf = load_sentencing_data()
    trial_jcs = sdf.loc[sdf.Trial == 1,['JudgeID','County']].drop_duplicates()
    trial_jcs['TrialJudgeCounty'] = True
    df = df.merge(trial_jcs,on=['JudgeID','County'],how='left')
    df = df.loc[df.TrialJudgeCounty.isna(),df.columns != 'TrialJudgeCounty']
    return df

def remove_days_with_few_sentencing_events(sdf,threshold=10):
    judge_clean_days = sdf.groupby(['JudgeID','Date']).size().reset_index(name='N')
    good_days = judge_clean_days.loc[judge_clean_days.N >= threshold,['JudgeID','Date']]
    sdf = sdf.merge(good_days,on=['JudgeID','Date'])
    return sdf

def main():
    judge_bar_charts()
    cdf_table_judge()
    busy_vs_idle_plea_hists_judge()

main()
