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

def estimate_service_rate_model(df):
    y, X = dmatrices('Days ~ Plea + Trial - 1',data=df,return_type='dataframe')
    model = OLS(y,X).fit()
    return model

##### Iterative Model #####
def utilization_model(group):
    df = make_regression_data(group)
    y, X = dmatrices('Days ~ Plea + Trial - 1',data=df,return_type='dataframe')
    model = OLS(y,X).fit()
    df['TrialDays'] = df['Trial']*model.params['Trial']
    df['PleaDays'] = df['Plea']*model.params['Plea']
    df['Utilization'] = (df.TrialDays + df.PleaDays)/df['Days']

    max_util = df.Utilization.max()
    df['Idleness'] = df['Utilization']/max_util

    tolerance = 0.05
    prev_mu_p = 0
    prev_mu_t = 0

    mu_p = model.params['Plea']
    mu_t = model.params['Trial']
    param_values  = [{'Iteration':0,'Beta P':mu_p, 'Beta T':mu_t}]
    iter = 1
    while (abs(mu_t - prev_mu_t) > tolerance) or (abs(mu_p - prev_mu_p) > tolerance):
        prev_mu_p = mu_p
        prev_mu_t = mu_t
        print('mu_p: {}, mu_t: {}'.format(mu_p,mu_t))

        y = df['Days']*df['Idleness']
        model = OLS(y,X).fit()

        df['TrialDays'] = df['Trial']*model.params['Trial']
        df['PleaDays'] = df['Plea']*model.params['Plea']
        df['Utilization'] = (df.TrialDays + df.PleaDays)/df['Days']

        max_util = df.Utilization.max()
        df['Idleness'] = df['Utilization']/max_util

        mu_p = model.params['Plea']
        mu_t = model.params['Trial']
        param_values.append({'Iteration':iter,'Beta P':mu_p, 'Beta T':mu_t})
        iter += 1

    print(model.summary())
    if isinstance(group,list):
        group = ''.join(group)
    # True vs fitted
    plt.figure()
    plt.plot(model.fittedvalues,y,'o')
    plt.ylabel('True values')
    plt.xlabel('Fitted values')
    filename = FIGURE_DIR + 'fit_utilization_{}.png'.format(group)
    plt.savefig(filename)

    filename = TABLE_DIR + 'utilization_model_convergence_{}.tex'.format(group)
    df.sort_values('Utilization',ascending=False,inplace=True)
    df.to_latex(filename,index=False,float_format="%.2f")

    iters = pd.DataFrame(param_values)
    filename = TABLE_DIR + 'utilization_model_iters_{}.tex'.format(group)
    iters.to_latex(filename,index=False,float_format="%.2f")

    # latex_string = model.summary().as_latex()
    # filename = TABLE_DIR + 'utilization_model_{}.tex'.format(group)
    # with open(filename,'w') as f:
    #     f.write(latex_string)
    return {'Model':'Utilization','Group':group,'Pleas per Day':1/mu_p,'Days per Trial':mu_t}

def min_model(group):
    df = make_regression_data(group)
    y, X = dmatrices('Days ~ Plea + Trial - 1',data=df,return_type='dataframe')
    model = OLS(y,X).fit()
    df['TrialDays'] = df['Trial']*model.params['Trial']
    df['PleaDays'] = df['Plea']*model.params['Plea']
    df['ExpectedDays'] = df.TrialDays + df.PleaDays

    tolerance = 0.05
    prev_mu_p = 0
    prev_mu_t = 0

    mu_p = model.params['Plea']
    mu_t = model.params['Trial']

    param_values  = [{'Iteration':0,'Beta P':mu_p, 'Beta T':mu_t}]
    iter = 1
    while (abs(mu_t - prev_mu_t) > tolerance) or (abs(mu_p - prev_mu_p) > tolerance):
        prev_mu_p = mu_p
        prev_mu_t = mu_t
        print('mu_p: {}, mu_t: {}'.format(mu_p,mu_t))

        y = df[['Days','ExpectedDays']].min(axis=1)
        model = OLS(y,X).fit()

        df['TrialDays'] = df['Trial']*model.params['Trial']
        df['PleaDays'] = df['Plea']*model.params['Plea']
        df['ExpectedDays'] = df.TrialDays + df.PleaDays

        mu_p = model.params['Plea']
        mu_t = model.params['Trial']

        param_values.append({'Iteration':iter,'Beta P':mu_p, 'Beta T':mu_t})
        iter += 1

    print(model.summary())
    if isinstance(group,list):
        group = ''.join(group)
    # True vs fitted
    plt.figure()
    plt.plot(model.fittedvalues,y,'o')
    plt.ylabel('True values')
    plt.xlabel('Fitted values')
    filename = FIGURE_DIR + 'fit_min_{}.png'.format(group)
    plt.savefig(filename)

    iters = pd.DataFrame(param_values)
    filename = TABLE_DIR + 'min_model_iters_{}.tex'.format(group)
    iters.to_latex(filename,index=False,float_format="%.2f")

    # latex_string = model.summary().as_latex()
    # filename = TABLE_DIR + 'min_model_{}.tex'.format(group)
    # with open(filename,'w') as f:
    #     f.write(latex_string)
    return {'Model':'Min','Group':group,'Pleas per Day':1/mu_p,'Days per Trial':mu_t}

def all_time_min_model(group):
    df = make_regression_data(group)
    y, X = dmatrices('Days ~ Plea + Trial',data=df,return_type='dataframe')
    model = OLS(y,X).fit()
    df['TrialDays'] = df['Trial']*model.params['Trial']
    df['PleaDays'] = df['Plea']*model.params['Plea']
    df['ExpectedDays0'] = df.TrialDays + df.PleaDays

    tolerance = 0.05
    prev_mu_p = 0
    prev_mu_t = 0

    mu_p = model.params['Plea']
    mu_t = model.params['Trial']

    param_values  = [{'Iteration':0,'Beta P':mu_p, 'Beta T':mu_t}]
    day_cols = ['Days','ExpectedDays0']
    iter = 1
    while (abs(mu_t - prev_mu_t) > tolerance) or (abs(mu_p - prev_mu_p) > tolerance):
        prev_mu_p = mu_p
        prev_mu_t = mu_t
        print('mu_p: {}, mu_t: {}'.format(mu_p,mu_t))
        new_day_col = 'ExpectedDays{}'.format(iter)

        y = df[day_cols].min(axis=1)
        model = OLS(y,X).fit()

        df['TrialDays'] = df['Trial']*model.params['Trial']
        df['PleaDays'] = df['Plea']*model.params['Plea']
        df[new_day_col] = df.TrialDays + df.PleaDays

        mu_p = model.params['Plea']
        mu_t = model.params['Trial']

        param_values.append({'Iteration':iter,'Beta P':mu_p, 'Beta T':mu_t})
        iter += 1
        day_cols.append(new_day_col)

    print(model.summary())
    if isinstance(group,list):
        group = ''.join(group)
    # True vs fitted
    plt.figure()
    plt.plot(model.fittedvalues,y,'o')
    plt.ylabel('True values')
    plt.xlabel('Fitted values')
    filename = FIGURE_DIR + 'fit_all_time_min_{}.png'.format(group)
    plt.savefig(filename)

    iters = pd.DataFrame(param_values)
    filename = TABLE_DIR + 'all_time_min_model_iters_{}.tex'.format(group)
    iters.to_latex(filename,index=False,float_format="%.2f")

    latex_string = model.summary().as_latex()
    filename = TABLE_DIR + 'all_time_min_model_{}.tex'.format(group)
    with open(filename,'w') as f:
        f.write(latex_string)
    return {'Model':'Min','Group':group,'Pleas per Day':1/mu_p,'Days per Trial':mu_t}

def main():
    overall_results = []
    for group in ['JudgeID','County',['JudgeID','County']]:
        util_results = utilization_model(group)
        min_results = min_model(group)
        overall_results.append(util_results)
        overall_results.append(min_results)

    results = pd.DataFrame(overall_results)
    filename = TABLE_DIR + 'regression_summary.tex'
    results.sort_values(['Model','Group'],inplace=True)
    results.to_latex(filename,index=False,float_format="%.2f")

main()
