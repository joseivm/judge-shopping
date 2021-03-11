import pandas as pd
import numpy as np
import os
import datetime
import re
import patsy as pt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

# TODO: include other regressors, bin age variable

# Input files/dirs
sentencing_data_file = PROJECT_DIR + '/data/raw/sentencing_data.csv'

# Output files
processed_sentencing_data_file = PROJECT_DIR + '/data/processed/sentencing_data.csv'

def make_covariate_matrix():
    df = pd.read_csv(sentencing_data_file)
    df = add_predicted_sentence(df)
    df = add_conviction_probability(df)
    df['ExpectedTrialSentence'] = df['ConvictionProb']*df['PredictedSentence']
    df = add_min_and_max_pleas(df)
    return(df)

def add_conviction_probability(df):
    theta = estimate_theta()
    formula = 'Black + OffenseType + C(OffenseSeriousness)'
    covariates = pt.dmatrix(formula,df,return_type='dataframe')
    df['ConvictionProb'] = theta.predict_proba(covariates)[:,1]
    return(df)

def estimate_theta():
    sdf = pd.read_csv(sentencing_data_file)
    sdf = sdf.loc[sdf.Trial == 1,:].dropna()
    formula = 'Incarceration ~ Black + OffenseType + C(OffenseSeriousness)'

    y, X = pt.dmatrices(formula,sdf,return_type='dataframe')
    theta = LogisticRegression(class_weight='balanced').fit(X,y.to_numpy().ravel())
    return(theta)

def add_predicted_sentence(df):
    tau = estimate_tau()
    formula = "Black + Age + OffenseType + C(OffenseSeriousness)"
    covariates = pt.dmatrix(formula,df,return_type='dataframe')
    df['PredictedSentence'] = tau.predict(covariates)
    return(df)

def estimate_tau():
    sdf = pd.read_csv(sentencing_data_file)
    sdf = sdf.loc[sdf.Trial == 1,:].dropna()
    sdf['ExpMinSentence'] = sdf['ExpMinSentence'].round(0)

    formula = "ExpMinSentence ~ Black + Age + OffenseType + C(OffenseSeriousness)"
    y, X = pt.dmatrices(formula,sdf,return_type='dataframe')
    poisson_model = sm.GLM(y,X,family=sm.families.Poisson()).fit()

    sdf['expmin_mu'] = poisson_model.mu
    sdf['ct_resp'] = sdf.apply(ct_response,axis=1)
    ols_model = smf.ols('ct_resp ~ expmin_mu -1',sdf).fit()
    dispersion_param = ols_model.params.to_numpy()[0]

    tau = sm.GLM(y,X,family=sm.families.NegativeBinomial(alpha=dispersion_param)).fit()
    return(tau)

def ct_response(row):
    y = row['ExpMinSentence']
    m = row['expmin_mu']
    return ((y-m)**2 -y)/m

def add_min_and_max_pleas(df):
    df['MinPlea'] = df.apply(get_min_plea,axis=1,args=(df,))
    df['MaxPlea'] = df.apply(get_max_plea,axis=1,args=(df,))
    return(df)

def get_min_plea(row,df):
    judge_id = row['JudgeID']
    expected_sentence = row['ExpectedTrialSentence']
    jdf = df.loc[df.JudgeID == judge_id,:]
    judge_pleas = df.loc[df.Trial == 0,['ExpectedTrialSentence','Sentence']].to_numpy()

    prior_expected_sentences = judge_pleas[:,0]
    distances = abs(prior_expected_sentences - expected_sentence)
    closest_indices = np.argsort(distances)[:6]
    closest_points = judge_pleas[closest_indices,:]
    min_plea = min(closest_points[:,1])
    return(min_plea)

def get_max_plea(row,df):
    judge_id = row['JudgeID']
    expected_sentence = row['ExpectedTrialSentence']
    jdf = df.loc[df.JudgeID == judge_id,:]
    judge_pleas = df.loc[df.Trial == 0,['ExpectedTrialSentence','Sentence']].to_numpy()

    prior_expected_sentences = judge_pleas[:,0]
    distances = abs(prior_expected_sentences - expected_sentence)
    closest_indices = np.argsort(distances)[:6]
    closest_points = judge_pleas[closest_indices,:]
    max_plea = max(closest_points[:,1])
    return(max_plea)

def main():
    df = make_covariate_matrix()
    df.to_csv(processed_sentencing_data_file,index=False)
