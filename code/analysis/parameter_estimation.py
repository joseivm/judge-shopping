import pandas as pd
import numpy as np
import os
import datetime
import re
import patsy as pt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import norm
from scipy.optimize import minimize
import sys
sys.path.append('code/analysis')
from court_simulation import Judge

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
PROJECT_DIR = os.environ.get("PROJECT_DIR")

#from court_simulation import Judge

# TODO: include other regressors, bin age variable

# Input files/dirs
sentencing_data_file = PROJECT_DIR + '/data/processed/augmented_sentencing_data.csv'

# Output files
processed_sentencing_data_file = PROJECT_DIR + '/data/processed/sentencing_data.csv'

def make_covariate_matrix():
    df = pd.read_csv(sentencing_data_file)
    df.loc[df.Sentence > 720,'Sentence'] = 720
    # df = add_predicted_sentence(df)
    df = add_conviction_probability(df)
    df['ExpectedTrialSentence'] = df['ConvictionProb']*df['PredictedSentence']
    df = add_min_and_max_pleas(df)
    # df = estimate_defendant_cost_of_trial(df)
    return(df)

def estimate_defendant_cost_of_trial(df):
    df.loc[(df.Sentence < df.MaxPlea) & (df.Plea == 1), 'DefendantType'] = 'SubMax'
    df.loc[(df.DefendantType != 'SubMax') & (df.Plea == 1),'DefendantType'] = 'Max'
    df.loc[df.Trial == 1,'DefendantType'] = 'Trial'

    df['ExtraSentence'] = df['Sentence'] - df['ExpectedTrialSentence']
    df['Leniency'] = df['MaxPlea'] - df['ExpectedTrialSentence']
    df['Harshness'] = df['MinPlea'] - df['ExpectedTrialSentence']

    submax_data = df.loc[df.DefendantType == 'SubMax','ExtraSentence'].to_numpy()
    max_data = df.loc[df.DefendantType == 'Max','Leniency'].to_numpy()
    trial_data = df.loc[df.DefendantType == 'Trial','Harshness'].to_numpy()

    x_0 = [8,8]
    res = minimize(NLL,x_0,args=(submax_data,max_data,trial_data,0.00001))

def NLL(x,submax_data,max_data,trial_data,eps=0.001):
    mu = x[0]
    sigma = x[1]
    dist = norm(loc=mu,scale=sigma)

    submax_pdf = dist.pdf(submax_data) + eps
    max_idf = 1-dist.cdf(max_data) + eps
    trial_cdf = dist.cdf(trial_data) + eps

    submax_ll = -np.log(submax_pdf).sum()
    max_ll = -np.log(max_idf).sum()
    trial_ll = -np.log(trial_cdf).sum()
    return(submax_ll+max_ll+trial_ll)

def add_conviction_probability(df):
    theta = estimate_theta()
    formula = 'Black + OffenseType + C(OffenseSeriousness)'
    covariates = pt.dmatrix(formula,df,return_type='dataframe')
    df['ConvictionProb'] = theta.predict_proba(covariates)[:,1]
    return(df)

def estimate_theta():
    sdf = pd.read_csv(sentencing_data_file)
    pred_cols = ['Incarceration','Black','OffenseType','OffenseSeriousness']
    sdf = sdf.loc[sdf.Trial == 1,pred_cols].dropna()
    formula = 'Incarceration ~ Black + OffenseType + C(OffenseSeriousness)'

    y, X = pt.dmatrices(formula,sdf)
    Cs = [10**i for i in range(-4,5)]
    X_train, X_test, y_train, y_test = train_test_split(X,y.ravel(),test_size = 0.33,stratify=y.ravel())
    theta = LogisticRegressionCV(cv=5,Cs=Cs,class_weight='balanced').fit(X_train,y_train)
    auc = roc_auc_score(y_test,theta.predict_proba(X_test)[:,1])
    f1 = f1_score(y_test,theta.predict(X_test))
    c = theta.C_
    print('f1 score: {} \n auc: {} \n C: {}'.format(f1,auc,c))

    theta = LogisticRegression(penalty='none',class_weight='balanced').fit(X_train,y_train)
    auc = roc_auc_score(y_test,theta.predict_proba(X_test)[:,1])
    f1 = f1_score(y_test,theta.predict(X_test))
    accuracy = (y_test == theta.predict(X_test)).mean()
    print('f1 score: {} \n auc: {} \n Accuracy: {}'.format(f1,auc,accuracy))
    return(theta)

def estimate_plea_conviction_probability():
    sdf = pd.read_csv(sentencing_data_file)
    pred_cols = ['Incarceration','Black','OffenseType','OffenseSeriousness','JudgeID']
    sdf = sdf.loc[sdf.Plea == 1,pred_cols].dropna()
    formula = 'Incarceration ~ Black + OffenseType + C(OffenseSeriousness) + JudgeID'

    y, X = pt.dmatrices(formula,sdf)
    X_train, X_test, y_train, y_test = train_test_split(X,y.ravel(),test_size = 0.33,stratify=y.ravel())

    theta = LogisticRegression(penalty='none',class_weight='balanced').fit(X_train,y_train)
    auc = roc_auc_score(y_test,theta.predict_proba(X_test)[:,1])
    f1 = f1_score(y_test,theta.predict(X_test))
    accuracy = (y_test == theta.predict(X_test)).mean()
    print('f1 score: {} \n auc: {} \n Accuracy: {}'.format(f1,auc,accuracy))
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
    judge_ids = ['Judge '+str(i) for i in range(1,51)]
    judge_dict = {id:Judge(id,df.loc[df.JudgeID == id],sentence='Sentence') for id in judge_ids}
    df['MinPlea'] = df.apply(get_min_plea,axis=1,args=(judge_dict,))
    df['MaxPlea'] = df.apply(get_max_plea,axis=1,args=(judge_dict,))
    return(df)

def get_min_plea(row,judge_dict):
    judge_id = row['JudgeID']
    judge = judge_dict[judge_id]
    expected_sentence = row['ExpectedTrialSentence']
    min_plea = judge.get_min_plea_ch(expected_sentence)
    return(min_plea)

def get_max_plea(row,judge_dict):
    judge_id = row['JudgeID']
    judge = judge_dict[judge_id]
    expected_sentence = row['ExpectedTrialSentence']
    max_plea = judge.get_max_plea_ch(expected_sentence)
    return(max_plea)

def main():
    tst = csim.Judge()
    # df = make_covariate_matrix()
    # df.to_csv(processed_sentencing_data_file,index=False)
