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

# TODO: test everything, add code for collecting/calculating metrics, consider adding tqdm thing
# add check for judge capacity at the end of each time period

# Input files/dirs
sentencing_data_file = PROJECT_DIR + '/data/processed/sentencing_data.csv'

# Output files/dirs
simulation_results_dir = PROJECT_DIR + '/data/simulation/'

class Judge:
    def __init__(self,ID,judge_df):
        self.ID = ID
        self.past_pleas = judge_df[['ExpectedTrialSentence','Sentence']].to_numpy()
        self.capacity = np.ones(50)*11
        self.current_week = 0

    def get_min_plea(self,expected_trial_sentence):
        prior_expected_sentences = self.past_pleas[:,0]
        distances = abs(prior_expected_sentences - expected_trial_sentence)
        closest_indices = np.argsort(distances)[:6]
        closest_points = self.past_pleas[closest_indices,:]
        min_plea = min(closest_points[:,1])
        return(min_plea)

    def get_max_plea(self,expected_trial_sentence):
        prior_expected_sentences = self.past_pleas[:,0]
        distances = abs(prior_expected_sentences - expected_trial_sentence)
        closest_indices = np.argsort(distances)[:6]
        closest_points = self.past_pleas[closest_indices,:]
        min_plea = max(closest_points[:,1])
        return(min_plea)

    def hear_case(self,weeks_from_now):
        case_week = self.current_week + weeks_from_now
        self.capacity[case_week] -= 1

    def time_step(self):
        self.current_week += 1

    def is_available(self,weeks_from_now):
        case_week = self.current_week + weeks_from_now
        return self.capacity[case_week] > 0

class County:
    def __init__(self,name,judges,arrival_rate,past_defendants):
        self.name = name
        self.judges = judges.to_numpy()
        self.backlog = np.array([])
        self.arrival_rate = arrival_rate
        self.past_defendants = past_defendants.to_numpy().ravel()
        self.current_week = 0
        self.sentences = pd.DataFrame()

    def get_defendants(self):
        num_defendants = np.random.poisson(self.arrival_rate)
        new_defendants = np.random.choice(self.past_defendants,num_defendants)
        # all_defendants = np.concatenate([self.backlog,new_defendants])
        all_defendants = new_defendants
        return all_defendants

    def add_backlog(self,defendant):
        self.backlog = np.append(self.backlog,defendant)

    def add_sentence(self,judge,defendant,sentence):
        self.sentences = self.sentences.append({'ExpectedTrialSentence':defendant,'Sentence':sentence,
                                'JudgeID':judge,'County':self.name},ignore_index=True)

    def time_step(self):
        self.current_week += 1

    def get_judges(self,choice_window):
        start_idx = self.current_week
        end_idx = min(start_idx + choice_window,len(self.judges)-1)
        return self.judges[start_idx:end_idx]

    def get_results(self):
        results_df = self.sentences
        results_df['Backlog'] = len(self.backlog)
        return(results_df)

def run_simulation(df,time_periods,choice_windows):
    judge_ids = ['Judge '+str(i) for i in range(1,51)]
    weekly_sentences = df.groupby(['County','Week']).size().reset_index(name='N')
    county_lambdas = weekly_sentences.groupby('County')['N'].mean()
    county_names = np.sort(df['County'].unique())

    for choice_window in choice_windows:
        print(choice_window)
        calendar = make_calendar(judge_ids,county_names,time_periods)
        judges = {id:Judge(id,df.loc[df.JudgeID == id]) for id in judge_ids}
        counties = {county: County(county,calendar[county],county_lambdas[county],df.loc[df.County==county,['ExpectedTrialSentence']])
                                                    for county in county_names}
        for period in range(time_periods):
            if period % 10 == 0: print(period)
            for name, county in counties.items():
                defendants = county.get_defendants()
                current_judges = county.get_judges(choice_window)
                for defendant in defendants:
                    available_judges = [(i,judge) for (i,judge) in enumerate(current_judges)
                                                if judges[judge].is_available(i)]
                    if len(available_judges) == 0:
                        county.add_backlog(defendant)
                    else:
                        judge_costs = [calculate_cost(judges[judge],defendant,i)
                                        for (i,judge) in available_judges]
                        sentences = [sentence for (cost,sentence) in judge_costs]
                        costs = [cost for (cost,sentence) in judge_costs]
                        min_cost_idx = np.argmin(costs)
                        delay, min_cost_judge = available_judges[min_cost_idx]
                        judges[min_cost_judge].hear_case(delay)
                        sentence = sentences[min_cost_idx]
                        county.add_sentence(min_cost_judge,defendant,sentence)

            [county.time_step() for (name,county) in counties.items()]
            [judge.time_step() for (name,judge) in judges.items()]

        results = [county.get_results() for (name,county) in counties.items()]
        results = pd.concat(results)
        results.to_csv(simulation_results_dir+'simulation_{}'.format(choice_window),index=False)

def calculate_cost(judge,defendant,weeks_from_now):
    trial_cost = 17
    delay_cost = 0.1
    max_plea = judge.get_max_plea(defendant)
    sentence = min(defendant + trial_cost,max_plea)
    cost =  sentence + weeks_from_now*delay_cost
    return cost, sentence

def make_calendar(judges,counties,time_periods):
    calendar = pd.DataFrame(columns=counties)
    num_counties = len(counties)
    for t in range(time_periods):
        active_judges = np.random.choice(judges,size=num_counties,replace=False)
        schedule = {county: active_judges[i] for (i,county) in enumerate(counties)}
        calendar = calendar.append(schedule,ignore_index=True)
    return(calendar)

def main():
    df = pd.read_csv(sentencing_data_file)
    choice_windows = [2,4,6,8,10]
    time_periods = 50
    run_simulation(df,time_periods,choice_windows)
