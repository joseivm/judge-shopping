# Judge Shopping Project

## Table of Contents
- [Overview](#overview)
  - [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Cleaning](#data-cleaning)
  - [Parameter Estimation](#parameter-estimation)
  - [Simulation](#simulation)
- [Matching Procedure](#matching-procedure)

## <a name="overview"/> Overview

This repository contains the code for the Judge Shopping porject. The Judge Shopping project
aims to study the judicial system of South Carolina. This judicial system is particularly of interest because in it, judges rotate across counties. This allows defendants to have some choice with regards to the judge that hears their case. Previous studies have found this system to be more equitable than systems with stricter sentencing guidlines. The purpose of this project is to study how changes to the system would affect sentencing outcomes.  This involves first specifying a model of the system, estimating the relevant parameters, and then simulating the system. 

The main functions of this repository are:
- Cleaning various datasets
- Estimating parameters
- Simulating the system

### <a name="data-sources"/> Data Sources

The main data sources used for this project are:

- Sentencing Data CSV file: this dataset contains information about 17,671 sentencing events in South Carolina from August 2000-July 2001. Each sentencing event contains an identifier for the judge who heard the case, the county the case was heard in, categorical variables describing the offense, and some defendant characteristics (e.g. race, age, and criminal history).
- Sentencing Data STATA file : this dataset contains very similar dataset to the Sentencing Data CSV file. We believe that it is a processed version of the CSV file. Some variables that appear in the CSV file don't appear in this file. We mainly use this file to get Hester's calculations of the expected minimum sentence for each offense. This is the 'expmin' variable in this dataset. 
- Master Calendar Data: this dataset contains information about each judge's schedule for each week of the fiscal year 2001. It has information about what county each judge was assigned to, and what kind of cases they were supposed to be hearing each day. The raw data is spread across 12 different files, one for each month. 

The raw data files corresponding to each of these data sources are:

- Sentencing Data CSV file: `data/raw/Sentencing Data/Hester_Data.csv`
- Sentencing Data STATA file: `data/raw/Sentencing Data/expected_min_sentence.dta`
- Master Calendar Data: `data/raw/Schedule Data/{Month}_{year}.xlsx`

## <a name="installation"/> Installation

To install, first clone the repository using the following command:

```
git clone git@github.com:joseivm/judge-shopping.git
```

Next, create a file called `.env` in the top level of the project directory (i.e. judge-shopping/.env) and add the following line to the file, replacing `/path/to/project` with the absolute path to the project on your local machine.

```
PROJECT_DIR=/path/to/project/judge-shopping
```

Next, create a file called `.env.R` in the top level of the project directory (i.e. judge-shopping/.env.R) and add the following line to the file, replacing `/path/to/project` with the absolute path to the project on your local machine.

```
PROJECT_DIR <- "/path/to/project/judge-shopping"
```

These two files will allow you to run all of the code on your local machine without having to change any filepaths. 

## <a name="usage"/> Usage
Note: all code must be run from the project's home directory.


### <a name="data-cleaning"/> Data Cleaning

The raw data files are located in `data/raw` and the cleaned data files are located in `data/processed`. To clean
the calendar data, run the following command from the project's home directory:

```
python code/data/clean_calendar_data.py
```

The cleaned files it produces are:

1. `weekly_schedule_data.csv`: this file contains an observation for every judge in every week of the year. The unit of observation here is each unique combination of judge, week, and county observed in the calendar data. We only use this dataset to map judge names to judge IDs. More information about this can be found in Section 1.4 of the write-up.
2. `daily_schedule_data.csv`: this file contains an observation for every judge in every week day of the year. The unit of observation here is each unique combination of judge, day, and county observed in the calendar data. This is the file we mainly use when working with the calendar data. 

Next, to clean the sentencing data, run the following command from the project's home directory:

```
python code/data/clean_sentencing_data.py
```

The cleaned files it produces are:

1. `sentencing_data.csv`: this file contains the sentencing data augmented with the expected minimum sentence, judge name, and imputed dates for sentencing events with missing dates. 
2. `judge_name_id_mapping_file.csv`: this file contains the final mapping obtained from our mapping procedure. It maps judge names to judge ID's.  

### <a name="parameter-estimation"/> Parameter Estimation

This section describes the different files used for parameter estimation. At the time of writing, this is still a work in progress, so the files are not finalized. All of the files used for parameter estimation are in the `code/analysis` directory. **Note**: `tau_estimation.R` must be run before `parameter_estimation.py` is run, because the output of the former is the input of the latter. 

1. `capacity_estimation.R`: this file estimates the plea and trial processing rates: $\mu_p$ and $\mu_t$.
2. `tau_estimation.R`: this file estimates $\tau$, the expected sentence length if convicted. It outputs a file called `augmented_sentencing_data.csv` which contains estimates of $\tau$ for each defendant. 
3. `parameter_estimation.py`: this file estimates $\theta$, the probability of conviction at trial, and $u_j(),l_j()$ the maximum/minimum sentence functions for each judge. This file depends on the file `court_simulation.py`, which has the code for creating the convex hull for every judge. 

### <a name="matching"/> Simluation

The code for the simulation is in the file `court_simulation.py`. 
