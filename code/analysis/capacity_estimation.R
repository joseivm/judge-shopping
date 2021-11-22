library(data.table)
source('.env.R')

# Input files/dirs
processed_daily_schedule_file <- paste0(PROJECT_DIR,'/data/processed/daily_schedule_data.csv')
processed_sentencing_data_file <- paste0(PROJECT_DIR,'/data/processed/sentencing_data.csv')

holidays <- c('2000-09-04','2000-10-09','2000-11-10','2000-11-23','2000-12-24',
'2000-12-25','2000-12-26','2001-01-01','2001-01-15','2001-02-19','2001-05-10',
'2001-05-28','2001-07-04','2000-07-04')

load_sentencing_data <- function(){
  sdf <- fread(processed_sentencing_data_file)
  sdf[, Circuit := as.character(Circuit)]
  return(sdf)
}

load_calendar_data <- function(){
  cdf <- fread(processed_daily_schedule_file)
  mapping <- fread(judge_name_id_mapping_file)
  cdf <- merge(cdf,mapping,by='JudgeName')
  cdf <- cdf[!(Date %in% holidays)]
  return(cdf)
}

sdf <- load_sentencing_data()
cdf <- load_calendar_data()

pleas <- sdf[WorkType == 'GS',.(Pleas=sum(Plea)),by=.(JudgeID,County)]
trials <- sdf[,.(Trials=sum(Trial)),by=.(JudgeID,County)]
days <- cdf[WorkType == 'GS',.(Days=sum(Days)),by=.(JudgeID,County)]

data <- merge(pleas,trials,by=c('JudgeID','County'))
data <- merge(data,days,by=c('JudgeID','County'))

model <- lm(Days ~ Pleas + Trials + County + 0,data=data)
