library(data.table)
library(VGAM)
source('.env.R')

# Input files/dirs
sentencing_data_file <- paste0(PROJECT_DIR,'/data/processed/sentencing_data.csv')
calendar_data_file <- paste0(PROJECT_DIR,'/data/processed/daily_schedule_data.csv')

# Output files/dirs
augmented_sentencing_data_file <- paste0(PROJECT_DIR,'/data/processed/augmented_sentencing_data.csv')

##### Data Loading #####
load_sentencing_data <- function(){
  df <- fread(sentencing_data_file)
  df[Sentence > 720, Sentence := 720]
  df[, Sentence := as.integer(Sentence)]
  df[OffenseSeriousness == 1, ModifiedOffenseSeriousness := TRUE]
  df[ModifiedOffenseSeriousness == TRUE, OffenseSeriousness := 2]
  df[, OffenseSeriousness := as.factor(OffenseSeriousness)]
  return(df)
}

##### Model fitting #####
create_augmented_data <- function(){
  df <- load_sentencing_data()
  defendant_covars <- c('OffenseType','OffenseSeriousness','Black','CriminalHistory',
  'Sex')

  nb_model <- vglm(Sentence ~ OffenseType + OffenseSeriousness + Black + CriminalHistory + Sex,
    family = posnegbinomial(),data=df[(Sentence > 0) & (Trial == 1)])

  predicted_sentence <- predict(nb_model,df[,..defendant_covars],type='response')
  df[, PredictedSentence := predicted_sentence]
  df[ModifiedOffenseSeriousness == TRUE,OffenseSeriousness := 1]
  fwrite(df,augmented_sentencing_data_file)
}

create_augmented_data()
