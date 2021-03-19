library(data.table)
library(ggplot2)
library(ggthemes)

source('.env.R')

# Input files/dirs
sentencing_data_file <- paste0(PROJECT_DIR,'/data/processed/temp_sentencing_data.csv')
schedule_data_file <- paste0(PROJECT_DIR,'/data/processed/temp_schedule_data.csv')

# Output files/dirs
FIGURES_DIR <- paste0(PROJECT_DIR,'/output/figures/Validation/')

figure_46 <- function(){
  sdf <- fread(sentencing_data_file)
  cdf <- fread(schedule_data_file)

  merge_cols <- c('County','Week')
  overlap <- merge(sdf[,.(JudgeID,County,Week)],cdf[,.(JudgeName,County,Week)],
                        by=merge_cols,allow.cartesian = TRUE)

  overlap_counts <- overlap[,.N,by=.(JudgeID,JudgeName)][order(JudgeID,-N)]
  best_matches <- overlap_counts[,.(Best=N[1],SecondBest=N[2],ThirdBest=N[3]),by=JudgeID]
  best_matches[, JudgeID := gsub('Judge ','',JudgeID)]
  best_matches[, JudgeID := as.numeric(JudgeID)]

  counts <- melt(best_matches,id.vars = 'JudgeID',variable.name = 'Rank')

  fig <- ggplot(counts,aes(JudgeID,value,color=Rank,group=Rank))+geom_line()+
          labs(y='Weeks of Overlap')+theme_bw()

  outfile <- paste0(FIGURES_DIR,'figure_46.png')
  ggsave(outfile,width=6,height=4)
}

figure_46()
