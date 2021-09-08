library(data.table)
library(ggplot2)
library(ggthemes)
source('.env.R')

# Input files/dirs
DATA_DIR <- paste0(PROJECT_DIR,'/data/optimization/')

# Output files/dirs
PLOTS_DIR <- paste0(PROJECT_DIR,'/output/figures/Optimization/')

NLL_plots <- function(){
  files <- list.files(DATA_DIR)
  files <- grep('opt_data',files,value=TRUE)
  for (file in files){
    filename <- paste0(DATA_DIR,file)
    df <- fread(filename)
    mu_p <- df$OptMuP[1]
    bf_mu_p <- df$BFMuP[1]
    ggplot(df,aes(MuP,NLL))+geom_line()+geom_vline(xintercept=mu_p)+
      geom_vline(xintercept=bf_mu_p,color='red')+theme_bw()

    plot_filename <- paste0(PLOTS_DIR,file)
    plot_filename <- gsub('.csv','.png',plot_filename)
    ggsave(plot_filename,height=4,width=5)
  }
}

mu_t_plots <- function(){
  files <- list.files(DATA_DIR)
  files <- grep('mut_data',files,value=TRUE)
  for (file in files){
    filename <- paste0(DATA_DIR,file)
    df <- fread(filename)
    ggplot(df[Trial > 0],aes(MuT,Trial))+geom_point()+theme_bw()

    plot_filename <- paste0(PLOTS_DIR,file)
    plot_filename <- gsub('.csv','.png',plot_filename)
    ggsave(plot_filename,height=4,width=5)
  }
}

x_1_plots <- function(){
  filename <- paste0(DATA_DIR,'gaussian_em_data.csv')
  df <- fread(filename)
  df <- melt(df,id.vars='Iteration')
  df <- df[Iteration < 11]
  ggplot(df[variable %in% c('X1','D1')],aes(Iteration,value,Group=variable,color=variable))+
    geom_line()+theme_bw()

  plot_filename <- paste0(PLOTS_DIR,'x1_plot.png')
  ggsave(plot_filename,height=4,width=5)
}

x_2_plots <- function(){
  filename <- paste0(DATA_DIR,'gaussian_em_data.csv')
  df <- fread(filename)
  df <- melt(df,id.vars='Iteration')
  df <- df[Iteration < 11]
  ggplot(df[variable %in% c('X2','D2')],aes(Iteration,value,Group=variable,color=variable))+
    geom_line()+theme_bw()

  plot_filename <- paste0(PLOTS_DIR,'x2_plot.png')
  ggsave(plot_filename,height=4,width=5)
}

NLL_plots()
mu_t_plots()
# x_1_plots()
# x_2_plots()
