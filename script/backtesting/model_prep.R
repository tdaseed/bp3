
###############################################################################################
####### model preparation module:                                                       #######
####### 1. ELT functions                                                                #######
#######    1.1a extractor/loader of ETF price data (csv or stored in a DB @Bloomberg?)  #######
#######    1.1b extractor/loader of ETF static data (csv or stored in a DB @Bloomberg?) #######
#######    1.2  transformer of the data into xgboost model input format (calling 2.)    #######
#######                                                                                 #######
####### 2. feature engineering functions                                                #######
#######    2.0 descriptive features builder                                             #######
#######    2.1 factor features builder                                                  #######
#######    2.2 TDA feature builder                                                      #######
#######    2.3 feature integrator:combining factors, TDA landscapes and static features #######
###############################################################################################

###############################################################################################
####### load libraries                                                                  #######
###############################################################################################
library(TDA)

###############################################################################################
####### input file path                                                                 #######
###############################################################################################
folder_path_in <- "/Users/tieqiangli/@bp3/data/raw/"
folder_path_out <- "/Users/tieqiangli/@bp3/data/i=2_TDA_L=52x8/" # 2nd test at w=52 and n=8
file_name_price <- "data_price.csv"
file_name_descr <- "data_static_com.csv"

###############################################################################################
####### codebase: functions                                                             #######
###############################################################################################

# 1.1 Extract
ExtractorPriceETF <- function(folder_path_in, file_name_price){
  # if data stored in a DB, some credentials and DB connection to access the DB, otherwise read.csv
  f <- paste0(folder_path_in,file_name_price)
  df_price <- read.csv(f)
  # data cleansing and filtering here
  return(df_price)
}


ExtractorDescrETF <- function(folder_path, file_name_descr)
  {
  # if data stored in a DB, some credentials and DB connection to access the DB, otherwise read.csv
  f <- paste0(folder_path,file_name_price)
  df_descr <- read.csv(f)
  # data cleansing and filtering here
  return(df_descr)
}

# 1.2 Transform

w <- 52  # batch window size
s <- 4 # stride size
n <- 8 # number of windows to cut the total data history length
L <- w*n + s + 3 # total length of the historical data in unit of weeks, need additional 4w for y

num_fea <- 14
TH_y <- 0.04 # return cut-off for binary target/classification

TransformDataframeETF <- function(w, s, L, num_fea, TH_y,  # data prep parameters
                              df_price, df_descr)      # data source
  {
  # mapping all the descriptors in df_descr into qunatities (according to the mapping excel)
    # 2.0. building quantitative descriptive features
  # loop over the entire data history length of L in the stride size of s and window size of w
  # append the transformed outcome matrix DF_w (loop number k+1) of each loop to the previous one DF_w (loop number k)
  # eventually build the giant model input DF
    # in each loop do the following:
    # 2.1. building normal factor features
    # 2.2. building TDa features
    # 2.3. concatenating all features in the 3 feature categories in one df
  write.csv(DF, paste0(folder_path_out,"df_model_input.csv"))
  return(DF)
}

# 2. feature engineering

# 2.0. building quantitative descriptive features
FeatureBuilderDescriptor <- function(df_descr){
  # mapping the descriptive columns to numeric format and store in a DF 
  # where each row is an ETF and each colomn is a quantified descriptor
  return(DF_descr)  
}
  
# 2.1. building normal factor features
FeatureBuilderFactor <- function(df_w) # df_w is the df_price truncated by window size w in each loop 
  {
  # build the following target and normal features and store in a df of 
  # dimension = e x 14 where e is the number of ETFs in each df_w, and 14 = num_fea
    # y
    # # r_4w
    # r_8w 
    # r_13w 
    # r_26w 
    # r_52w 
    # h_l_4w 
    # h_l_8w 
    # h_l_13w 
    # h_l_26w     
    # h_l_52w 
    # std_13w 
    # std_26w 
    # std_52w 
  return(df_factor)
}

# 2.2. building TDa features
FeatureBuilderTDA <- function(df_w){
  # loop over each etf with the window timeseries as input to compute the persistent landscape
  return(df_TDA)
}

# 2.3. concatenating all features in the 3 feature categories in one DF_w
FeatureAggregator <- function(DF_descr, df_factor, df_TDA){
  # remove the non-common ETFs only keep the ones exisiting in all three dataframes
  # merge/right-join by ticker
  return(DF_w)
}

