
rm(list=ls(all=TRUE))

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
folder_path_in <- "C:/bp3/data/raw/" # "/Users/tieqiangli/@bp3/data/raw/"
folder_path_out <- "C:/bp3/data/i=1_TDA_L=52x8/"  # "/Users/tieqiangli/@bp3/data/i=2_TDA_L=52x8/" # 2nd test at w=52 and n=8
file_name_px <- "data_price_etf.csv"
file_name_des <- "data_static_etf.csv"

###############################################################################################
####### codebase: functions                                                             #######
###############################################################################################

# 1.1 Extract
ExtractorETFPx <- function(folder_path_in, file_name_px){
  # if data stored in a DB, some credentials and DB connection to access the DB, otherwise read.csv
  f <- paste0(folder_path_in,file_name_px)
  df_px <- read.csv(f)
  # data cleansing and filtering here
  return(df_px)
}

# f_des <- ExtractorETFDes(folder_path_in = folder_path_in, file_name_des = file_name_des)
ExtractorETFDes <- function(folder_path_in, file_name_des)
  {
  # if data stored in a DB, some credentials and DB connection to access the DB, otherwise read.csv
  f <- paste0(folder_path_in,file_name_des)
  df_des <- read.csv(f)
  # data cleansing and filtering here
  return(df_des)
}

# 1.2 Transform

w <- 52  # batch window size
s <- 4 # stride size
n <- 8 # number of windows to cut the total data history length
L <- w*n + s + 3 # total length of the historical data in unit of weeks, need additional 4w for y

num_fea <- 14
TH_y <- 0.04 # return cut-off for binary target/classification

TransformDataframeETF <- function(w, s, L, num_fea, TH_y,  # data prep parameters
                              df_px, df_des)      # data source
  {
  # mapping all the descriptors in df_descr into qunatities (according to the mapping excel)
    # 2.0. building quantitative descriptive features
  # loop over the entire data history length of L in the stride size of s and window size of w
  # append the transformed outcome matrix DF_w (loop number k+1) of each loop to the previous one DF_w (loop number k)
  # eventually build the giant model input DF
    # in each loop do the following:
    # 2.1. building normal factor features
    # 2.2. building TDa features
    # 2.3. concatenating and merging all features in the 3 feature categories in one df
  write.csv(df, paste0(folder_path_out,"df_model_input.csv"))
  return(df)
}

# 2. feature engineering

# 2.0. building quantitative descriptive features
FeatureBuilderDes <- function(df_des){
  # mapping the descriptive columns to numeric format and store in a DF 
  # where each row is an ETF and each colomn is a quantified descriptor
  df_des_ <- df_des
  colnames(df_des_)[8] <- 'ISSUER'  
  # 2.0.1 replace missing values by "NA" for the first 3 numeric columns
  df_des_$FUND_EXPENSE_RATIO <- as.numeric(as.character(df_des_$FUND_EXPENSE_RATIO))
  df_des_$FUND_ASSETS_USD <- as.numeric(as.character(df_des_$FUND_ASSETS_USD))
  df_des_$AVERAGE_BID_ASK_SPREAD <- as.numeric(as.character(df_des_$AVERAGE_BID_ASK_SPREAD))
  # 2.0.2 replace binary columns with 0 and 1
    # FUND_LEVERAGE
  levels(df_des_$FUND_LEVERAGE) <- c(levels(df_des_$FUND_LEVERAGE), 0, 1) # adding 0 & 1 to the same level set
  df_des_$FUND_LEVERAGE[df_des_$FUND_LEVERAGE == "N"] <- 0
  df_des_$FUND_LEVERAGE[df_des_$FUND_LEVERAGE == "Y"] <- 1
  df_des_$FUND_LEVERAGE <- as.numeric(as.character(df_des_$FUND_LEVERAGE))
    # INVERSE_FUND_INDICATOR
  levels(df_des_$INVERSE_FUND_INDICATOR) <- c(levels(df_des_$INVERSE_FUND_INDICATOR), 0, 1) # adding 0 & 1 to the same level set
  df_des_$INVERSE_FUND_INDICATOR[df_des_$INVERSE_FUND_INDICATOR == "N"] <- 0
  df_des_$INVERSE_FUND_INDICATOR[df_des_$INVERSE_FUND_INDICATOR == "Y"] <- 1
  df_des_$INVERSE_FUND_INDICATOR <- as.numeric(as.character(df_des_$INVERSE_FUND_INDICATOR))
    # ACTIVELY_MANAGED
  levels(df_des_$ACTIVELY_MANAGED) <- c(levels(df_des_$ACTIVELY_MANAGED), 0, 1) # adding 0 & 1 to the same level set
  df_des_$ACTIVELY_MANAGED[df_des_$ACTIVELY_MANAGED == "N"] <- 0
  df_des_$ACTIVELY_MANAGED[df_des_$ACTIVELY_MANAGED == "Y"] <- 1
  df_des_$ACTIVELY_MANAGED <- as.numeric(as.character(df_des_$ACTIVELY_MANAGED))
  # 2.0.3 dynamically replace other classification columns based on the given list of classes

  # remove zombie rows/funds
  
  return(df_des_)  
}

ClassMapper <- function(df_col, col_names)
{
  l <- unique(df_col)
  l <- l[!grepl("#", l)] # remove the "#N/A" values for the mapping
  n <- length(l)
  mapping <- data.frame(l,seq(1,n,by=1)) 
  colnames(mapping) = c(col_names, paste0(col_names,'_Num'))
  x <- data.frame(df_col)
  colnames(x) <- col_names
  df <- join(x=x,y=mapping)
  df$ISSUER_Num <- as.numeric(as.character(df$ISSUER_Num))
  return(df$ISSUER_Num)
}
  
# 2.1. building normal factor features
FeatureBuilderFactor <- function(df_px) # df_w is the df_price truncated by window size w in each loop 
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
FeatureBuilderTDA <- function(df_px){
  # loop over each etf with the window timeseries as input to compute the persistent landscape
  return(df_TDA)
}

# 2.3. concatenating all features in the 3 feature categories in one DF_w
FeatureAggregator <- function(df_des_, df_factor, df_TDA){
  # remove the non-common ETFs only keep the ones exisiting in all three dataframes
  # merge/right-join by ticker
  return(df)
}

