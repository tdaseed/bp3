library(TDA)

################################
### 0. data preprocessing
################################

f <- "N:/Risk_and_Performance/Public/Analytics/Risk@Weekly/input.csv"
df <- read.csv(f)

tgt_ind <- 'EQY_SPX'

### meta parameters
w <- 26
s <- 1
d <- 2
date_backtest <- '26/12/2003'
TH <- -0.05

################################
### 1. feature engineering
################################

df_t <- df[1:26,]

feaBuilder <- function(df_t){
  ret_1w_tgt <- df_t[tgt_ind][1,1]/df_t[tgt_ind][2,1]-1
  ret_2q_tgt <- df_t[tgt_ind][1,1]/df_t[tgt_ind][w,1]-1
  ret_1w_DXY <- df_t$FX_DXY[1]/df_t$FX_DXY[2]-1
  ret_2q_DXY <- df_t$FX_DXY[1]/df_t$FX_DXY[w]-1
  ret_1w_Gld <- df_t$CTY_Gold[1]/df_t$CTY_Gold[2]-1
  ret_2q_Gld <- df_t$CTY_Gold[1]/df_t$CTY_Gold[w]-1
  ret_1w_Oil <- df_t$CTY_WTICrude[1]/df_t$CTY_WTICrude[2]-1
  ret_2q_Oil <- df_t$CTY_WTICrude[1]/df_t$CTY_WTICrude[w]-1

  sprd_2y10y <- df_t$BON_USD2Y[1] - df_t$BON_USD10Y[1]
  sprd_2y5y  <- df_t$BON_USD2Y[1] - df_t$BON_USD5Y[1]

  vol_tgt <- sd(df_t[tgt_ind][1:w-1,1]/df_t[tgt_ind][2:w,1]-1)*sqrt(52)
  vol_DXY <- sd(df_t$FX_DXY[1:w-1]/df_t$FX_DXY[2:w]-1)*sqrt(52)
  vol_Gld <- sd(df_t$CTY_Gold[1:w-1]/df_t$CTY_Gold[2:w]-1)*sqrt(52)
  vol_Oil <- sd(df_t$CTY_WTICrude[1:w-1]/df_t$CTY_WTICrude[2:w]-1)*sqrt(52)
  
  tda_tgt <- landscaperNorm(df[tgt_ind])
  tda_DXY <- landscaperNorm(df$FX_DXY)
  tda_Gld <- landscaperNorm(df$CTY_Gold)
  tda_Oil <- landscaperNorm(df$CTY_WTICrude)
  
  fea <- data.frame(ret_1w_tgt,ret_2q_tgt,ret_1w_DXY,ret_2q_DXY,ret_1w_Gld,ret_2q_Gld,ret_1w_Oil,ret_2q_Oil,
                    sprd_2y10y,sprd_2y5y,
                    vol_tgt,vol_DXY,vol_Gld,vol_Oil)
}


landscaperNorm <- funciton(vec){
  # rebase to 100
  # call gridDiag
  # call landscape
  # calc norm at KK=1
  # calc norm at KK=2
}

################################
### 2. model training/validating
################################




################################
### 3. forecasting/backtesting
################################
