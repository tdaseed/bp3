library(TDA)
library(dplyr)
library(xgboost)
library(scales)
library(data.table)
################################
### 0. data preprocessing
################################

f <- "N:/Risk_and_Performance/Public/Analytics/Risk@Weekly/dev/1.0/input.csv"
f <- "N:/Risk_and_Performance/Public/Analytics/Risk@Weekly/dev/1.0/input_FF.csv"
#f <- "/Users/tieqiangli/@crisis_monitor/input.csv"
  
df <- read.csv(f)
df <- arrange(df, -row_number())

tgt_ind <- 'EQY_SPX'  # 'EQY_HSI'

### meta parameters
w <- 26
s <- 1
d <- 2
n <- 20 #4 # number of batches before backtesting
l <- n*w
# date_backtest <- '26/12/2003'
TH <- -0.05

#ts <- rev(df$Date)
X_tr <- data.frame()
Y_tr <- data.frame()

###
### w TDA features
###
for (i in seq(1, l-w-s, s)){
#  print(i)
  j <- i+w-1
#  print(j)
  fea <- feaBuilder_TDA(df[i:j,])
  X_tr <- rbind(X_tr, fea) # appending the features of each batch/datapoint as a new row
  ret <- df[tgt_ind][j+d,1]/df[tgt_ind][j,1]-1
  y <- labeler(ret)
  # if (ret < TH){
  #   y <- 1
  # } else {
  #   y = 0
  # }
  Y_tr <- rbind(Y_tr,y) # appending the binary y of each batch/datapoint as a label vector entry
  print(percent(j/l))
#  print(df$Date[j])
}

###
### x TDA features
###
for (i in seq(1, l-w-s, s)){
  #  print(i)
  j <- i+w-1
  #  print(j)
  fea <- feaBuilder(df[i:j,])
  X_tr <- rbind(X_tr, fea) # appending the features of each batch/datapoint as a new row
  ret <- df[tgt_ind][j+d,1]/df[tgt_ind][j,1]-1
  y <- labeler(ret)
  # if (ret < TH){
  #   y <- 1
  # } else {
  #   y = 0
  # }
  Y_tr <- rbind(Y_tr,y) # appending the binary y of each batch/datapoint as a label vector entry
  print(percent(j/l))
  #  print(df$Date[j])
}

#X_tr = as.matrix(X_tr)
#Y_tr = as.matrix(Y_tr)

################################
### 1. feature engineering
################################

df_t <- df[1:26,]

labeler <- function(ret){
  if (ret < TH){
    y <- 1
  } else {
    y = 0
  }
  return(y)
}

feaBuilder <- function(df_t){
  # 1w return
  ret_1w_tgt <- df_t[tgt_ind][w,1]/df_t[tgt_ind][w-1,1]-1
  ret_1w_DXY <- df_t$FX_DXY[w]/df_t$FX_DXY[w-1]-1
  ret_1w_Gld <- df_t$CTY_Gold[w]/df_t$CTY_Gold[w-1]-1
  ret_1w_Oil <- df_t$CTY_WTICrude[w]/df_t$CTY_WTICrude[w-1]-1
  
  # full window reutnr (w = 2 quarters in our case)
  ret_2q_tgt <- df_t[tgt_ind][w,1]/df_t[tgt_ind][1,1]-1  
  ret_2q_DXY <- df_t$FX_DXY[w]/df_t$FX_DXY[1]-1  
  #  ret_2q_Gld <- df_t$CTY_Gold[w]/df_t$CTY_Gold[1]-1  
  ret_2q_Oil <- df_t$CTY_WTICrude[w]/df_t$CTY_WTICrude[1]-1
  
  # term structure spread
  sprd_2y10y <- df_t$BON_USD2Y[w] - df_t$BON_USD10Y[w]
  sprd_2y5y  <- df_t$BON_USD2Y[w] - df_t$BON_USD5Y[w]
  
  # vol of each market driver
  vol_tgt <- sd(df_t[tgt_ind][2:w,1]/df_t[tgt_ind][1:w-1,1]-1)*sqrt(52)
  vol_DXY <- sd(df_t$FX_DXY[2:w]/df_t$FX_DXY[1:w-1]-1)*sqrt(52)
  #  vol_Gld <- sd(df_t$CTY_Gold[2:w]/df_t$CTY_Gold[1:w-1]-1)*sqrt(52)
  vol_Oil <- sd(df_t$CTY_WTICrude[2:w]/df_t$CTY_WTICrude[1:w-1]-1)*sqrt(52)

  # Fama French 3 factors  
  ret_1w_FF_Rm_Rf <- df_t$FF_Rm_Rf[w]/df_t$FF_Rm_Rf[w-1]-1
  ret_1w_FF_SMB <- df_t$FF_SMB[w]/df_t$FF_SMB[w-1]-1
  ret_1w_FF_HML <- df_t$FF_HML[w]/df_t$FF_HML[w-1]-1
  
  # 1-norm of the landscape of each market driver
  # tda_tgt <- landscaperNorm(df_t[tgt_ind])
  # tda_DXY <- landscaperNorm(df_t["FX_DXY"])
  # #  tda_Gld <- landscaperNorm(df_t["CTY_Gold"])
  # tda_Oil <- landscaperNorm(df_t["CTY_WTICrude"])
  
  fea <- data.frame(ret_1w_tgt,ret_1w_DXY,ret_1w_Gld,ret_1w_Oil,
                    ret_2q_tgt,ret_2q_DXY,ret_2q_Oil,#ret_2q_Gld,
                    sprd_2y10y,sprd_2y5y,
                    ret_1w_FF_Rm_Rf, ret_1w_FF_SMB, ret_1w_FF_HML,
                    vol_tgt,vol_DXY,vol_Oil) #vol_Gld,
                    # tda_tgt,tda_DXY,tda_Oil)#tda_Gld)
  return(fea)
}

feaBuilder_TDA <- function(df_t){
  # 1w return
  ret_1w_tgt <- df_t[tgt_ind][w,1]/df_t[tgt_ind][w-1,1]-1
  ret_1w_DXY <- df_t$FX_DXY[w]/df_t$FX_DXY[w-1]-1
  ret_1w_Gld <- df_t$CTY_Gold[w]/df_t$CTY_Gold[w-1]-1
  ret_1w_Oil <- df_t$CTY_WTICrude[w]/df_t$CTY_WTICrude[w-1]-1
  
  # full window reutnr (w = 2 quarters in our case)
  ret_2q_tgt <- df_t[tgt_ind][w,1]/df_t[tgt_ind][1,1]-1  
  ret_2q_DXY <- df_t$FX_DXY[w]/df_t$FX_DXY[1]-1  
#  ret_2q_Gld <- df_t$CTY_Gold[w]/df_t$CTY_Gold[1]-1  
  ret_2q_Oil <- df_t$CTY_WTICrude[w]/df_t$CTY_WTICrude[1]-1
  
  # term structure spread
  sprd_2y10y <- df_t$BON_USD2Y[w] - df_t$BON_USD10Y[w]
  sprd_2y5y  <- df_t$BON_USD2Y[w] - df_t$BON_USD5Y[w]

  # vol of each market driver
  vol_tgt <- sd(df_t[tgt_ind][2:w,1]/df_t[tgt_ind][1:w-1,1]-1)*sqrt(52)
  vol_DXY <- sd(df_t$FX_DXY[2:w]/df_t$FX_DXY[1:w-1]-1)*sqrt(52)
#  vol_Gld <- sd(df_t$CTY_Gold[2:w]/df_t$CTY_Gold[1:w-1]-1)*sqrt(52)
  vol_Oil <- sd(df_t$CTY_WTICrude[2:w]/df_t$CTY_WTICrude[1:w-1]-1)*sqrt(52)

  # Fama French 3 factors  
  ret_1w_FF_Rm_Rf <- df_t$FF_Rm_Rf[w]/df_t$FF_Rm_Rf[w-1]-1
  ret_1w_FF_SMB <- df_t$FF_SMB[w]/df_t$FF_SMB[w-1]-1
  ret_1w_FF_HML <- df_t$FF_HML[w]/df_t$FF_HML[w-1]-1
  
  # 1-norm of the landscape of each market driver
  tda_tgt <- landscaperNorm(df_t[tgt_ind])
  tda_DXY <- landscaperNorm(df_t["FX_DXY"])
#  tda_Gld <- landscaperNorm(df_t["CTY_Gold"])
  tda_Oil <- landscaperNorm(df_t["CTY_WTICrude"])
  
  fea <- data.frame(ret_1w_tgt,ret_1w_DXY,ret_1w_Gld,ret_1w_Oil,
                    ret_2q_tgt,ret_2q_DXY,ret_2q_Oil,#ret_2q_Gld,
                    sprd_2y10y,sprd_2y5y,
                    vol_tgt,vol_DXY,vol_Oil,#vol_Gld,
                    ret_1w_FF_Rm_Rf, ret_1w_FF_SMB, ret_1w_FF_HML,
                    tda_tgt,tda_DXY,tda_Oil)#tda_Gld)
  return(fea)
}


landscaperNorm <- function(ts){
  # rebase to 100
  ret <- ts[2:w,1]/ts[1:w-1,1]-1
  ret <- rev(ret)
  ret[w] <- 0 # adding the dummy return at t_0
  x <- 100*cumprod(rev(ret) + 1)
  # call gridDiag
  Diag <- gridDiag(FUNvalues = x,sublevel = FALSE, printProgress = FALSE)
  # call landscape
  Lmin <- min(Diag[['diagram']][,2:3])
  Lmax <- max(Diag[['diagram']][,2:3])
  tseq <- seq(Lmin, Lmax, length = 50)
    # calc norm at KK=2  
  l2 <- landscape(Diag[["diagram"]],dimension = 0,KK=2, tseq = tseq)
  n2 <- norm(l2, type = "O") # specifies the one norm
  # calc norm at KK=3  
#   l3 <- landscape(Diag[["diagram"]],dimension = 0,KK=3, tseq = tseq)
#   n3 <- norm(l3, type = "O")
  return(n2)
}

################################
### 2. model training/validating
################################

modelTrainer <- function(X,Y, nrounds){
  params <- list(booster = "gbtree", objective = "binary:logistic", max_depth = 3, 
                 eta = 0.3, silent = 1, nthread = 4, verbose = 1, eval_metric = "auc",
                 gamma = 0, min_child_weight = 1, subsample = 1, colsample_bytree = 1)
  dtrain <- xgb.DMatrix(data = as.matrix(X), label = as.matrix(Y))
  bst <- xgboost(data = dtrain, nrounds = nrounds, params = params)
  return(bst)
}

# nrounds <- 20
# bst <- modelTrainer(X_tr, Y_tr, nrounds)

fea_importance <- xgb.importance (feature_names = colnames(X_tr),model = bst)
xgb.plot.importance (importance_matrix = fea_importance,xlab = "Feature Importance (wTDA)")


################################
### 3. forecasting/backtesting
################################

L <- dim(df)[1] # total time length (# of weeks) of the raw data
k <- L - l # remaining weeks

# -w to cover the last round of the loop
# -d to leave the forecast d weeks disjoint from X
# -s to leave the stride s do its last move
date_test_end <- L-w-d-s  

nrounds <- 20

Y_date  <- data.frame()
Y_true  <- data.frame()
Y_pred  <- data.frame()

###################
###### w TDA ######
###################
for (i in seq(l-1, date_test_end ,s)){
  j <- i+w-1
  #  print(j)
  
  ####
  #### re-calibrate the model with new weeks of training data
  ####
  fea <- feaBuilder_TDA(df[i:j,])
  X_tr <- rbind(X_tr, fea) # appending the features of each batch/datapoint as a new row
  ret <- df[tgt_ind][j+d,1]/df[tgt_ind][j,1]-1
  y <- labeler(ret)
  Y_tr <- rbind(Y_tr,y) # appending the binary y of each batch/datapoint as a label vector entry
  
  print(percent((j-l)/(date_test_end-l)))
  
  ####
  #### re-train the model with the data from new weeks
  ####
  bst <- modelTrainer(X_tr, Y_tr, nrounds)
  
  ####
  #### build the test x and true y, and record the predition y in each loop
  ####
  x_test <- feaBuilder_TDA(df[i+d:j+d,])
  ret_true <- df[tgt_ind][j+d+d,1]/df[tgt_ind][j+d,1]-1
  
  # y_date <- df$Date[j]
  # Y_date <- rbind(Y_date, as.Date(y_date, "%m/%d/%Y"))
  
  y_true <- labeler(ret_true)
  Y_true <- rbind(Y_true, y_true)
  
  y_pred <- predict(bst, as.matrix(x_test))
  Y_pred <- rbind(Y_pred, y_pred)
}

###################
###### x TDA ######
###################
for (i in seq(l-1, date_test_end ,s)){
  j <- i+w-1
  #  print(j)
  
  ####
  #### re-calibrate the model with new weeks of training data
  ####
  fea <- feaBuilder(df[i:j,])
  X_tr <- rbind(X_tr, fea) # appending the features of each batch/datapoint as a new row
  ret <- df[tgt_ind][j+d,1]/df[tgt_ind][j,1]-1
  y <- labeler(ret)
  Y_tr <- rbind(Y_tr,y) # appending the binary y of each batch/datapoint as a label vector entry
  
  print(percent((j-l)/(date_test_end-l)))
  
  ####
  #### re-train the model with the data from new weeks
  ####
  bst <- modelTrainer(X_tr, Y_tr, nrounds)
  
  ####
  #### build the test x and true y, and record the predition y in each loop
  ####
  x_test <- feaBuilder(df[i+d:j+d,])
  ret_true <- df[tgt_ind][j+d+d,1]/df[tgt_ind][j+d,1]-1

  # y_date <- df$Date[j]
  # Y_date <- rbind(Y_date, as.Date(y_date, "%m/%d/%Y"))
  
  y_true <- labeler(ret_true)
  Y_true <- rbind(Y_true, y_true)
  
  y_pred <- predict(bst, as.matrix(x_test))
  Y_pred <- rbind(Y_pred, y_pred)
}


start <- l-1+w-1 #l-1
end <- j
# end <- start+k-4 -2
Y_date <- df$Date[start:end]
crisis_pred <- cbind.data.frame(Y_date,Y_pred)
f_out <- "N:/Risk_and_Performance/Public/Analytics/Risk@Weekly/Backtest/output_5_wTDA_SPX_FF.csv"
write.csv(crisis_pred,f_out)


library(ROCR)
library(ggplot2)
pred <- prediction(Y_pred,Y_true)
perf <- performance(pred,"tpr","fpr")
plot(perf)

