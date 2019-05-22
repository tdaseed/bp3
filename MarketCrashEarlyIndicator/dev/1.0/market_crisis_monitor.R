rm(list=ls(all=TRUE))

library(TDA)
library(dplyr)
library(xgboost)
library(scales)
library(data.table)
library(ggplot2)
library(stringr)
library(ROCR)
library(ggplot2)


f_out <- "C:/HyperParameter_FineTune_SPX/"
f_out <- "C:/HyperParameter_FineTune_UKX/"

f_out <- "C:/bp3/MarketCrashEarlyIndicator/dev/1.0/results/201904/"

###
### define a function to handle the backtesting
###
backtester <- function(
  df, tgt_ind,                    # input dataset
  w, s, d, n, TH,                 # data preprocessing parameters
  Use_TDA,                        # swith on/off TDA features
  nrounds, max_depth,             # model parameters  
  f_out                           # output folder path
)  
{
  X_tr <- data.frame()
  Y_tr <- data.frame()
  
  if (Use_TDA == TRUE)
  {
    for (i in seq(1, l-w-s, s))
    {
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
  } else 
  {
    for (i in seq(1, l-w-s, s))
    {
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
  }
  
  L <- dim(df)[1] # total time length (# of weeks) of the raw data
  k <- L - l # remaining weeks
  
  print("preprocessing DONE!")
  
  # -w to cover the last round of the loop
  # -d to leave the forecast d weeks disjoint from X
  # -s to leave the stride s do its last move
  date_test_end <- L-w-d-s
  
  Y_date  <- data.frame()
  Y_true  <- data.frame()
  Y_pred  <- data.frame()
  Y_feat  <- data.frame(colnames(X_tr))
  colnames(Y_feat) <- "Feature"
  
  print("online trainig and backtesting START!")
  
  if (Use_TDA == TRUE)
  {
    for (i in seq(l-1, date_test_end ,s)){
      j <- i+w-1
      #  print(j)
      
      ####
      #### re-calibrate the model with new weeks of training data
      ####
      fea <- feaBuilder_TDA(df[i:j,])
      X_tr <- rbind(X_tr, fea) # appending the features of each batch/datapoint as a new row
      
      # replace the TDA features by its changes over the previous period
      l_X <- dim(X_tr)[1]
      X_tr$tda_tgt[2:l_X] <- X_tr$tda_tgt[2:l_X]/X_tr$tda_tgt[1:l_X-1]-1 # replace the tda norm level by the change of norms 
      X_tr$tda_DXY[2:l_X] <- X_tr$tda_DXY[2:l_X]/X_tr$tda_DXY[1:l_X-1]-1 # replace the tda norm level by the change of norms 
      X_tr$tda_Oil[2:l_X] <- X_tr$tda_Oil[2:l_X]/X_tr$tda_Oil[1:l_X-1]-1 # replace the tda norm level by the change of norms 
      X_tr <- X_tr[2:l_X, ] # remove the 1st row of the training dataset
      
      ret <- df[tgt_ind][j+d,1]/df[tgt_ind][j,1]-1
      y <- labeler(ret)
      Y_tr <- rbind(Y_tr,y) # appending the binary y of each batch/datapoint as a label vector entry
      
      Y_tr <- data.frame(Y_tr[2:l_X, ]) # remove the 1st row of the label dataset
      
      
      print(percent((j-l)/(date_test_end-l)))
      
      ####
      #### re-train the model with the data from new weeks
      ####
      bst <- modelTrainer(X_tr, Y_tr, nrounds, max_depth)
      
      ####
      #### build the test x and true y, and record the predition y in each loop
      ####
      x_test <- feaBuilder_TDA(df[i+d:j+d,])
      
      x_test_1 <- feaBuilder_TDA(df[(i+d-s):(j+d-s),])
      x_test$tda_tgt <- x_test$tda_tgt/x_test_1$tda_tgt -1
      x_test$tda_DXY <- x_test$tda_DXY/x_test_1$tda_DXY -1
      x_test$tda_Oil <- x_test$tda_Oil/x_test_1$tda_Oil -1
      
      ret_true <- df[tgt_ind][j+d+d,1]/df[tgt_ind][j+d,1]-1
      
      # y_date <- df$Date[j]
      # Y_date <- rbind(Y_date, as.Date(y_date, "%m/%d/%Y"))
      
      y_true <- labeler(ret_true)
      Y_true <- rbind(Y_true, y_true)
      
      y_pred <- predict(bst, as.matrix(x_test))
      Y_pred <- rbind(Y_pred, y_pred)
      
      fea_importance <- xgb.importance (feature_names = colnames(X_tr),model = bst)
      fea_importance <- cbind.data.frame(fea_importance$Feature, fea_importance$Gain)
      colnames(fea_importance) <- cbind("Feature", paste("Importance",toString(i)))
      Y_feat <- merge(x=Y_feat, y=fea_importance, by="Feature", all.x=TRUE)
    }
  } else
  {
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
      bst <- modelTrainer(X_tr, Y_tr, nrounds, max_depth)
      
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
      
      fea_importance <- xgb.importance (feature_names = colnames(X_tr),model = bst)
      fea_importance <- cbind.data.frame(fea_importance$Feature, fea_importance$Gain)
      colnames(fea_importance) <- cbind("Feature", paste("Importance",toString(i)))
      Y_feat <- merge(x=Y_feat, y=fea_importance, by="Feature", all.x=TRUE)
    }
  }
  print("backtesting DONE!")
  
  pred <- prediction(Y_pred,Y_true)
  auc <- performance(pred,measure = "auc")
  # auc <- perf@y.values
  
  ### name the file suffixed with auc score
  if (Use_TDA == TRUE)
  {f_out <- paste(f_out, "output_auc=", percent(auc@y.values[[1]]),
                  "_md=", toString(max_depth), "_n=", toString(nrounds), "_",
                  toString(w), "w_", str_sub(tgt_ind,-3,-1), 
                  "_TDA.csv", sep = "")}
  else
  {f_out <- paste(f_out, "output_auc=", percent(auc@y.values[[1]]),
                  "_md=", toString(max_depth), "_n=", toString(nrounds), "_",
                  toString(w), "w_", str_sub(tgt_ind,-3,-1), 
                  ".csv", sep = "")}
  # plot(perf)
  
  start <- d + l-1+w-1 #l-1
  end <- d + j # j
  Y_date <- df$Date[start:end]
  #Y_price <- df[tgt_ind][start:end,]
  crisis_pred <- cbind.data.frame(Y_date,Y_pred, Y_true, t(Y_feat[,2:dim(Y_feat)[2]]))
  colnames(crisis_pred)[2] <- "Y_pred"
  colnames(crisis_pred)[3] <- "Y_true"
  colnames(crisis_pred)[4:dim(crisis_pred)[2]] <- as.character(Y_feat$Feature)
  write.csv(crisis_pred,f_out, row.names=FALSE)
  return(crisis_pred)
}

###
### loop over different model parameters to compare - TDA
###
for (md in seq(3,10,1)){
  for (nrounds in seq(5,50,5)){
    # print(paste("max_depth=", toString(md)), "nrounds=", toString(nrounds))
    backtester(df=df, tgt_ind=tgt_ind, w=26, s=1, d=2, n=20, TH=-0.05, nrounds=nrounds, Use_TDA=FALSE, max_depth=md, f_out=f_out)
  }
}

###
### loop over different model parameters to compare + TDA
###
for (md in seq(3,10,1)){
  for (nrounds in seq(5,50,5)){
    # print(paste("max_depth=", toString(md)), "nrounds=", toString(nrounds))
    backtester(df=df, tgt_ind=tgt_ind, w=26, s=1, d=2, n=20, TH=-0.05, nrounds=nrounds, Use_TDA=TRUE, max_dep=md, f_out=f_out)
  }
}


################################
### 0. data preprocessing
################################

f <- "N:/Risk_and_Performance/Public/Analytics/Risk@Weekly/dev/1.0/input.csv"
f <- "N:/Risk_and_Performance/Public/Analytics/Risk@Weekly/dev/1.0/input_FF.csv"

f <- "C:/bp3/MarketCrashEarlyIndicator/dev/1.0/input.csv"
f <- "C:/bp3/MarketCrashEarlyIndicator/dev/1.0/input_FF.csv"
f <- "C:/bp3/MarketCrashEarlyIndicator/dev/1.0/input_FTSE.csv"
#f <- "/Users/tieqiangli/@crisis_monitor/input.csv"

f <- "C:/bp3/MarketCrashEarlyIndicator/dev/1.0/input_201904.csv"

df <- read.csv(f)
df <- arrange(df, -row_number())

tgt_ind <- 'EQY_HSI' # 'EQY_SPX'#  
tgt_ind <- 'EQY_SPX' # 'EQY_HSI'#  
tgt_ind <- 'EQY_UKX'   

### meta parameters
w <- 26
s <- 1
d <- 2
n <- 20 #4 # number of batches before backtesting
l <- n*w
# date_backtest <- '26/12/2003'
TH <-  -0.05 #  -0.1

#ts <- rev(df$Date)
X_tr <- data.frame()
Y_tr <- data.frame()

Use_TDA <- FALSE # TRUE
Use_TDA <- TRUE # FALSE 

f_out <- "C:/bp3/MarketCrashEarlyIndicator/dev/1.0/results/misc/"

# f_out <- "N:/Risk_and_Performance/Public/Analytics/Risk@Weekly/Backtest/"
# f_out <- "N:/Risk_and_Performance/Public/Analytics/Risk@Weekly/Backtest/20190109/"
# 
# f_out <- "C:/Users/TanL/Documents/N/Risk_and_Performance/Public/Analytics/Risk@Weekly/Backtest/"
# f_out <- "C:/Users/TanL/Documents/N/Risk_and_Performance/Public/Analytics/Risk@Weekly/dev/1.0/results/"

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
for (i in seq(1, 1302, s)){
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
for (i in seq(1, 1302, s)){
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

################################
### 1. feature engineering
################################

# df_t <- df[1:26,]

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
  # ret_1w_FF_Rm_Rf <- df_t$FF_Rm_Rf[w]/df_t$FF_Rm_Rf[w-1]-1
  # ret_1w_FF_SMB <- df_t$FF_SMB[w]/df_t$FF_SMB[w-1]-1
  # ret_1w_FF_HML <- df_t$FF_HML[w]/df_t$FF_HML[w-1]-1
  
  # 1-norm of the landscape of each market driver
  # tda_tgt <- landscaperNorm(df_t[tgt_ind])
  # tda_DXY <- landscaperNorm(df_t["FX_DXY"])
  # #  tda_Gld <- landscaperNorm(df_t["CTY_Gold"])
  # tda_Oil <- landscaperNorm(df_t["CTY_WTICrude"])
  
  fea <- data.frame(ret_1w_tgt,ret_1w_DXY,ret_1w_Gld,ret_1w_Oil,
                    ret_2q_tgt,ret_2q_DXY,ret_2q_Oil,#ret_2q_Gld,
                    sprd_2y10y,sprd_2y5y,
                    # ret_1w_FF_Rm_Rf, ret_1w_FF_SMB, ret_1w_FF_HML,
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
  # ret_1w_FF_Rm_Rf <- df_t$FF_Rm_Rf[w]/df_t$FF_Rm_Rf[w-1]-1
  # ret_1w_FF_SMB <- df_t$FF_SMB[w]/df_t$FF_SMB[w-1]-1
  # ret_1w_FF_HML <- df_t$FF_HML[w]/df_t$FF_HML[w-1]-1
  
  # 1-norm of the landscape of each market driver
  tda_tgt <- landscaperNorm(df_t[tgt_ind])
  tda_DXY <- landscaperNorm(df_t["FX_DXY"])
  #  tda_Gld <- landscaperNorm(df_t["CTY_Gold"])
  tda_Oil <- landscaperNorm(df_t["CTY_WTICrude"])
  
  fea <- data.frame(ret_1w_tgt,ret_1w_DXY,ret_1w_Gld,ret_1w_Oil,
                    ret_2q_tgt,ret_2q_DXY,ret_2q_Oil,#ret_2q_Gld,
                    sprd_2y10y,sprd_2y5y,
                    vol_tgt,vol_DXY,vol_Oil,#vol_Gld,
                    # ret_1w_FF_Rm_Rf, ret_1w_FF_SMB, ret_1w_FF_HML,
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

modelTrainer <- function(X,Y, nrounds, max_depth){
  params <- list(booster = "gbtree", objective = "binary:logistic", max_depth = max_depth, early.stop.round = 3,
                 eta = 0.3, nthread = 4, verbose = 0, eval_metric = "error", seed = 123,
                 gamma = 0, min_child_weight = 1, subsample = 1, colsample_bytree = 1)
  dtrain <- xgb.DMatrix(data = as.matrix(X), label = as.matrix(Y))
  bst <- xgboost(data = dtrain, nrounds = nrounds, params = params)
  return(bst)
}

# nrounds <- 10
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

max_depth <- 5 # 3 # 5
nrounds <- 10 # 5 #30
max_depth <- 3 # 5
nrounds <- 5 #30

Y_date  <- data.frame()
Y_true  <- data.frame()
Y_pred  <- data.frame()
Y_feat  <- data.frame(colnames(X_tr))
colnames(Y_feat) <- "Feature"


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
  
  # replace the TDA features by its changes over the previous period
  l_X <- dim(X_tr)[1]
  X_tr$tda_tgt[2:l_X] <- X_tr$tda_tgt[2:l_X]/X_tr$tda_tgt[1:l_X-1]-1 # replace the tda norm level by the change of norms 
  X_tr$tda_DXY[2:l_X] <- X_tr$tda_DXY[2:l_X]/X_tr$tda_DXY[1:l_X-1]-1 # replace the tda norm level by the change of norms 
  X_tr$tda_Oil[2:l_X] <- X_tr$tda_Oil[2:l_X]/X_tr$tda_Oil[1:l_X-1]-1 # replace the tda norm level by the change of norms 
  X_tr <- X_tr[2:l_X, ] # remove the 1st row of the training dataset
  
  ret <- df[tgt_ind][j+d,1]/df[tgt_ind][j,1]-1
  y <- labeler(ret)
  Y_tr <- rbind(Y_tr,y) # appending the binary y of each batch/datapoint as a label vector entry
  Y_tr <- data.frame(Y_tr[2:l_X, ]) # remove the 1st row of the label dataset
  
  print(percent((j-l)/(date_test_end-l)))
  
  ####
  #### re-train the model with the data from new weeks
  ####
  bst <- modelTrainer(X_tr, Y_tr, nrounds, max_depth)
  
  ####
  #### build the test x and true y, and record the predition y in each loop
  ####
  x_test <- feaBuilder_TDA(df[(i+d):(j+d),])
  x_test_1 <- feaBuilder_TDA(df[(i+d-s):(j+d-s),])
  x_test$tda_tgt <- x_test$tda_tgt/x_test_1$tda_tgt -1
  x_test$tda_DXY <- x_test$tda_DXY/x_test_1$tda_DXY -1
  x_test$tda_Oil <- x_test$tda_Oil/x_test_1$tda_Oil -1
  
  ret_true <- df[tgt_ind][j+d+d,1]/df[tgt_ind][j+d,1]-1
  
  # y_date <- df$Date[j]
  # Y_date <- rbind(Y_date, as.Date(y_date, "%m/%d/%Y"))
  
  y_true <- labeler(ret_true)
  Y_true <- rbind(Y_true, y_true)
  
  y_pred <- predict(bst, as.matrix(x_test))
  Y_pred <- rbind(Y_pred, y_pred)
  
  fea_importance <- xgb.importance (feature_names = colnames(X_tr),model = bst)
  fea_importance <- cbind.data.frame(fea_importance$Feature, fea_importance$Gain)
  colnames(fea_importance) <- cbind("Feature", paste("Importance",toString(i)))
  Y_feat <- merge(x=Y_feat, y=fea_importance, by="Feature", all.x=TRUE)
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
  bst_xTDA <- modelTrainer(X_tr, Y_tr, nrounds, max_depth)
  
  ####
  #### build the test x and true y, and record the predition y in each loop
  ####
  x_test <- feaBuilder(df[i+d:j+d,])
  ret_true <- df[tgt_ind][j+d+d,1]/df[tgt_ind][j+d,1]-1
  
  # y_date <- df$Date[j]
  # Y_date <- rbind(Y_date, as.Date(y_date, "%m/%d/%Y"))
  
  y_true <- labeler(ret_true)
  Y_true <- rbind(Y_true, y_true)
  
  y_pred <- predict(bst_xTDA, as.matrix(x_test))
  Y_pred <- rbind(Y_pred, y_pred)
  
  
  fea_importance <- xgb.importance (feature_names = colnames(X_tr),model = bst_xTDA)
  fea_importance <- cbind.data.frame(fea_importance$Feature, fea_importance$Gain)
  colnames(fea_importance) <- cbind("Feature", paste("Importance",toString(i-L+2)))
  Y_feat <- merge(x=Y_feat, y=fea_importance, by="Feature", all.x=TRUE)
}


start <- d + l-1+w-1 #l-1
end <- d + j
# end <- start+k-4 -2
Y_date <- df$Date[start:end]
#Y_price <- df[tgt_ind][start:end,]
crisis_pred <- cbind.data.frame(Y_date,Y_pred)
crisis_pred <- cbind.data.frame(Y_date,Y_pred, Y_true, t(Y_feat[,2:dim(Y_feat)[2]]))
colnames(crisis_pred)[4:dim(crisis_pred)[2]] <- as.character(Y_feat$Feature)
colnames(crisis_pred)[2] <- "Y_pred"
colnames(crisis_pred)[3] <- "Y_true"


library(ROCR)
library(ggplot2)
pred <- prediction(Y_pred,Y_true)
auc <- performance(pred,measure = "auc")
print(auc@y.values)
acc <- performance(pred,measure = "acc")
err <- performance(pred,measure = "err")
# print(acc@y.values)
perf <- performance(pred,measure = "tpr", x.measure = "fpr")
# plot(perf)
prec <- performance(pred, measure = 'prec')
rec <- performance(pred, measure = 'rec')
fpr <- performance(pred, measure = 'fpr')
fnr <- performance(pred, measure = 'fnr')
mat <- performance(pred, measure = 'mat')

f <- performance(pred, measure = "f")


pred_xTDA <- prediction(Y_pred,Y_true)
auc_xTDA <- performance(pred_xTDA,measure = "auc")
print(auc_xTDA@y.values)
acc_xTDA <- performance(pred_xTDA,measure = "acc")
err_xTDA <- performance(pred_xTDA,measure = "err")
# print(acc_xTDA@y.values)
# plot(acc_xTDA)
perf_xTDA <- performance(pred_xTDA,measure = "tpr", x.measure = "fpr")
# plot(perf_xTDA)
prec_xTDA <- performance(pred_xTDA, measure = 'prec')
rec_xTDA <- performance(pred_xTDA, measure = 'rec')
fpr_xTDA <- performance(pred_xTDA, measure = 'fpr')
fnr_xTDA <- performance(pred_xTDA, measure = 'fnr')
mat_xTDA <- performance(pred_xTDA, measure = 'mat')

f_xTDA <- performance(pred_xTDA, measure = "f")

legend1 <- paste('model with TDA, auc=', toString(percent(auc@y.values[[1]])), sep = '')
legend2 <- paste('model without TDA, auc=', toString(percent(auc_xTDA@y.values[[1]])), sep = '')
plot(perf, col='red')
plot(perf_xTDA, add = TRUE, col='blue')
legend(0.4,0.2,legend=c(legend1,legend2),col=c('red','blue'),lwd=3)

plot(f_xTDA, legend=TRUE,col='blue')
plot(f, col='red',legend=TRUE, add = TRUE)
legend(0.1,0.1,legend=c('model with TDA','model without TDA'),col=c('red','blue'),lwd=3)
# plot(f, add = TRUE, colorize = TRUE)

# plot(prec_xTDA, legend=TRUE,col='blue')
# plot(prec, col='red',legend=TRUE, add = TRUE)
# legend(0.1,0.1,legend=c('model with TDA','model without TDA'),col=c('red','blue'),lwd=3)
# 
# plot(rec_xTDA, legend=TRUE,col='blue')
# plot(rec, col='red',legend=TRUE, add = TRUE)
# legend(0.1,0.1,legend=c('model with TDA','model without TDA'),col=c('red','blue'),lwd=3)
# 
# plot(fpr_xTDA, legend=TRUE,col='blue')
# plot(fpr, col='red',legend=TRUE, add = TRUE)
# legend(0.5, 1,legend=c('model with TDA','model without TDA'),col=c('red','blue'),lwd=3)
# 
# plot(fnr_xTDA, legend=TRUE,col='blue')
# plot(fnr, col='red',legend=TRUE, add = TRUE)
# legend(0.5, 0.2,legend=c('model with TDA','model without TDA'),col=c('red','blue'),lwd=3)
# 
# plot(acc_xTDA, legend=TRUE,col='blue')
# plot(acc, col='red',legend=TRUE, add = TRUE)
# legend(0.5, 0.2,legend=c('model with TDA','model without TDA'),col=c('red','blue'),lwd=3)
# 
# plot(err_xTDA, legend=TRUE,col='blue')
# plot(err, col='red',legend=TRUE, add = TRUE)
# legend(0.5, 0.2,legend=c('model with TDA','model without TDA'),col=c('red','blue'),lwd=3)

# plot(mat_xTDA, legend=TRUE,col='blue')
# plot(mat, col='red',legend=TRUE, add = TRUE)
# legend(0.5, 0.25,legend=c('model with TDA','model without TDA'),col=c('red','blue'),lwd=3)


library(DiagrammeR)
xgb.plot.tree(model=bst,trees=nrounds-1)
xgb.plot.tree(model=bst_xTDA,trees=nrounds-1)
xgb.plot.multi.trees(model=bst_xTDA)
xgb.plot.multi.trees(model=bst)
xgb.plot.tree(model=bst,trees=0)

gr <- xgb.plot.tree(model=bst,trees=nrounds-1)
export_graph(gr, 'tree.png')

###
### save the results
###
f_out <- "C:/bp3/MarketCrashEarlyIndicator/dev/1.0/results/"

if (Use_TDA == TRUE)
  {f_out <- paste(f_out, "output_auc=", percent(auc@y.values[[1]]),
                  "_md=", toString(max_depth), "_n=", toString(nrounds), "_",
                  toString(w), "w_", str_sub(tgt_ind,-3,-1), 
                  "_TDA.csv", sep = "")} else
  {f_out <- paste(f_out, "output_auc=", percent(auc_xTDA@y.values[[1]]),
                  "_md=", toString(max_depth), "_n=", toString(nrounds), "_",
                  toString(w), "w_", str_sub(tgt_ind,-3,-1), 
                  ".csv", sep = "")}

write.csv(crisis_pred,f_out, row.names=FALSE)

Y_date_tr <- df$Date[27:j]
X_tr_out <- cbind.data.frame(Y_date_tr,X_tr)
f_out_tr <- "C:/bp3/MarketCrashEarlyIndicator/dev/1.0/results/output_SPX_TDA_tr.csv"
write.csv(X_tr,f_out_tr, row.names=FALSE)
