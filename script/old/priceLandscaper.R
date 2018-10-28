
library(TDA)

######################
## NEW TDA features ##
######################

f_in <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\data_prices_w=13.csv"
f_out <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\TDA_features_w=13_.csv"

f_in <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\data_prices_w=26.csv"
f_out <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\TDA_features_w=26.csv"

f_in <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\data_prices_w=39.csv"
f_out <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\TDA_features_w=39.csv"

f_in <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\data_prices_w=52.csv"
f_out <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\TDA_features_w=52.csv"

f_in <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\data_prices_w=65.csv"
f_out <- "C:\\@ChinaLifeFranklin\\bicycleProject_cont\\data\\TDA\\TDA_features_w=65.csv"

landscaper(f_in, f_out)

landscaper <- function(f_in, f_out){
  df <- read.csv(f_in)
  
  num_col <- dim(df)[2]
  
  matrix_TDA_features <- matrix( ,nrow=5,ncol=num_col)
  colnames(matrix_TDA_features) <- colnames(df)[1:num_col]
  
  for(i in 1:num_col){
    X <- df[colnames(df)[i]]
    Diag <- gridDiag(FUNvalues = X,sublevel = FALSE, printProgress = TRUE)
    for(j in 1:5){
      L <- landscape(Diag[["diagram"]],dimension = 0,KK=j)
      m <- mean(L)
      matrix_TDA_features[j,i] <- m
    }
    pct <- paste(round(100*i/num_col, 2), "%", sep="")
    print(paste("progress =", pct))
    # for(k in 6:10){
    #   L <- landscape(vecDiag[[vecHeader[[i]]]],dimension = 1,KK=k-5)
    #   m <- mean(L)
    #   matrix_TDA_features[k,i] <- m
    # }
  }
  write.csv(matrix_TDA_features,f_out)
}  

#######################
## training features ##
#######################

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_tr\\data_prices.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_tr\\TDA_features.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_tr\\data_prices_1m.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_tr\\TDA_features_1m.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_tr\\data_prices_3m.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_tr\\TDA_features_3m.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_tr\\data_prices_6m.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_tr\\TDA_features_6m.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_tr\\data_prices_1y.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_tr\\TDA_features_1y.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_tr\\data_prices_2y.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_tr\\TDA_features_2y.csv"

######################
## testing features ##
######################

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_te\\data_prices.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_te\\TDA_features.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_te\\data_prices_1m.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_te\\TDA_features_1m.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_te\\data_prices_3m.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_te\\TDA_features_3m.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_te\\data_prices_6m.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_te\\TDA_features_6m.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_te\\data_prices_1y.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_te\\TDA_features_1y.csv"

f_in <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl_te\\data_prices_2y.csv"
f_out <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl_te\\TDA_features_2y.csv"
