
########################################################################################
####### model training module:                            ##############################
####### to train an xgboost model with cutoff facility    ##############################
########################################################################################

########################################################################################
####### load libraries                                    ##############################                              
########################################################################################
library(TDA)
require(xgboost)

########################################################################################
####### input file path                                   ##############################                              
########################################################################################
folder_path_in <- "/Users/tieqiangli/@bp3/data/i=2_TDA_L=52x8/"
file_name <- "df_model_input.csv"

folder_path_model <- "/Users/tieqiangli/@bp3/script/backtesting/"
model_name <- "xgb_model_TDA"
########################################################################################
####### codebase: functions                               ##############################                              
########################################################################################

# training, validation, test data portion split
port_tr <- 60
port_va <- 20
port_te <- 20

# model parameters
objective <- 'binary:logistic' # logistic binary classification tast
max_deph <- 5 # depth of the tree
eta <- 1
silent <- 1
nthread <- 4 # the number of cpu threads to use
eval_metric <- 'auc' # area under curve

nrounds <- 20 
    
ModelTrainer <- function(folder_path_in, file_name, # dataframe file path if its from csv to train and test the model
                         DF,                        # dataframe if its from previous function, let it auto-decide
                         port_tr, port_va, port_te, # training, validation and testing portions
                         # model parameters specification
                         objective,
                         max_deph,
                         eta,
                         silent,
                         nthread,
                         eval_metric
                         nrounds)
  {
  # prepare the dtrain and dtest from DF and feed it into xgb.train to 
  # obtain optimally trained model for prediction
  # save in designated folder afterwards
  etf_model_TDA <- xgb.train()
  xgb.save(etf_model_TDA)
}

ModelTester <- function(elt_model_TDA, DF)
  {
  # predict/test the model on disjoined dataset split accordingly
  # call utility function to visulise and understand the results
  pred <- predict(elt_model_TDA, dtest)
}

