
library(TDA)

######################
## NEW TDA features ##
######################

# i = 1
folder_tr <- "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/TDA/tr/"
folder_te <- "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/TDA/te/"
folder_va <- "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/TDA/va/"

folder_out <- "/Users/tieqiangli/@bp3/data/i=1_TDA_L=52x8/TDA/"

GenerateFeatureTDA(folder_tr=folder_tr, folder_te=folder_te,folder_va = folder_va, folder_out = folder_out)

# TDA features for training dataset
GenerateFeatureTDA <- function(folder_tr, folder_te, folder_va, folder_out){
  #matrix_TDA_tr <- matrix(, nrow = 0, ncol = 6)
  #matrix_TDA_tr <- matrix(, nrow = 0, ncol = 11)
  matrix_TDA_tr <- matrix(, nrow = 0, ncol = 101)
  #matrix_TDA_tr <- matrix(, nrow = 0, ncol = 201)
  #colnames(matrix_TDA_tr) <- c('ticker','dim_0_KK_1','dim_0_KK_2','dim_0_KK_3','dim_0_KK_4','dim_0_KK_5')
  #colnames(matrix_TDA_tr) <- c('ticker','norm_l1_KK_1','norm_l1_KK_2','norm_l1_KK_3','norm_l1_KK_4','norm_l1_KK_5',
  #                             'norm_l2_KK_1','norm_l2_KK_2','norm_l2_KK_3','norm_l2_KK_4','norm_l2_KK_5')
  
  n <- (length(list.files(folder_tr)) + length(list.files(folder_te)) + length(list.files(folder_va)))
  t=1 # recording the number of TDA files processed
  for(f in list.files(folder_tr)){
    path = paste0(folder_tr,f)
#    print(path)
#    M = landscaper(path, t)
#    M = landscaperFULL(path, t)
    M = landscaperFULL(path)
    matrix_TDA_tr = rbind(matrix_TDA_tr,M)
    pct <- paste(round(100*t/n, 1), "%", sep="")
    print(paste("progress =", pct))
    t = t+1
  }
  file_out_tr <- paste0(folder_out,"TDA_features_tr.csv")
  write.csv(matrix_TDA_tr,file_out_tr)
  
  # TDA features for testing dataset
  #matrix_TDA_te <- matrix(, nrow = 0, ncol = 6)
  #matrix_TDA_te <- matrix(, nrow = 0, ncol = 11)
  matrix_TDA_te <- matrix(, nrow = 0, ncol = 101)
  #matrix_TDA_te <- matrix(, nrow = 0, ncol = 201)
  #colnames(matrix_TDA_te) <- c('ticker','dim_0_KK_1','dim_0_KK_2','dim_0_KK_3','dim_0_KK_4','dim_0_KK_5')
  #colnames(matrix_TDA_te) <- c('ticker','norm_l1_KK_1','norm_l1_KK_2','norm_l1_KK_3','norm_l1_KK_4','norm_l1_KK_5',
  #                             'norm_l2_KK_1','norm_l2_KK_2','norm_l2_KK_3','norm_l2_KK_4','norm_l2_KK_5')
  #t=1 # recording the number of TDA files processed
  for(f in list.files(folder_te)){
    path = paste0(folder_te,f)
    #print(path)
#    M = landscaper(path, t)
#    M = landscaperFULL(path, t)
    M = landscaperFULL(path)
    matrix_TDA_te = rbind(matrix_TDA_te,M)
    pct <- paste(round(100*t/n, 1), "%", sep="")
    print(paste("progress =", pct))
    t = t+1
  }
  file_out_te <- paste0(folder_out,"TDA_features_te.csv")
  write.csv(matrix_TDA_te,file_out_te)
  
  # TDA features for validation dataset
  #matrix_TDA_va <- matrix(, nrow = 0, ncol = 6)
  #matrix_TDA_va <- matrix(, nrow = 0, ncol = 11)
  matrix_TDA_va <- matrix(, nrow = 0, ncol = 101)
  #matrix_TDA_va <- matrix(, nrow = 0, ncol = 201)
  #colnames(matrix_TDA_va) <- c('ticker','dim_0_KK_1','dim_0_KK_2','dim_0_KK_3','dim_0_KK_4','dim_0_KK_5')
  #colnames(matrix_TDA_va) <- c('ticker','norm_l1_KK_1','norm_l1_KK_2','norm_l1_KK_3','norm_l1_KK_4','norm_l1_KK_5',
  #                             'norm_l2_KK_1','norm_l2_KK_2','norm_l2_KK_3','norm_l2_KK_4','norm_l2_KK_5')
  #t=1 # recording the number of TDA files processed
  for(f in list.files(folder_va)){
    path = paste0(folder_va,f)
    #print(path)
#    M = landscaper(path, t)
#    M = landscaperFULL(path, t)
    M = landscaperFULL(path)
    matrix_TDA_va = rbind(matrix_TDA_va,M)
    pct <- paste(round(100*t/n, 1), "%", sep="")
    print(paste("progress =", pct))
    t = t+1
  }
  file_out_va <- paste0(folder_out,"TDA_features_va.csv")
  write.csv(matrix_TDA_va,file_out_va)
}
    
#landscaper(f_in, f_out)

landscaper <- function(file_in, t){
  df <- read.csv(file_in)
  
  num_row <- dim(df)[2]
  
  #matrix_TDA_features <- matrix( ,nrow=num_row,ncol=6)
  matrix_TDA_features <- matrix( ,nrow=num_row,ncol=11)
  matrix_TDA_features[,1] <- colnames(df)[1:num_row]
  
  for(i in 1:num_row){
    X <- df[colnames(df)[i]]
    Diag <- gridDiag(FUNvalues = X,sublevel = FALSE, printProgress = FALSE)
#    Diag <- gridDiag(FUNvalues = X,sublevel = FALSE, printProgress = TRUE)
    for(j in 2:6){
      L <- landscape(Diag[["diagram"]],dimension = 0,KK=j-1)
      #m <- mean(L)
      l1 <- norm(L, type="1")
      matrix_TDA_features[i,j] <- l1
    }
    for(k in 7:11){
      L <- landscape(Diag[["diagram"]],dimension = 0,KK=k-6)
#      L <- landscape(vecDiag[[vecHeader[[i]]]],dimension = 1,KK=k-5)
      l2 <- norm(L, type="2")
      matrix_TDA_features[i,k] <- l2
    }
    pct <- paste(round(100*i/num_row, 1), "%", sep="")
    str1 = paste(t, "batch(es), progress =")
    print(paste(str1, pct))
  }
  return(matrix_TDA_features)
#  write.csv(matrix_TDA_features,f_out)
}  

landscaperFULL <- function(file_in){
  df <- read.csv(file_in)
  
  num_row <- dim(df)[2]
  
  #matrix_TDA_features <- matrix( ,nrow=num_row,ncol=6)
  matrix_TDA_features <- matrix( ,nrow=num_row,ncol=101)
  #matrix_TDA_features <- matrix( ,nrow=num_row,ncol=201)
  matrix_TDA_features[,1] <- colnames(df)[1:num_row]
  
  for(i in 1:num_row){
    X <- df[colnames(df)[i]]
    Diag <- gridDiag(FUNvalues = X,sublevel = FALSE, printProgress = FALSE)
    #    Diag <- gridDiag(FUNvalues = X,sublevel = FALSE, printProgress = TRUE)
    Lmin <- min(Diag[['diagram']][,2:3])
    Lmax <- max(Diag[['diagram']][,2:3])
    tseq <- seq(Lmin, Lmax, length = 50)
    L2 <- landscape(Diag[["diagram"]],dimension = 0,KK=2, tseq = tseq)
    matrix_TDA_features[i,2:51] = L2
    L3 <- landscape(Diag[["diagram"]],dimension = 0,KK=3, tseq = tseq)
    matrix_TDA_features[i,52:101] = L3
    #L4 <- landscape(Diag[["diagram"]],dimension = 0,KK=3, tseq = tseq)
    #matrix_TDA_features[i,102:151] = L4
    #L5 <- landscape(Diag[["diagram"]],dimension = 0,KK=3, tseq = tseq)
    #matrix_TDA_features[i,152:201] = L5
    # pct <- paste(round(100*i/num_row, 1), "%", sep="")
    # str1 = paste(t, "batch(es), progress =")
    # print(paste(str1, pct))
  }
  return(matrix_TDA_features)
  #  write.csv(matrix_TDA_features,f_out)
}  

landscape_plotter <- function(f){
  df <- read.csv(f)
  
  num_row <- dim(df)[2]
  
  for(i in 2:num_row){
    X <- df[colnames(df)[i]]
    Diag <- gridDiag(FUNvalues = X,sublevel = FALSE, printProgress = FALSE)
    #    Diag <- gridDiag(FUNvalues = X,sublevel = FALSE, printProgress = TRUE)
    L <- landscape(Diag[["diagram"]],dimension = 0,KK=1)
    tseq <- seq(min(Diag[['diagram']][,2:3]), max(Diag[['diagram']][,2:3]), length = 500)
    plot(tseq, L, type='l')
    title(colnames(df)[i])
#    print(paste("progress = ", pct))
  }
  #  write.csv(matrix_TDA_features,f_out)
}  


X <- df$GVI
Diag <- gridDiag(FUNvalues = X,sublevel = FALSE, printProgress = FALSE)
Lmin <- min(Diag[['diagram']][,2:3])
Lmax <- max(Diag[['diagram']][,2:3])
tseq <- seq(Lmin, Lmax, length = 50)
L <- landscape(Diag[["diagram"]],dimension = 0,KK=2, tseq=tseq)
L1 <- landscape(Diag[["diagram"]],dimension = 0,KK=1)
L2 <- landscape(Diag[["diagram"]],dimension = 0,KK=2)
L3 <- landscape(Diag[["diagram"]],dimension = 0,KK=3)
L4 <- landscape(Diag[["diagram"]],dimension = 0,KK=4)
L5 <- landscape(Diag[["diagram"]],dimension = 0,KK=5)
plot(Diag[["diagram"]])
plot(tseq,L, type='l')
plot(tseq,L1, type='l')
plot(tseq,L2, type='l')
plot(tseq,L3, type='l')
plot(tseq,L4, type='l')
plot(tseq,L5, type='l')
