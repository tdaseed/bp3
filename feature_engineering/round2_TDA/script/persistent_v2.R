library("TDA")

inputFolder <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\round2_TDA\\chartsPH_input_tmp\\"
outputFolder <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\round2_TDA\\chartsPH_output_tmp\\"

inputFolder <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\input_pl\\"
outputFolder <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl\\"

inputFolder <- "C:/Users/tan.li/Documents/Investment/Projects/bicycleProject_cont/round2_TDA/chartsPH_input_tmp/"
outputFolder <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\round2_TDA\\chartsPH_output_tmp\"

fileNames <- dir(inputFolder, pattern="*.csv")

maxdimension <- 1
maxscale <- 0.01
maxscale <- 0.1
maxscale <- 0.06

# mappingFile <- read.csv(paste0(inputFolder,"\\tmp\\mappingTable.csv"),sep="\t",header = FALSE)

setwd(inputFolder)

# placefolders for bottleneck matrix input
vecHeader <- list()
vecDiag <- list()

for(i in 1:length(fileNames)){
  stock <- read.csv(fileNames[i], header=FALSE, row.name = 1, sep=",")
  
  ticker <- substr(fileNames[i],1,nchar(fileNames[i])-4)
  
#  rowNumer <- which(mappingFile$V1 == ticker)
#  sector <- mappingFile$V2[rowNumer]
  
#  sector_ticker <- paste0(sector, "_", ticker)
  
  # storing the tickers as header names for the matrix
  vecHeader <- c(vecHeader, ticker)
  
  Diag <- ripsDiag(stock, maxdimension, maxscale, library = "Dionysus",location = TRUE, printProgress = TRUE)
  # storing the Diag in a list for bottleneck matrix after the charting
  vecDiag[[ticker]] <- Diag[["diagram"]]
  
  # outputFilePD <- paste(outputFolder, "PD_", ticker,".png", sep = "")
  # outputFileBarcode <- paste(outputFolder, "Barcode_", ticker,".png", sep = "")
  # 
  # png(outputFilePD)
  # plot(Diag[["diagram"]], main = paste(ticker, "Persistent Diagram", sep=" "), rotate=TRUE)
  # dev.off()
  # 
  # png(filename=outputFileBarcode)
  # plot(Diag[["diagram"]], main = paste(ticker, "Barcode", sep=" "), rotate=TRUE, barcode = TRUE)
  # dev.off()
} 

plot(Diag[["diagram"]], main = "Persistent Diagram", rotate=TRUE)
plot(Diag[["diagram"]], main = "Barcode", rotate=TRUE, barcode = TRUE)

# moving on to bottleneck matrix computation
bottleneckMatrix_dim0 <- matrix( ,nrow=length(vecDiag),ncol=length(vecDiag))
bottleneckMatrix_dim1 <- matrix( ,nrow=length(vecDiag),ncol=length(vecDiag))
dimnames(bottleneckMatrix_dim0) <- list(vecHeader,vecHeader)
dimnames(bottleneckMatrix_dim1) <- list(vecHeader,vecHeader)

for(i in 1:length(vecDiag)){
  for(j in i:length(vecDiag)){
    bottleneckMatrix_dim0[i,j] <- bottleneck(vecDiag[[i]],vecDiag[[j]],0)
    bottleneckMatrix_dim1[i,j] <- bottleneck(vecDiag[[i]],vecDiag[[j]],1)
    
  } 
}

matrix_TDA_features <- matrix( ,nrow=10,ncol=length(vecDiag))
colnames(matrix_TDA_features) <- vecHeader

for(i in 1:length(vecDiag)){
  for(j in 1:5){
    L <- landscape(vecDiag[[vecHeader[[i]]]],dimension = 0,KK=j)
    m <- mean(L)
    matrix_TDA_features[j,i] <- m
  }
  for(k in 6:10){
    L <- landscape(vecDiag[[vecHeader[[i]]]],dimension = 1,KK=k-5)
    m <- mean(L)
    matrix_TDA_features[k,i] <- m
  }
}

f <- "C:\\Users\\tan.li\\Documents\\Investment\\Projects\\bicycleProject_cont\\feature_engineering\\output_pl\\TDA_features.csv"
write.csv(matrix_TDA_features,f)

# drawPersistentCharts <- function(file) {
#   stock <- read.csv(x, header=FALSE, row.name = 1, sep=",")
#   Diag <- ripsDiag(stock, maxdimension, maxscale, library = "Dionysus",location = TRUE, printProgress = TRUE)  
#   
#   png(outputFilePD)
#   plot(Diag[["diagram"]], main = paste(ticker, "Persistent Diagram", sep=" "), rotate=TRUE)
#   dev.off()
#   
#   png(filename=outputFileBarcode)
#   plot(Diag[["diagram"]], main = paste(ticker, "Barcode", sep=" "), rotate=TRUE, barcode = TRUE)
#   dev.off()
# }
# 
# lapply(files, drawPersistentCharts)