##########################################################
# Install and load rpackages
##########################################################
if (!require("tidyverse")) {
  install.packages("tidyverse")
  library(tidyverse)
}
if (!require("caret")) {
  install.packages("caret")
  library(caret)
}
if (!require("data.table")) {
  install.packages("data.table")
  library(data.table)
}
if (!require("dplyr")) {
  install.packages("dplyr")
  library(dplyr)
}
if (!require("readr")) {
  install.packages("readr")
  library(readr)
}
if (!require("ggplot2")) {
  install.packages("ggplot2")
  library(ggplot2)
}
if (!require("klaR")) {
  install.packages("klaR")
  library(klaR)
}
if (!require("randomForest")){
  install.packages("randomForest")
  library(randomForest)
}
if (!require("gam")){
  install.packages("gam")
  library(gam)
}
if (!require("gmodels")){
  install.packages("gmodels")
  library(gmodels)
}
if (!require("gridExtra")){
  install.packages("gridExtra")
  library(gridExtra)
}
if (!require("cowplot")){
  install.packages("cowplot")
  library(cowplot)
}
if (!require("rlang")){
  install.packages("rlang")
  library(rlang)
}
if (!require("ggpubr")){
  install.packages("ggpubr")
  library(ggpubr)
}


##########################################################
# Generate bio dataset
##########################################################
# Biomechanical features of orthopedic patients dataset:
# Original Link: https://www.kaggle.com/uciml/biomechanical-features-of-orthopedic-patients
# Github Link: https://raw.githubusercontent.com/ciaranboyle/edx-capstone/main/orthopedic_data.csv

orthopedic_data <- 
  read_csv(url("https://raw.githubusercontent.com/ciaranboyle/edx-capstone/main/orthopedic_data.csv"))


##########################################################
# Generate Training and Test Datasets
##########################################################
#generate training and test sets
set.seed(28, sample.kind="Rounding")
partition_index <- createDataPartition(y = orthopedic_data$class, times = 1, p = 0.7, list=FALSE)
orthopedic_training <- orthopedic_data[partition_index,]
orthopedic_test <- orthopedic_data[-partition_index,]

# Clean up environment
rm(partition_index)


##########################################################
# Basic Examination
##########################################################
str(orthopedic_data, give.attr = FALSE) #view structure of dataset
orthopedic_data %>% group_by(class) %>% summarize(count=n()) #observations per bone class in dataset

#refactor the data
orthopedic_data$class <- factor(orthopedic_data$class, 
                                levels = c("Normal", "Hernia", "Spondylolisthesis"))
orthopedic_training$class <- factor(orthopedic_training$class, 
                                levels = c("Normal", "Hernia", "Spondylolisthesis"))
orthopedic_test$class <- factor(orthopedic_test$class, 
                                levels = c("Normal", "Hernia", "Spondylolisthesis"))

##########################################################
# LDA
##########################################################
set.seed(28, sample.kind="Rounding")

#running the LDA algorithm on the training set
train_lda <- train(class ~ ., 
                   method = "lda", 
                   data = orthopedic_training)

#running the trained algorithm on the test set
confusionMatrix(predict(train_lda, orthopedic_test),
                orthopedic_test$class)$overall["Accuracy"]

##########################################################
# RDA
##########################################################
set.seed(28, sample.kind="Rounding")

#running the RDA algorithm on the training set
train_rda <- train(class ~ ., 
                   method = "rda", 
                   data = orthopedic_training)

#running the trained algorithm on the test set
confusionMatrix(predict(train_rda, orthopedic_test), 
                orthopedic_test$class)$overall["Accuracy"]

##########################################################
# KNN
##########################################################
set.seed(28, sample.kind="Rounding")

#running the KNN algorithm on the training set
train_knn <- train(class ~ ., 
                   method = "knn", 
                   data = orthopedic_training)

#running the trained algorithm on the test set
confusionMatrix(predict(train_knn, orthopedic_test), 
                orthopedic_test$class)$overall["Accuracy"]

##########################################################
# KNN (Tuned)
##########################################################
set.seed(28, sample.kind="Rounding")

#running the KNN algorithm (tuned) on the training set
train_knn2 <- train(class ~ ., 
                   method = "knn", 
                   data = orthopedic_training,
                   tuneGrid = data.frame(k=seq(3,51,2)))

#running the trained algorithm on the test set
confusionMatrix(predict(train_knn2, orthopedic_test), 
                orthopedic_test$class)$overall["Accuracy"]

##########################################################
# Decision trees
##########################################################
set.seed(28, sample.kind="Rounding")

#running the Decision trees algorithm on the training set
train_rpart <- train(class ~ .,
                     method = "rpart",
                     data = orthopedic_training)

#running the trained algorithm on the test set
confusionMatrix(predict(train_rpart, orthopedic_test), 
                orthopedic_test$class)$overall["Accuracy"]

##########################################################
# Random forests
##########################################################
set.seed(28, sample.kind="Rounding")

#running the Random forests algorithm on the training set
train_rf1 <- train(class ~ .,
                   method = "rf",
                   data = orthopedic_training)

#running the trained algorithm on the test set
confusionMatrix(predict(train_rf1, orthopedic_test), 
                orthopedic_test$class)$overall["Accuracy"]

