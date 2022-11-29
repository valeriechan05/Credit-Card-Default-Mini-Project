
#Load the Packages

library(ggplot2)
library(naivebayes)
library(Hmisc)
library(caret)
library(e1071)
library(party)
library(kernlab)
library(LiblineaR)
library(nnet)
library(randomForest)
library(mfx)

# Clean the data environment and import the data
rm(list=ls())
de <- read.csv("C:/Users/Desktop/Val/de.csv",stringsAsFactors = FALSE)
View(de)

########################
# EXPLORATORY ANALYSIS
########################
head(de)
summary(de)

# Data cleaning
de$EDUCATION[de$EDUCATION<1|de$EDUCATIO>4]=NA
def=na.omit(de)
def$PAY_0[def$PAY_0==-1]=0
def$PAY_2[def$PAY_2==-1]=0
def$PAY_3[def$PAY_3==-1]=0
def$PAY_0[def$PAY_0==-2]=0
def$PAY_2[def$PAY_2==-2]=0
def$PAY_3[def$PAY_3==-2]=0
summary(def)

# LIMIT_BAL appeared to have a long tail, so I applied a log function to balance the distribution
hist(de$LIMIT_BAL)
de$LIMIT_BAL = log(de$LIMIT_BAL)
hist(de$LIMIT_BAL)

########################
# REGRESSIONS
########################

# Test: if I use Education and Limit of Balance as my predictors
def$EDUCATION =  as.factor(def$EDUCATION)
def$PAY_0 =  as.factor(def$PAY_0)
def$PAY_2 =  as.factor(def$PAY_2)
def$PAY_3 =  as.factor(def$PAY_3)
def$Status_Sep = as.factor(def$Status_Sep)
def$Status_Aug = as.factor(def$Status_Aug)
def$Status_July = as.factor(def$Status_July)
def$default.payment.next.month = as.factor(def$default.payment.next.month)
summary(def)

# Logistic regression 1
split = sample(c("train", "test"), nrow(def), replace=TRUE, prob = c(0.8, 0.2))
train = def[split=="train",]
test = def[split=="test",]
logit_train = glm(default.payment.next.month ~ LIMIT_BAL + EDUCATION + PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug +Status_July, family = "binomial", data = train)
summary(logit_train)
logitmfx(default.payment.next.month ~ LIMIT_BAL + EDUCATION + PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug +Status_July, data=train)

# Generate the predicted probabilities, if P > 0.5 then default should be assigned true
probabilities = predict(logit_train, test, type = "response")
predicted_class = ifelse(probabilities>0.5, 1, 0)

# Confusion matrix & Accuracy
confusion_matrix = table(predicted_class, test$default.payment.next.month) 
confusion_matrix
accuracy_logit = (confusion_matrix[1,1] + confusion_matrix[2,2]) / sum(confusion_matrix)
accuracy_logit

# Probit regression 1
probit_regression = glm(default.payment.next.month ~ LIMIT_BAL + EDUCATION + PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug +Status_July, family = binomial(link="probit"), data = train)
summary(probit_regression)
probitmfx(default.payment.next.month ~ LIMIT_BAL + EDUCATION + PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug +Status_July, data = train)
probabilities2 = predict(probit_regression, test, type = "response")

predicted_class2 = ifelse(probabilities2>0.5, 1, 0)
confusion_matrix2 = table(predicted_class2, test$default.payment.next.month) 
confusion_matrix2

accuracy_probit = (confusion_matrix2[1,1] + confusion_matrix2[2,2]) / sum(confusion_matrix2)
accuracy_probit

# Test 2: If not using EDUCATION and LIMIT_BALANCE
de$PAY_0[de$PAY_0==-1]=0
de$PAY_2[de$PAY_2==-1]=0
de$PAY_3[de$PAY_3==-1]=0
de$PAY_0[de$PAY_0==-2]=0
de$PAY_2[de$PAY_2==-2]=0
de$PAY_3[de$PAY_3==-2]=0

de$PAY_0 =  as.factor(de$PAY_0)
de$PAY_2 =  as.factor(de$PAY_2)
de$PAY_3 =  as.factor(de$PAY_3)
de$Status_Sep = as.factor(de$Status_Sep)
de$Status_Aug = as.factor(de$Status_Aug)
de$Status_July = as.factor(de$Status_July)
de$default.payment.next.month = as.factor(de$default.payment.next.month)
summary(de)

# Logistic regression 2
split2 = sample(c("train", "test"), nrow(de), replace=TRUE, prob = c(0.8, 0.2))
train2 = de[split=="train",]
test2 = de[split=="test",]

logit_train0 = glm( default.payment.next.month ~ PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug +Status_July, family = "binomial", data = train2)
summary(logit_train0)
probabilities0 = predict(logit_train0, test2, type = "response")
predicted_class0 = ifelse(probabilities0>0.5, 1, 0)

confusion_matrix0 = table(predicted_class0, test2$default.payment.next.month) 
confusion_matrix0
accuracy_logit0 = (confusion_matrix0[1,1] + confusion_matrix0[2,2]) / sum(confusion_matrix0)
accuracy_logit0

# Probit regression 2
probit_regression0 = glm(default.payment.next.month ~ PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug +Status_July, family = binomial(link="probit"), data = train2)
summary(probit_regression0)
probitmfx(default.payment.next.month ~ PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug +Status_July, data = train2)

probabilities1 = predict(probit_regression0, test2, type = "response")
predicted_class1 = ifelse(probabilities1>0.5, 1, 0) 

confusion_matrix1 = table(predicted_class1, test2$default.payment.next.month) 
confusion_matrix1

accuracy_probit0 = (confusion_matrix1[1,1] + confusion_matrix1[2,2]) / sum(confusion_matrix1)
accuracy_probit0


###############################
# PREDICTION - CROSS VALIDATION
###############################

library(caret)
set.seed(0)
train_Control = trainControl(method = "cv", number = 10)

# KNN (doesn't work)
knn_caret2 = train(default.payment.next.month ~ EDUCATION + PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug + Status_July, 
                  data = def,
                  method = "knn", trControl = train_Control,
                  tuneLength = 10)
summary(knn_caret2)

tree1 = train(default.payment.next.month ~ EDUCATION + PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug + Status_July, 
                   data = def,
                   method = "ctree2", trControl = train_Control,
                   tuneLength = 10)
summary(tree1)

# Naivebayes model
naivebayes = train(default.payment.next.month ~ EDUCATION + PAY_0 + PAY_2 + PAY_3 + Status_Sep + Status_Aug +Status_July, 
                         data = def, method = "naive_bayes", trControl = train_Control)
naivebayes

# Random Forest
# 1st try
forest1 = train(default.payment.next.month ~ EDUCATION + PAY_0 + PAY_3 + Status_Sep + Status_July,data=def, method = "rf", trControl = train_Control,
              tuneLength = 3)
forest1

# 2nd try
forest2 = train(default.payment.next.month ~ EDUCATION + PAY_0 + PAY_3 + Status_Sep + Status_July,data=def, method = "rf", trControl = train_Control,
                  tuneLength = 5)
forest2
plot(forest2)

# 3rd try
forest3 = train(default.payment.next.month ~ EDUCATION + PAY_0 + PAY_3 + Status_Sep + Status_July,data=def, method = "rf", trControl = train_Control,
                tuneLength = 10)
forest3
plot(forest3)

# SVM linear
set.seed(0)
train_Control2 = trainControl(method = "cv", number = 5)

svm_linear_kernel = train(default.payment.next.month ~ EDUCATION + PAY_0 + PAY_3 + Status_Sep + Status_July, 
                          data = def,
                          method="svmLinear2", trControl=train_Control2,
                          tuneLength=5) 
svm_linear_kernel
