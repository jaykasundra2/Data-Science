# load the required packages
rm(list=ls())
pkgs <- c("data.table","dplyr","caTools","caret","rpart","rpart.plot","ROCR","ggplot2","gridExtra","Information")
sapply(pkgs,require,character.only=TRUE)

# load the data. Import '?' as null values
data <- read.csv("Classification Problem.csv",na.strings = "?")
str(data)

################################################## missing values analysis
# percentage of complete data
sum(complete.cases(data))/nrow(data)*100
# missing values percentage in each variable
sapply(data,function(x) sum(is.na(x))/length(x)*100 )
str(data)
hist(data$C2)
mean(data$C2,na.rm = TRUE);median(data$C2,na.rm = TRUE)
# impute mean values in C2
data$C2[is.na(data$C2)] <- mean(data$C2,na.rm = TRUE)
hist(data$C14)
mean(data$C14,na.rm = TRUE);median(data$C14,na.rm = TRUE)
# impute median value in C14 - since C14 has very wide range of values median is the better measure to replace with
data$C14[is.na(data$C14)] <- median(data$C14,na.rm = TRUE)

sum(complete.cases(data))/nrow(data)
# since observations with missing values are ~2.75%, we can work by removing the data points with missing values
data <- data[complete.cases(data),]

################################################## check for class imbalance
table(data$Hired) # class is not imbalanced

################################################## EDA
# plot histogram
hist_num <- function (x_label){
  ggplot(data = data, aes(x=data[,x_label],y=..density..))+
    geom_histogram(binwidth = 1,fill="blue",color="red") + geom_density() + xlab(label = x_label)
}
hist_num("C2")
hist_num("C3")
hist_num("C8")
hist_num("C11")
table(data$C11); quantile( data$C11, c(.05, .95 ) )
hist_num("C14")
table(data$C14);quantile( data$C14, c(.05, .95 ) )
summary(data$C14);boxplot(data$C14);
outlier_treat <- function(x){
  quantiles <- quantile( x, c(.05, .95 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  x
}
data$C14 <- outlier_treat(data$C14) # outliers are replaced with 5-95%le values
summary(data$C14);boxplot(data$C14)
hist_num("C15")
summary(data$C15)
# C15 has very wide range of values with very few number of observations with sparse distribution.
# better to create bins 
C15_bin <- cut(data$C15,breaks = c(0,10,50,100,500,1000,100000), include.lowest=TRUE, 
                        labels = c("0-10","11-50","51-100","101-500","501-1000","1001-100000"))
table(C15_bin)
data$C15_bin <- C15_bin
data$C15 <- NULL

# plot the histograms on a single graph
p1<-hist_num("C2");p2<-hist_num("C3");p3<-hist_num("C8");p4<-hist_num("C11");p5<-hist_num("C11")
grid.arrange(p1,p2,p3,p4,p5,nrow=2)


####################################################### Feature importance and selection
factcols <- c(1,4,5,6,7,9,10,12,13,15,16)
numcols <- setdiff(c(1:16),factcols)

data[,factcols] <- lapply(data[,factcols],factor)
data[,numcols] <- lapply(data[,numcols],as.numeric)
str(data)

data_cat <- data[,factcols]
data_num <- data[,numcols]

# preprocessParams <- preProcess(data_num,method = c("YeoJohnson"))
# data_num <- predict(preprocessParams, data_num)
# hist(data_num$C14)

# Correlation among numerical predictors
library(corrplot)
cor_mat <- cor(data_num)
corrplot(cor_mat,method = "circle")
# check VIF
library(usdm)
vif(data_num) # VIF does not seem to be very high - No action required

# vifcor checks the correlation between pairs of variables and 
# removes one of the variable from the pair that has high VIF value
vifcor(data_num, th=0.5) # th is the correlation threshold - see cor_mat 

# vifstep removes variables with highest VIF (beyond threshold) one at a time and recalculates VIF values
vifstep(data_num, th=1.1)

# bar graph
bar_cat <- function (x_label){
    ggplot(data = data, aes(x=data[,x_label]))+
    geom_bar(fill="blue",color="red")+theme(axis.text.x =element_text(angle  = 60,hjust = 1))+xlab(label = x_label)
}
p1 <- bar_cat("C1");prop.table(table(data$C1,data$Hired))
p2 <- bar_cat("C4");prop.table(table(data$C4,data$Hired))
p3 <- bar_cat("C5");prop.table(table(data$C5,data$Hired))
p4 <- bar_cat("C6");prop.table(table(data$C6,data$Hired))
p5 <- bar_cat("C7");prop.table(table(data$C1,data$Hired))
p6 <- bar_cat("C9");prop.table(table(data$C9,data$Hired)) # very high predictive power for response variable
p7 <- bar_cat("C10");prop.table(table(data$C10,data$Hired))
p8 <- bar_cat("C12");prop.table(table(data$C12,data$Hired))
p9 <- bar_cat("C13");prop.table(table(data$C13,data$Hired))
p10 <- bar_cat("C15_bin");prop.table(table(data$C1,data$Hired))

# plot the histograms on a single graph
grid.arrange(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,nrow=2)

# chisquare among categorical variables
chisqmatrix <- function(x) {
  names = colnames(x);  num = length(names)
  m = matrix(nrow=num,ncol=num,dimnames=list(names,names))
  for (i in 1:(num-1)) {
    for (j in (i+1):num) {
      m[i,j] =format(round(chisq.test(x[,i],x[,j])$p.value,6), scientific = FALSE)
    }
  }
  return (m)
}
mat = chisqmatrix(data.frame(data_cat))
# will not able to interpret results directly using chisquare since the numbr of observations 
# in many levels are very low

# combine categories that has less than 5% values as 'other'
for (i in names(data_cat)){
  ld <- names(which(prop.table(table(data_cat[[i]]))<0.05))
  levels(data_cat[[i]])[which(levels(data_cat[[i]]) %in% ld)] <- "Other"
} 

library(InformationValue)
# 
# WOETable(data_cat$C9,data_cat$Hired,valueOfGood = 1)
# iv <- IV(data_cat$C9,data_cat$Hired,valueOfGood = 1)

sapply(data,function(x){IV(x,data$Hired,valueOfGood=1)})
sapply(data,function(x){iv <- IV(x,data$Hired,valueOfGood=1); attr(iv,which = "howgood")})

library(Information)

data$Hired <- as.numeric(as.character(data$Hired))
IV <- create_infotables(data=data,
                        y="Hired")
# train$Hired <- as.numeric(as.character(train$Hired))
# test$Hired <- as.numeric(as.character(test$Hired))
# IV <- create_infotables(data=train,valid=test,y="Hired")
plotFrame <- IV$Summary[order(-IV$Summary$IV),]
ggplot(plotFrame, aes(x = Variable, y = IV)) +
  geom_bar(width = .5, stat = "identity", color = "red", fill = "blue") +
  ggtitle("Information Value") +
  theme_bw() +
  theme(plot.title = element_text(size = 10)) +
  theme(axis.text.x = element_text(angle = 60,hjust=1)) +
  scale_x_discrete(limits = plotFrame$Variable) # to sort the variables in descending order

IV$Summary
# or use below code to first reorder the levels to plot in descending order
# plotFrame$Variable <- factor(plotFrame$Variable, levels = plotFrame$Variable[order(-plotFrame$IV)])

# C9 has the maximum Information Value
 
library(Boruta)
boruta_output <- Boruta(Hired~.,data,doTrace=1) # doTrace for the level of verbose
print(boruta_output)
print(attStats(boruta_output))

data <- cbind(data_cat,data_num)

# let's drop the variable with less than 0.01 IV
data <- data[,!names(data) %in% c("C1","C12")]
data$Hired <- as.factor(data$Hired)
########################### Modelling ###########################

# split the data in train and test
set.seed(100)
spl = sample.split(data$Hired,SplitRatio = 0.7)
train = data[spl==TRUE,]
test = data[spl==FALSE,]
prop.table(table(train$Hired));prop.table(table(test$Hired));


# logistic regression model

log_model <- glm(Hired~.,data = train,  family = "binomial",maxit=26)
summary(log_model)
# model with significant variables
log_model <- glm(Hired~C6+C9+C14+C15_bin,data = train,  family = "binomial",maxit=26)
summary(log_model)

log_fitted <- predict(log_model,newdata = train, type = "response")

# check for linearity between logit and numeric variable that is part of the model
train_num <- data.frame(train[,c("C14")])
colnames(train_num)<-c("C14")
library(tidyr)
train_num <- train_num %>%
  mutate(logit = log(log_fitted/(1-log_fitted))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(train_num, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

# we should drop C14 since it does not seem to be random and does not seem to have any relationship with logit
log_model <- glm(Hired~C6+C9+C15_bin,data = train,  family = "binomial",maxit=26)
summary(log_model)

# Evaluation on Train
log_fitted <- predict(log_model,newdata = train, type = "response")
# RCO Curve
ROCRpred <- prediction(log_fitted, train$Hired)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf)
# KS Statistic
ks=max(attr(ROCRperf,'y.values')[[1]]-attr(ROCRperf,'x.values')[[1]])
plot(ROCRperf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1));print(ks);
# Area Under Curve
auc <- performance(ROCRpred, measure = "auc")
auc <- auc@y.values[[1]];print(auc)

# Confusion Matrix
log_fitted_class <- ifelse(log_fitted>0.5,1,0)
log_cm <- caret::confusionMatrix(data=factor(log_fitted_class),reference = train$Hired)
log_cm$byClass;log_cm$overall

## LIFT CHART
lift.obj <- performance(ROCRpred, measure="lift", x.measure="rpp")
plot(lift.obj,main="Lift Chart",xlab="% Populations",ylab="Lift",col="blue")
abline(1,0,col="grey")

#GAINS TABLE
library(gains)
# gains table
gains.cross <- gains(actual = as.numeric(train$Hired)-1,predicted = log_fitted, groups=10)
print(gains.cross)

# Evaluation on Test
log_fitted <- predict(log_model,newdata = test, type = "response")
# ROC Curve
ROCRpred <- prediction(log_fitted, test$Hired)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf)
# KS Statistic
ks=max(attr(ROCRperf,'y.values')[[1]]-attr(ROCRperf,'x.values')[[1]])
plot(ROCRperf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks);
# Area Under Curve
auc <- performance(ROCRpred, measure = "auc")
auc <- auc@y.values[[1]]
print(auc)

# Confusion Matrix
log_fitted_class <- ifelse(log_fitted>0.5,1,0)
log_cm <- caret::confusionMatrix(data=factor(log_fitted_class),reference = test$Hired)
log_cm$byClass;log_cm$overall


#### Decision Tree
# set seed
set.seed(111)
# cross validation
fitControl = trainControl(method = "cv", number = 10)
cartGrid=expand.grid(.cp=seq(0.001,0.5,0.002))
cv_train <- train(Hired ~ ., data=train, method="rpart", trControl = fitControl, tuneGrid = cartGrid)

# build model with best tuned CP
CART_model = rpart(Hired ~ ., data=train, cp=cv_train$bestTune)
# train data evaluation
# predict values of test set
CART_prediction <- predict(CART_model, newdata = train, type = "class")
#confusion matrix
CART_cm <- confusionMatrix(data=factor(CART_prediction),reference = train$Hired)
CART_cm$byClass;CART_cm$overall

# predict values of test set
CART_prediction <- predict(CART_model, newdata = test, type = "class")
#confusion matrix
CART_cm <- confusionMatrix(data=factor(CART_prediction),reference = test$Hired)
CART_cm$byClass;CART_cm$overall
# plot the tree
rpart.plot(CART_model,extra = 1,type = 1,digits = -3);title("Hiring Challenge")

## Random Forest
# train control for rf
train_control <- trainControl(method="cv", number=10)

rf_model <- train(Hired~., data=train, trControl=train_control, method="rf")
print(rf_model)

# evaluation on train
rf_predictions <- predict(rf_model,newdata= train)
#Confusion Matrix
rf_model_cm <- confusionMatrix(data=rf_predictions,reference = train$Hired)
rf_model_cm$byClass;rf_model_cm$overall

# Evaluation on Test
rf_predictions <- predict(rf_model,newdata= test)
rf_model_cm <- confusionMatrix(data=rf_predictions,reference = test$Hired)
rf_model_cm$byClass;rf_model_cm$overall

### XGB
#### XGB using caret
library(xgboost)
library(dummies)

# convert the data frame into matrix with only numeric values
str(train)
train_y <- train$Hired
# dummy variables for categorical variables
dummy_cols <- names(train)[sapply(train,function(x) class(x)=="factor")]
train_with_dummy <- dummy.data.frame(train, names=dummy_cols, sep="_")

train_with_dummy$Hired_0 <- NULL;train_with_dummy$Hired_1 <- NULL;
train_matrix <- data.matrix(train_with_dummy)

test_with_dummy <- dummy.data.frame(test, names=dummy_cols, sep="_")
test_y <- test$Hired
test_with_dummy$Hired_0 <- NULL;test_with_dummy$Hired_1 <- NULL;
test_matrix <- data.matrix(test_with_dummy)

# drop levels from train which are missing in test
levels_to_drop <- setdiff(colnames(train_matrix),colnames(test_matrix))
train_matrix <- train_matrix[,-which(colnames(train_matrix) %in% (levels_to_drop))]
xgb_params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.1,max_depth=5, 
                   min_child_weight=1, subsample=0.5, colsample_bytree=0.5,
                   eval_metric = "error", verbose=TRUE, seed =1)
xgbcv <- xgb.cv( params = xgb_params, data = train_matrix,label = data.matrix(train_y), nrounds = 100, 
                 nfold = 10, showsd = T, stratified = T, print_every_n = 1)

xgb_model <- xgboost(data = train_matrix,label = data.matrix(train_y),objective = "binary:logistic", eta = 0.1,max_depth = 5,
                     nround=25,min_child_weight=1,subsample = 0.5,colsample_bytree = 0.5,eval_metric = "error",
                     verbose = TRUE,seed = 1)

# evaluation on train
xgb_prediction <- predict(xgb_model,newdata = train_matrix)
ROCRpred <- prediction(xgb_prediction, train_y)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf)

ks=max(attr(ROCRperf,'y.values')[[1]]-attr(ROCRperf,'x.values')[[1]])
plot(ROCRperf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks);
auc <- performance(ROCRpred, measure = "auc")
auc <- auc@y.values[[1]]
print(auc)

# Confusion Matrix
xgb_prediction_class <- ifelse(xgb_prediction>0.5,1,0)
xgb_cm <- confusionMatrix(data=factor(xgb_prediction_class),reference = train_y)
xgb_cm$byClass;xgb_cm$overall

# evaluation of Test
xgb_prediction <- predict(xgb_model,newdata = test_matrix)
#view variable importance plot
mat <- xgb.importance (feature_names = colnames(train_matrix),model = xgb_model)
xgb.plot.importance (importance_matrix = mat[1:20]) 

library(ROCR)
ROCRpred <- prediction(xgb_prediction, test_y)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf)
ks=max(attr(ROCRperf,'y.values')[[1]]-attr(ROCRperf,'x.values')[[1]])
plot(ROCRperf,main=paste0(' KS=',round(ks*100,1),'%'))
lines(x = c(0,1),y=c(0,1))
print(ks);
auc <- performance(ROCRpred, measure = "auc")
auc <- auc@y.values[[1]]
print(auc)

# Confusion Matrix
xgb_prediction_class <- ifelse(xgb_prediction>0.5,1,0)
xgb_cm <- confusionMatrix(data=factor(xgb_prediction_class),reference = test_y)
xgb_cm$byClass;xgb_cm$overall

