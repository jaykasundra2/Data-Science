# clear the environement and load the required packages
rm(list=ls())
pkgs <- c("data.table","dplyr","ggplot2","caret","dummies","mlr")
sapply(pkgs,require,character.only=TRUE)
# read the data
train <- fread("train.csv")
test <- fread("test.csv")

str(train)
# check the null values proportion
sapply(train,function(x) sum(is.na(x))/length(x)*100)

#check the values of income level variable
unique(train$income_level)
train[,income_level := ifelse(income_level=="-50000",0,1)]
test[,income_level := ifelse(income_level=="-50000",0,1)]

# identify and convert the factor and numeric columns
factcol <- c(2:5,7,8:16,20:29,31:38,40,41)
numcol <- setdiff(1:41,factcol)

factcol_names <- names(train)[factcol]

train[,(factcol):=lapply(.SD,factor),.SDcols=factcol]
test[,(factcol):=lapply(.SD,factor),.SDcols=factcol]

cat_train <- train[,.SD,.SDcols=factcol]
num_train <- train[,.SD,.SDcols=numcol]
cat_test <- test[,.SD,.SDcols=factcol]
num_test <- test[,.SD,.SDcols=numcol]

# plot histogram
hist_num <- function (var_x){
  ggplot(data = num_train, aes(x=var_x,y=..density..))+
    geom_histogram(binwidth = 1,fill="blue",color="red") + geom_density()
}

hist_num(num_train$age)
hist_num(num_train$capital_losses)

#bar graph
bar_cat <- function (var_x){
  ggplot(data = num_train, aes(x=var_x))+
    geom_bar(fill="blue",color="red")+theme(axis.text.x =element_text(angle  = 60,hjust = 1))
}
bar_cat(cat_train$class_of_worker)

sum(is.na(num_train))
sum(is.na(num_test))

#remove variables with high correlation
cols_rm <- findCorrelation(x = cor(num_train),cutoff = 0.7)

num_train <- num_train[,-cols_rm,with=FALSE]
num_test <- num_test[,-cols_rm,with=FALSE]

sum(is.na(cat_train))
sum(is.na(cat_test))

# remove variables with more than 50% NAs
mvtr <- sapply(cat_train,function(x) sum(is.na(x))/length(x)*100)
mvte <- sapply(cat_test,function(x) sum(is.na(x))/length(x)*100)

cat_train <- cat_train[,.SD,.SDcols=names(cat_train)[mvtr<5]]
cat_test <- cat_test[,.SD,.SDcols=names(cat_test)[mvtr<5]]

# replace NAs with Unavailable
str(cat_train)
cat_train[,names(cat_train):= lapply(.SD,as.character),.SDcols=names(cat_train)]
cat_train[is.na(cat_train)] <- "Unavailable"
cat_train[,names(cat_train):= lapply(.SD,factor),.SDcols=names(cat_train)]

cat_test[,names(cat_test):= lapply(.SD,as.character),.SDcols=names(cat_test)]
cat_test[is.na(cat_test)] <- "Unavailable"
cat_test[,names(cat_test):= lapply(.SD,factor),.SDcols=names(cat_test)]

sum(is.na(cat_train))
sum(is.na(cat_test))

#group the categories with less than 5% data points into 'other'
for (i in names(cat_train)){
  ld <- names(which(prop.table(table(cat_train[[i]]))<0.05))
  levels(cat_train[[i]])[which(levels(cat_train[[i]]) %in% ld)] <- "Other"
} 

for (i in names(cat_test)){
  ld <- names(which(prop.table(table(cat_test[[i]]))<0.05))
  levels(cat_test[[i]])[which(levels(cat_test[[i]]) %in% ld)] <- "Other"
} 


# create bins for the variables 
num_train[,age:= cut(x = age, breaks = c(0,30,60,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_train[,age:factor(age)]
num_test[,age:= cut(x = age, breaks = c(0,30,60,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_test[,age:factor(age)]

num_train[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_train[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains := as.factor(capital_gains)]
num_train[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses := as.factor(capital_losses)]
num_train[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks := as.factor(dividend_from_Stocks)]

num_test[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_test[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains := as.factor(capital_gains)]
num_test[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses := as.factor(capital_losses)]
num_test[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks := as.factor(dividend_from_Stocks)]

#combine data and make test & train files
d_train <- cbind(num_train,cat_train)
d_test <- cbind(num_test,cat_test)
#remove unwanted files
rm(num_train,num_test,cat_train,cat_test) #save memory

combined <- rbind(d_train,d_test)

d_train <- combined[1:nrow(d_train),]
d_test <- combined[(nrow(d_train)+1):nrow(combined),]
rm(combined)
################################## Modelling ######################################
library(mlr)
#get variable importance chart

train.task <- makeClassifTask(data = as.data.frame(d_train),target = "income_level")
test.task <- makeClassifTask(data=as.data.frame(d_test),target = "income_level")

var_imp <- generateFilterValuesData(train.task, method = c("information.gain"))
plotFilterValues(var_imp,feat.type.cols = TRUE)

#### Sampling for imbalanced class

#undersampling 
train.under <- undersample(train.task,rate = 0.1) #keep only 10% of majority class
table(getTaskTargets(train.under))
train.under.data <- getTaskData(train.under)

#oversampling
train.over <- oversample(train.task,rate=15) #make minority class 15 times
table(getTaskTargets(train.over))
train.over.data <- getTaskData(train.over)

#SMOTE
train.smote <- smote(train.task,rate = 15,nn = 5) 
train.smote.data <- getTaskData(train.smote)
# rate is number of times minority class to be multiplied and
# nn means number of neighbours to consider

#### Logistic
library(caret)
log_model <- glm(income_level~., data = train.smote.data, family="binomial")
summary(log_model)
# prediction using logistic regression model
log_prediction <- predict(log_model, newdata = d_test, type = "response")
log_prediction_class <- ifelse(log_prediction>0.4,1,0)
# ROC Curve
library(ROCR)
ROCRpred <- prediction(log_prediction_class, d_test$income_level)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf)
# Confusion Matrix
log_cm <- confusionMatrix(data=factor(log_prediction_class),reference = d_test$income_level)
log_cm$byClass;log_cm$overall

#### Decision Tree
library(rpart)
set.seed(111)
fitControl = trainControl(method = "cv", number = 10)
cartGrid=expand.grid(.cp=seq(0.01,0.5,0.01))
cv_train <- caret::train(income_level ~ ., data=d_train, method="rpart", trControl = fitControl, tuneGrid = cartGrid)
CART_model = rpart(income_level ~ ., data=d_train, cp=cv_train$bestTune)
CART_prediction <- predict(CART_model, newdata = d_test, type = "class")
CART_cm <- confusionMatrix(data=factor(CART_prediction),reference = d_test$income_level)
CART_cm$byClass;CART_cm$overall
varImp(object = CART_model)

#### Naive Bayse
library(e1071)
fitControl = trainControl(method = "cv", number = 5)
nbGrid = expand.grid(laplace=seq(0,0.6,0.2),adjust=c(1,2),usekernel=c(TRUE,FALSE))
names(getModelInfo()) # possible values of method in train function
nb_model <- caret::train(income_level ~ ., data=d_train, method="naive_bayes", trControl = fitControl, tuneGrid = nbGrid)
nb_model
nb_model$finalModel
nb_model$results
nb_model$bestTune
nb_prediction <- predict(nb_model$finalModel,newdata = d_test)
nb_cm <- confusionMatrix(data=factor(nb_prediction),reference = d_test$income_level)
nb_cm$byClass;nb_cm$overall


#### gbm
fitControl <- trainControl(method = "cv",number = 5)
tune_Grid <-  expand.grid(interaction.depth = 2,n.trees = 100,shrinkage = 0.1,
                          n.minobsinnode = 100)
set.seed(825)
gbm_model <- caret::train(income_level~.,data=train.smote.data,method="gbm",
                   trControl = fitControl, tuneGrid = tune_Grid)

gbm_prediction= predict(gbm_model,d_test,type= "class") 

#### XGB using caret
library(xgboost)
library(dummies)

str(d_train)

dummy_cols <- names(d_train)[sapply(d_train,function(x) class(x)=="factor")]
d_train_with_dummy <- dummy.data.frame(d_train, names=dummy_cols, sep="_")
train_y <- d_train$income_level
d_train_with_dummy$income_level <- NULL
d_train_matrix <- data.matrix(d_train_with_dummy)


d_test_with_dummy <- dummy.data.frame(d_test, names=dummy_cols, sep="_")
test_y <- d_test$income_level
d_test_with_dummy$income_level <- NULL
d_test_matrix <- data.matrix(d_test_with_dummy)

xgb_params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.1, gamma=0, max_depth=5, 
               min_child_weight=1, subsample=0.5, colsample_bytree=0.5,
               eval_metric = "error",nthread = 3, verbose=TRUE, seed =1)

xgbcv <- xgb.cv( params = xgb_params, data = d_train_matrix,label = data.matrix(train_y), nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, print_every_n = 1)
 
xgb_model <- xgboost(data = d_train_matrix,label = data.matrix(train_y), eta = 0.1,max_depth = 5,nround=25, 
               subsample = 0.5,colsample_bytree = 0.5,seed = 1,eval_metric = "error",
               objective = "binary:logistic",nthread = 3,verbose = TRUE)

xgb_prediction <- predict(xgb_model,newdata = d_test_matrix)

xgb_prediction <- predict(xgb_model,newdata = d_test_matrix)
xgb_prediction_class <- ifelse(xgb_prediction>0.4,1,0)
library(ROCR)
ROCRpred <- prediction(xgb_prediction_class, test_y)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf)
# Confusion Matrix
xgb_cm <- confusionMatrix(data=factor(xgb_prediction_class),reference = test_y)
xgb_cm$byClass;xgb_cm$overall


#### xgboost using mlr
str(d_train)
train_xgb <- d_train
train_xgb$income_level <- as.numeric(train_xgb$income_level)
str(train_xgb)
dummy_cols <- names(train_xgb)[sapply(train_xgb,function(x) class(x)=="factor")]
train_xgb <- dummy.data.frame(train_xgb, names=dummy_cols, sep="_")
names(train_xgb) <- gsub("[^[:alnum:]\\_]", "", names(train_xgb))
train_xgb$income_level <- as.factor(train_xgb$income_level)
train_xgb_matrix <- data.matrix(train_xgb)

test_xgb <- d_test
test_xgb$income_level <- as.numeric(test_xgb$income_level)
str(test_xgb)
dummy_cols <- names(test_xgb)[sapply(test_xgb,function(x) class(x)=="factor")]
test_xgb <- dummy.data.frame(test_xgb, names=dummy_cols, sep="_")
names(test_xgb) <- gsub("[^[:alnum:]\\_]", "", names(test_xgb))
test_xgb$income_level <- as.factor(test_xgb$income_level)
test_xgb_matrix <- data.matrix(test_xgb)

str(train_xgb)
train.task <- makeClassifTask(data = train_xgb,target = "income_level")
test.task <- makeClassifTask(data=test_xgb,target = "income_level")


set.seed(2002)
xgb_learner <- makeLearner("classif.xgboost",predict.type = "response")
xgb_learner$par.vals <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  nrounds = 150,
  print_every_n = 50
)

#define hyperparameters for tuning
xg_ps <- makeParamSet( 
  makeIntegerParam("max_depth",lower=3,upper=10),
  makeNumericParam("lambda",lower=0.05,upper=0.5),
  makeNumericParam("eta", lower = 0.01, upper = 0.5),
  makeNumericParam("subsample", lower = 0.50, upper = 1),
  makeNumericParam("min_child_weight",lower=2,upper=10),
  makeNumericParam("colsample_bytree",lower = 0.50,upper = 0.80)
)

#define search function
rancontrol <- makeTuneControlRandom(maxit = 5L) #do 5 iterations

#5 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L,stratify = TRUE)

#tune parameters
xgb_tune <- tuneParams(learner = xgb_learner, task = train.task, resampling = set_cv, measures = list(acc,tpr,tnr,fpr,fp,fn), par.set = xg_ps, control = rancontrol)

#set optimal parameters
xgb_new <- setHyperPars(learner = xgb_learner, par.vals = xgb_tune$x)

#train model
xgmodel <- train(xgb_new, train.task)

#test model
predict.xg <- predict(xgmodel, test.task)
#make prediction
xg_prediction <- predict.xg$data$response

levels(xg_prediction)[c(1,2)] <- c(0,1)
#make confusion matrix
xg_confused <- confusionMatrix(d_test$income_level,xg_prediction)

precision <- xg_confused$byClass['Pos Pred Value']
recall <- xg_confused$byClass['Sensitivity']

f_measure <- 2*((precision*recall)/(precision+recall))

