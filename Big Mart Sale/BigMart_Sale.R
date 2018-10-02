# clean up the environment and load the required packages
rm(list=ls())
pkgs <- c("data.table","dplyr","ggplot2","caret","mlr")
sapply(pkgs,require,character.only=TRUE)

# read the data
train <- fread("train.csv",na.strings = c("NA","?"," "))
test <- fread("test.csv",na.strings = c("NA","?"," "))

#check the NAs
sum(is.na(train));sum(is.na(test));
sapply(train, function(x) sum(is.na(x)))
sapply(test, function(x) sum(is.na(x)))
glimpse(train);summary(train);str(train);

# check uniques values in each variable
sapply(train, function(x) length(unique(x)))

# visualize the data
ggplot(train, aes(x=Item_Visibility, y= Item_Outlet_Sales))+geom_point(size = 2.5, color="navy") + 
  ggtitle("Item_Visibility Vs Item_Outlet_Sales") +
  xlab("Item Visibility") + ylab("Item Outlet Sales")

ggplot(train, aes(Outlet_Identifier, Item_Outlet_Sales)) + 
  geom_bar(stat = "identity", color = "purple") +
  theme(axis.text.x = element_text(angle = 70, hjust = 0.5, color = "black"))  + 
  ggtitle("Outlets vs Total Sales") + theme_bw()

ggplot(train, aes(Item_Type, Item_Outlet_Sales)) + 
  geom_bar(stat = "identity", color = "purple") +
  theme(axis.text.x = element_text(angle = 60, vjust = 0.5, color = "black"))  + 
  ggtitle("Item_Type vs Total Sales") + theme_bw()

ggplot(train, aes(Item_Type, Item_MRP)) +
  geom_boxplot(outlier.color = "red") + 
  theme(axis.text.x = element_text(angle = 70, vjust = 0.5, color = "navy")) + 
  xlab("Item Type") + ylab("Item MRP") + ggtitle("Item Type vs Item MRP")

# combine the test and train
test$Item_Outlet_Sales <- 0
combined <- rbind(train,test)

# fill the missing values of Item_Weight and Item Visibility=0 with median value
set(combined, i=which(is.na(combined$Item_Weight)), j=c("Item_Weight"), value=median(combined$Item_Weight[!is.na(combined$Item_Weight)]))
set(combined, i=which(combined$Item_Visibility==0), j=c("Item_Visibility"), value=median(combined$Item_Visibility[!is.na(combined$Item_Visibility)]))
levels(factor(combined$Outlet_Size))
str(combined)

factcols <- c(1,3,5,7,8,9,10,11)
combined[,(factcols):=lapply(.SD,factor),.SDcols=factcols]
sapply(combined[,factcols,with=FALSE],levels)
levels(combined$Outlet_Size)[1] <- "Other"
levels(combined$Item_Fat_Content)[levels(combined$Item_Fat_Content) %in% c("LF","low fat")] <- "Low Fat"
levels(combined$Item_Fat_Content)[levels(combined$Item_Fat_Content) == "reg"] <- "Regular"
sapply(combined[,factcols,with=FALSE],levels)

combined[,Outlet_Count := .N,by=Outlet_Identifier]
combined[,Item_Count := .N,by=Item_Identifier]

combined[,Item_Type_New:=substr(Item_Identifier,1,2)]
combined[,Item_Type_New := if_else(Item_Type_New=="FD","Food",
                  if_else(Item_Type_New=="DR","Drinks","Non-Consumable"))]
combined[,Item_Type_New:=factor(Item_Type_New)]
combined[,Outlet_Year:= 2018-as.integer(as.character(combined$Outlet_Establishment_Year))]
combined[,c("Item_Identifier","Outlet_Identifier","Outlet_Establishment_Year"):=NULL]

d_train <- combined[1:nrow(train),]
d_test <- combined[-(1:nrow(train)),]
################ Modelling #############

# linear reg model
cor(d_train[,which(sapply(d_train,class) %in% c("numeric","integer")),with=FALSE])
linear_reg_model <- lm(log(Item_Outlet_Sales)~.,data=d_train)
summary(linear_reg_model)
par(mfrow=c(2,2))
plot(linear_reg_model)
postResample(pred = exp(linear_reg_model$fitted.values), obs = d_train$Item_Outlet_Sales)
# sqrt(mean((exp(linear_reg_model$fitted.values)-d_train$Item_Outlet_Sales)^2))
test_predict <- exp(predict(linear_reg_model,newdata = d_test))
linear_model_submission <- data.table(Item_Identifier=test$Item_Identifier,
                                      Outlet_Identifier=test$Outlet_Identifier,
                                      Item_Outlet_Sales=test_predict)
fwrite(linear_model_submission,"linear_model_submission.csv")
# rpart model
library(rpart)
set.seed(111)
fitControl = trainControl(method = "cv", number = 10)
cartGrid=expand.grid(.cp=seq(0.01,0.5,0.01))
cv_train <- caret::train(Item_Outlet_Sales ~ ., data=d_train, method="rpart", trControl = fitControl, tuneGrid = cartGrid)
CART_model = rpart(Item_Outlet_Sales ~ ., data=d_train, cp=cv_train$bestTune)
CART_prediction <- predict(CART_model, newdata = d_train, type = "vector")
head(CART_prediction)
varImp(object = CART_model)
postResample(pred = CART_prediction, obs = d_train$Item_Outlet_Sales)
library(rpart.plot)
prp(CART_model)
test_predict <- predict(CART_model,newdata = d_test)
rpart_model_submission <- data.table(Item_Identifier=test$Item_Identifier,
                                      Outlet_Identifier=test$Outlet_Identifier,
                                      Item_Outlet_Sales=test_predict)
fwrite(rpart_model_submission,"rpart_model_submission.csv")


#### gbm
fitControl <- trainControl(method = "cv",number = 5)
tune_Grid <-  expand.grid(interaction.depth = 4,n.trees = 150,shrinkage = 0.1,
                          n.minobsinnode = 30)
set.seed(825)
gbm_model <- caret::train(Item_Outlet_Sales~.,data=d_train,method="gbm",
                          trControl = fitControl, tuneGrid = tune_Grid)

gbm_prediction= predict(gbm_model,d_train,type= "raw") 
postResample(pred = gbm_prediction, obs = d_train$Item_Outlet_Sales)
test_predict <- predict(gbm_model,newdata = d_test)
gbm_model_submission <- data.table(Item_Identifier=test$Item_Identifier,
                                     Outlet_Identifier=test$Outlet_Identifier,
                                     Item_Outlet_Sales=test_predict)
fwrite(gbm_model_submission,"gbm_model_submission.csv")

#### xgboost using mlr
str(d_train)
train_xgb <- d_train
str(train_xgb)
dummy_cols <- names(train_xgb)[sapply(train_xgb,function(x) class(x)=="factor")]
train_xgb <- dummy.data.frame(train_xgb, names=dummy_cols, sep="_")
names(train_xgb) <- gsub("[^[:alnum:]\\_]", "", names(train_xgb))
train_xgb_matrix <- data.matrix(train_xgb)

test_xgb <- d_test
str(test_xgb)
dummy_cols <- names(test_xgb)[sapply(test_xgb,function(x) class(x)=="factor")]
test_xgb <- dummy.data.frame(test_xgb, names=dummy_cols, sep="_")
names(test_xgb) <- gsub("[^[:alnum:]\\_]", "", names(test_xgb))
test_xgb_matrix <- data.matrix(test_xgb)

str(train_xgb)
train.task <- makeRegrTask(data = train_xgb,target = "Item_Outlet_Sales")
test.task <- makeRegrTask(data=test_xgb,target = "Item_Outlet_Sales")

set.seed(2002)
listLearners("regr")
xgb_learner <- makeLearner("regr.xgboost",predict.type = "response")
xgb_learner$par.vals <- list(
  objective = "reg:linear",
  eval_metric = "rmse",
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
set_cv <- makeResampleDesc("CV",iters = 3L)

#tune parameters
xgb_tune <- tuneParams(learner = xgb_learner, task = train.task, resampling = set_cv, par.set = xg_ps, control = rancontrol)

#set optimal parameters
xgb_new <- setHyperPars(learner = xgb_learner, par.vals = xgb_tune$x)

#train model
xgmodel <- train(xgb_new, train.task)

#test model
predict.xg <- predict(xgmodel, train.task)
#make prediction
xg_prediction <- predict.xg$data$response

postResample(pred = gbm_prediction, obs = d_train$Item_Outlet_Sales)

test_predict <- predict(xgmodel,newdata = test_xgb)
xgb_model_submission <- data.table(Item_Identifier=test$Item_Identifier,
                                   Outlet_Identifier=test$Outlet_Identifier,
                                   Item_Outlet_Sales=test_predict$data$response)
fwrite(xgb_model_submission,"xgb_model_submission.csv")

# random forest

train.task <- makeRegrTask(data = train_xgb,target = "Item_Outlet_Sales")
test.task <- makeRegrTask(data=test_xgb,target = "Item_Outlet_Sales")

getParamSet("regr.randomForest")

#create a learner
rf <- makeLearner("regr.randomForest", predict.type = "response", 
                  par.vals = list(ntree = 200, mtry = 3))
rf$par.vals <- list(
  importance = TRUE
)

#set tunable parameters
#grid search to find hyperparameters
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 100),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

#let's do random search for 50 iterations
rancontrol <- makeTuneControlRandom(maxit = 5L)
#set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#hypertuning
rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = train.task,
                      par.set = rf_param, control = rancontrol, measures = rmse)

#cv accuracy
rf_tune$y
#best parameters
rf_tune$x

#using hyperparameters for modeling
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)

#train a model
rf_model <- train(rf.tree, train.task)

#make predictions
rf_prediction <- predict(rforest, train.task)

postResample(pred = rf_prediction$data$response, obs = d_train$Item_Outlet_Sales)

test_predict <- predict(rf_model,newdata = test_xgb)
rf_model_submission <- data.table(Item_Identifier=test$Item_Identifier,
                                   Outlet_Identifier=test$Outlet_Identifier,
                                   Item_Outlet_Sales=test_predict$data$response)
fwrite(rf_model_submission,"rf_model_submission.csv")
