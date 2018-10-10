# clear the environment and load the packages
rm(list=ls())
pkgs<-c("data.table","dplyr","uplift","plyr","ROCR","caTools","glmnet")
sapply(pkgs,require,character.only=TRUE)
#load the data
# data has been downloaded from http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
df = read.csv("hillstrom_data.csv") 
df$treat <- ifelse(as.character(df$segment) != "No E-Mail", 1, 0)
df$mens <- as.factor(df$mens)
df$womens <- as.factor(df$womens)
df$newbie <- as.factor(df$newbie)
table(df$segment)

# filter the data with segment as Womens E-Mail (Treatment group) and No-Email (Control Group)
loc_df <- df[df$segment!="Mens E-Mail",]
spl <- sample.split(loc_df$segment,SplitRatio = 0.7)
loc_df_train <- loc_df[spl==TRUE,]
loc_df_test <- loc_df[spl==FALSE,]

######## modified outcome method (MOM) model

# Response Variable Transform for Uplift
df_rvtu <- rvtu(visit~recency+history_segment+history+mens+womens+zip_code+newbie+channel+trt(treat),
                data=loc_df,
                method="none")


# ct is same as treatment variable - in our case 'treat'
# y is the original conversion variable - in our case 'visit'
# z is calculated as z=1 if (ct=1,y=1) - customers that were targeted and converted +
#                        or (ct=0,y=0) - customers that were not targeted and did not convert
names(df_rvtu)
table(df_rvtu$ct)
explore(y~recency+history_segment+history+mens+womens+zip_code+newbie+channel+trt(ct),
        data=df_rvtu)

# targeted
count(df_rvtu[df_rvtu$ct == 1,], "y")$freq / sum(df_rvtu$ct == 1)
# control 
count(df_rvtu[df_rvtu$ct == 0,], "y")$freq / sum(df_rvtu$ct == 0)
par(mfrow=c(2,3))
boxplot(recency~visit, data=df[df$treat,],
        ylab="recency", xlab="visit")
boxplot(recency~conversion, data=df[df$treat,],
        ylab="recency", xlab="conversion")
boxplot(split(df$recency[df$spend != 0 & df$treat],
              cut(df$spend[df$spend != 0 & df$treat], 3)),
        ylab="recency", xlab="spend for converted")
mtext("recency target", side=3, line=-3, outer=TRUE, cex=2, font=2)
boxplot(recency~visit, data=df[! df$treat,],
        ylab="recency", xlab="visit")
boxplot(recency~conversion, data=df[df$treat,],
        ylab="recency", xlab="conversion")
boxplot(split(df$recency[df$spend != 0 & ! df$treat],
              cut(df$spend[df$spend != 0 & ! df$treat], 3)),
        ylab="recency", xlab="spend for converted")
mtext("recency control", side=3, line=-39, outer=TRUE, cex=2, font=2)

# split the data in train and test
df_rvtu_train <- df_rvtu[spl==TRUE,]
df_rvtu_test <- df_rvtu[spl==FALSE,]

# prepare the data for modelling
logit_x_train <- model.matrix(~recency+mens+womens+zip_code+newbie+channel,data=df_rvtu_train)
logit_y_train <- df_rvtu_train$y
logit_z_train <- df_rvtu_train$z
logit_x_test <- model.matrix(~recency+mens+womens+zip_code+newbie+channel,data=df_rvtu_test)
logit_y_test <- df_rvtu_test$y
logit_z_test <- df_rvtu_test$z

# alpha=1 -> Lasso model; alpha=0 -> Ridge model
# cv.glmnet() performs cross-validation, by default 10-fold
# glmnet() will perform regression for an automatically selected range of lambda
logit_y_model <- cv.glmnet(logit_x,logit_y,alpha=1,family="binomial")
plot(logit_y_model);log(logit_y_model$lambda.min)
logit_z_model <- cv.glmnet(logit_x,logit_z,alpha=1,family="binomial")
plot(logit_z_model);log(logit_z_model$lambda.min)

# check the non zero coefficients
coef(logit_z_model)[which(coef(logit_z_model) != 0),]

# check the non zero coed=fficients of the model with min lambda
coef(logit_z_model,
     s=logit_z_model$lambda.min)[which(coef(logit_z_model,s=logit_z_model$lambda.min) != 0),]

# predict the values
predict_y_1se <- predict(logit_y_model, logit_x_test,
                                    s=logit_y_model$lambda.1se, type="response")
pred.y <- prediction(predict_y_1se, logit_y_test)
auc.y <- ROCR:::performance(pred.y, "auc")
as.numeric(auc.y@y.values)

predict_z_1se <- predict(logit_z_model, logit_x_test,
                                    s=logit_z_model$lambda.1se, type="response")

pred.z <- prediction(predict_z_1se, logit_z_test)
auc.z <- ROCR:::performance(pred.z, "auc")
as.numeric(auc.z@y.values)

##### Random Forest Model

# build RF model using upliftRF() of uplift package
rf_model <- upliftRF(visit~recency+history_segment+history+mens+womens+zip_code+newbie+channel+trt(treat),
                     data=loc_df_train,ntree=50,verbose = TRUE,split_method = "KL" )

# predict for the test data
rf_predict <- predict(rf_model, newdata=loc_df_test)
head(rf_predict)

# check the decile wise uplift
rf_decile_uplift <- uplift::performance(rf_predict[, 1], rf_predict[, 2],
                                        loc_df_test$visit, loc_df_test$treat)
rf_decile_uplift
# variable importance chart
varImportance(rf_model, plotit = TRUE, normalize = TRUE)

# ROC curve
pred.y <- prediction(rf_predict[,1], loc_df_test$visit)
plot(ROCR::performance(pred.y,"tpr","fpr"))
as.numeric(performance(pred.y, "auc")@y.values)

# qini curve
qini(rf_decile_uplift, plotit = TRUE)

# using z from df_rvtu

# rf_model <- upliftRF(z~recency+history_segment+history+mens+womens+zip_code+newbie+channel+trt(ct),
#                data=df_rvtu,ntree=50,verbose = TRUE,split_method = "KL" )
# rf_predict <- predict(rf_model, newdata=df_rvtu)
# head(rf_predict)
# plot(rf_predict[,2])
# 
# rf_decile_uplift <- uplift::performance(rf_predict[, 1], rf_predict[, 2],
#                        df_rvtu$y, df_rvtu$ct)
# varImportance(rf_model, plotit = TRUE, normalize = TRUE)
# 
# pred.y <- prediction(rf_predict[,1], df_rvtu$y)
# plot(ROCR::performance(pred.y,"tpr","fpr"))
# as.numeric(performance(pred.y, "auc")@y.values)


###### Causal conditional inference forests (ccif)
set.seed(100)
# build ccif model
ccif_model <- ccif(visit~recency+history_segment+history+mens+womens+zip_code+newbie+channel+trt(treat),
                   data = loc_df_train,
                     ntree = 20,
                     split_method = "Int",
                     distribution = approximate (B=999),
                     verbose = TRUE)
# variable importance chart
varImportance(ccif_model, plotit = TRUE)

# predict the resposne variable for test data
ccif_predict <- predict(ccif_model, newdata=loc_df_test)

# check decile wise uplift
ccif_decile_uplift <- uplift::performance(pr.y1_ct1 = ccif_predict[, 1],
                         pr.y1_ct0 = ccif_predict[, 2],
                         y = loc_df_test$visit,
                         ct = loc_df_test$treat)
ccif_decile_uplift

# ROC curve
pred.y <- prediction(ccif_predict[,1], loc_df_test$visit)
plot(ROCR::performance(pred.y,"tpr","fpr"))
as.numeric(performance(pred.y, "auc")@y.values)

# qini curve
qini(ccif_decile_uplift, plotit=TRUE)
