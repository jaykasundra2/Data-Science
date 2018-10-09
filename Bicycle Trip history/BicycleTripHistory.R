# load the packages
rm(list=ls())
pkgs <- c("data.table","dplyr","rpart","rpart.plot","RColorBrewer","rattle")
sapply(pkgs,require,character.only=TRUE)

# read the data
train<-read.csv("train.csv")
test<-read.csv("test.csv")

# combine train and test
test$registered=0
test$casual=0
test$count=0
data=rbind(train,test)

# EDA
par(mfrow=c(4,2))
par(mar = rep(2, 4))
hist(data$season)
hist(data$weather)
hist(data$humidity)
hist(data$holiday)
hist(data$workingday)
hist(data$temp)
hist(data$atemp)
hist(data$windspeed)

# reset par 
par(mfrow=c(1,1)) #or dev.off()

data$season=as.factor(data$season)
data$weather=as.factor(data$weather)
data$holiday=as.factor(data$holiday)
data$workingday=as.factor(data$workingday)

# create hour variable
data$hour=substr(data$datetime,12,13)
data$hour=as.factor(data$hour)

# create day of the week variable
date=substr(data$datetime,1,10)
days<-weekdays(as.Date(date))
data$day=days

par(mfrow=c(2,1));boxplot(data$registered~data$day);boxplot(data$casual~data$day);par(mfrow=c(1,1));
par(mfrow=c(2,1));boxplot(data$registered~data$weather);boxplot(data$casual~data$weather);par(mfrow=c(1,1));

# create year, month and hour variable
data$year <- as.integer(substr(data$datetime,1,4))
data$month <- as.integer(substr(data$datetime,6,7))
data$hour <- as.integer(data$hour)

# split train and test
train=data[as.integer(substr(data$datetime,9,10))<20,]
test=data[as.integer(substr(data$datetime,9,10))>=20,]

# plot the count by hour
boxplot(train$count~train$hour,xlab="hour", ylab="count of users")
boxplot(log(train$count)~train$hour,xlab="hour",ylab="log(count)")

# check the hourly traffic
d=rpart(registered~hour,data=train)
prp(d)

# create a variable for registered user hourly pattern using above classification
data$dp_reg=0
data$dp_reg[data$hour<8]=1
data$dp_reg[data$hour>=22]=2
data$dp_reg[data$hour>9 & data$hour<18]=3
data$dp_reg[data$hour==8]=4
data$dp_reg[data$hour==9]=5
data$dp_reg[data$hour==20 | data$hour==21]=6
data$dp_reg[data$hour==19 | data$hour==18]=7
table(data$dp_reg)

# create quarter variable
data$year_part[data$year=='2011']=1
data$year_part[data$year=='2011' & data$month>3]=2
data$year_part[data$year=='2011' & data$month>6]=3
data$year_part[data$year=='2011' & data$month>9]=4
data$year_part[data$year=='2012']=5
data$year_part[data$year=='2012' & data$month>3]=6
data$year_part[data$year=='2012' & data$month>6]=7
data$year_part[data$year=='2012' & data$month>9]=8
table(data$year_part)

# create day type (weekend/holiday/working day)
data$day_type=""
data$day_type[data$holiday==0 & data$workingday==0]="weekend"
data$day_type[data$holiday==1]="holiday"
data$day_type[data$holiday==0 & data$workingday==1]="working day"

# create weekend flag
data$weekend=0
data$weekend[data$day=="Sunday" | data$day=="Saturday" ]=1

train$hour=as.factor(train$hour)
test$hour=as.factor(test$hour)

# create log of regular user and casual user
data$logreg <- log(data$registered+1)
data$logcas <- log(data$casual+1)


data$day <- as.factor(data$day)
data$day_type <- as.factor(data$day_type)

train=data[as.integer(substr(data$datetime,9,10))<20,]
test=data[as.integer(substr(data$datetime,9,10))>19,]
str(train)

# create randomforest model for regular users
library(randomForest)
set.seed(415)
fit1 <- randomForest(logreg ~ hour +workingday+day+holiday+ day_type +humidity+atemp+windspeed+season+weather+dp_reg+weekend+year+year_part, data=train,importance=TRUE, ntree=250)

# predict for the test data 
pred1=predict(fit1,test)
test$logreg=pred1

# create randomforest model for casual users
set.seed(415)
fit2 <- randomForest(logcas ~hour + day_type+day+humidity+atemp+windspeed+season+weather+holiday+workingday+weekend+year+year_part, data=train,importance=TRUE, ntree=250)
# predict for the test data 
pred2=predict(fit2,test)
test$logcas=pred2

# convert the log value back to normal count
test$registered=exp(test$logreg)-1
test$casual=exp(test$logcas)-1

# calculate the total count
test$count=test$casual+test$registered

# create the submission file
s<-data.frame(datetime=test$datetime,count=test$count)
write.csv(s,file="submit.csv",row.names=FALSE)

