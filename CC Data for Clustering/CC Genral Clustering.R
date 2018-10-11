# setup the environement
rm(list=ls())
pkgs<-c("data.table","dplyr","caret","NbClust","clustMixType","factoextra","cluster","caTools","mlr","C50","flexclust")
sapply(pkgs,require,character.only=TRUE)

# load the data - downloaded from kaggle
data <- fread("CC GENERAL.csv") 

# drop the ID variable
data <- data[,CUST_ID:=NULL]

# preprocess the data to be center scaled
transformation <- preProcess(data, method = c("center","scale"))
trans_data <- predict(transformation,data)

# check null values
colSums(is.na(trans_data))

# remove the data points with null values
trans_data <- trans_data[complete.cases(trans_data)]

spl = sample.split(1:nrow(trans_data),SplitRatio = 0.7 )
train <- trans_data[spl==TRUE,]
test <- trans_data[spl==FALSE,]

# calculate the distance betwen data points
distance <- get_dist(train)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# plot the total within sum of square (wss) by plotting wss for different numbers of clusters
# also known as elbow method
# ideal number of clusters are where there is steep decline in wss
set.seed(111)
fviz_nbclust(train, kmeans, method = "wss",k.max = 6)

# plot the average silhouette for different numbers of clusters
# also known as silhouette method
# ideal number of clusters are where there is maximum silhouette
fviz_nbclust(train, kmeans, method = "silhouette",k.max = 6)

# compute gap statistic
set.seed(111)
gap_stat <- clusGap(train, FUN = kmeans, nstart = 25,
                    K.max = 6, B = 50)
fviz_gap_stat(gap_stat)

# identify the ideal number of clusters using nbclust
nbclust_clusters <- NbClust(train,min.nc = 3,max.nc = 5,method = "kmeans")
fviz_nbclust(nbclust_clusters, method = "wss")

cluster_suggested <- as.data.frame(table(nbclust_clusters$Best.nc[1,]))
names(cluster_suggested)[1] <- "No_of_Clusters"
ideal_cluster <- as.numeric(cluster_suggested$No_of_Clusters[which(cluster_suggested$Freq==max(cluster_suggested$Freq))])

# perform the clustering with ideal number of clusters
set.seed(111)
final <- kmeans(train,centers = ideal_cluster)
print(final)
#fviz builds the components from the variables using PCA and plots the top 2 principal components
fviz_cluster(final, data = train)

# build a classification model to check the goodness of clustering
# target variable cluster id
train_with_cluster <- train
train_with_cluster$cluster <- final$cluster


train.task <- makeClassifTask(data = data.frame(train_with_cluster),target = "cluster")

getParamSet("classif.randomForest")

#create a learner
rf <- makeLearner("classif.randomForest", predict.type = "response", 
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
                      par.set = rf_param, control = rancontrol, measures = acc)

#cv accuracy
rf_tune$y
#best parameters
rf_tune$x

#using hyperparameters for modeling
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)

#train a model
rf_model <- train(rf.tree, train.task)

# make predictions
# predict the clusters for test data points
km.kcca = as.kcca(final, train)
clusterTest = predict(km.kcca, newdata=test)
test_with_cluster <- test
test_with_cluster$cluster <- clusterTest

test_predict <- predict(rf_model,newdata = data.frame(test_with_cluster))

# consufionMatrix
cm <- confusionMatrix(test_predict$data$response,factor(test_with_cluster$cluster))
cm

# ~96% accuracy on the test data says the clustering has been performed well overall

# build cluster profiles using C5.0
C50_train_x <- train_with_cluster[,c(1:17)]
C50_train_y <- factor(train_with_cluster$cluster)
C50_test_x <- test_with_cluster[,c(1:17)]
C50_test_y <- factor(test_with_cluster$cluster)

c50_model <- C5.0(C50_train_x,C50_train_y,rules = TRUE)
summary(c50_model)

# compare the mean values across clusters
setDT(train_with_cluster)[order(cluster), lapply(.SD,mean),by=cluster]

# Based on the Important variables from C5_model and mean value of those variables across clusters 
# we can build cluster profiles as below

# 1 : Low Purchase Freq, Low Purchases, Low Credit Limit, Low ONEOFF purchases
# 2 : Low Cash Advance, Low Balance, 
# 3 : High Cash Advance, High Balance,   Low PRC Full Payment
# 4 : High Purchase Freq, High Purchases, High Credit Limit, High ONEOFF Purchases,High PRC Full Payment
