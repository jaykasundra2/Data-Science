rm(list=ls())
pkgs<-c("data.table","dplyr","caret","h2o","readr","purrr","stringr","jsonlite","ggplot2",
        "scales","tidyr","lubridate","Rmisc","caTools","Metrics")
sapply(pkgs,require,character.only=TRUE)

train <- read_csv("train.csv")
test <- read_csv("test.csv")
fullVisitorId = test$fullVisitorId
head(fullVisitorId)

ParseJSONColumn <- function(column){
  str_c("[ ", str_c(column, collapse = ",", sep=" "), " ]")  %>% 
    fromJSON(flatten = T) %>% as_tibble()}

ParseJSONDataset <- function(data){
  dataset = data  %>% select(trafficSource, totals, geoNetwork, device)  %>% 
    map_dfc(.f = ParseJSONColumn)
  return(dataset)}

train = ParseJSONDataset(train)
test = ParseJSONDataset(test)
ParseColumn <-function(data){
  #converts selected columns to factor
  factor_cols <- c("campaign", "source", "medium","keyword",
                   "referralPath","continent", "subContinent", "country",
                   "browser", "operatingSystem","deviceCategory")
  data[, factor_cols] <- lapply(data[, factor_cols], as.factor)
  
  #converting selected columns to numeric
  num_cols <- c("visits", "hits",
                "bounces", "pageviews", "newVisits")
  data[, num_cols] <- lapply(data[, num_cols], as.numeric)
  
  return(data)}

train = ParseColumn(train)
test = ParseColumn(test)

train['transactionRevenue'] = as.numeric(unlist(train['transactionRevenue']))
train['log_transactionRevenue'] = log(unlist(train['transactionRevenue']))

glimpse(train)

g1 <- head(train,30000) %>% 
  is.na %>% melt %>%
  ggplot(data = .,aes(x = Var1,y = Var2)) +
  geom_raster(aes(fill = value))  +
  scale_fill_grey(name = "",labels = c("Present","Missing")) +
  labs(x = "Observations",y = "Variables")
g1
g2 <- train %>% 
  is.na %>% melt %>% table() %>% colSums() %>% data.frame() %>%
  .[order(-.$TRUE.),] %>% 
  ggplot(data = .,aes(x=rownames(.),y = TRUE.,fill=rownames(.))) +
  geom_bar(stat='identity') + coord_flip() + theme(legend.position="none") +
  labs(x = "Variables",y = "Missing Count")
g2
g3 <- ggplot(train, aes(x = 'log(transactionRevenue)',y=log_transactionRevenue)) +
  geom_boxplot(fill='Blue') + 
  scale_y_continuous(labels = comma)
g3
g4 <- ggplot(train, aes(x = isMobile,y = log_transactionRevenue,fill=isMobile)) +
  geom_boxplot() + theme(legend.position="none") +
  scale_y_continuous(labels = comma)
g4
g5 <- ggplot(train, aes(x = deviceCategory,y = log_transactionRevenue,fill = deviceCategory)) +
  geom_bar(stat = "summary", fun.y = "mean") + theme(legend.position="none") +
  scale_y_continuous(labels = comma) + labs(y='avg (log_transactionRevenue)')
g5
g6 <- ggplot(train, aes(x = medium,y = log_transactionRevenue,fill = medium)) +
  geom_bar(stat = "summary", fun.y = "mean") + theme(legend.position="none") +
  scale_y_continuous(labels = comma) + labs(y='avg (log_transactionRevenue)')
g6
g7 <- ggplot(train, aes(x = continent,y = log_transactionRevenue,fill=continent)) +
  geom_bar(stat = "summary", fun.y = "mean") + theme(legend.position="none") +
  scale_y_continuous(labels = comma) + labs(y='avg (log_transactionRevenue)')
g7
g8 <- ggplot(train, aes(x = subContinent,y = log_transactionRevenue,fill = subContinent)) +
  geom_bar(stat = "summary", fun.y = "mean") + theme(legend.position="none") + coord_flip() +
  scale_y_continuous(labels = comma) +labs(y='avg (log_transactionRevenue)')
g8
rm(list=c(paste0("g",1:8)))

setdiff(names(train), names(test))
train %>% select(-one_of("campaignCode"))
fea_uniq_values <- sapply(train, n_distinct)
fea_del <- names(fea_uniq_values[fea_uniq_values == 1])
train %>% select(-one_of(fea_del))
test %>% select(-one_of(fea_del))
is_na_val <- function(x) x %in% c("not available in demo dataset", "(not provided)",
                                  "(not set)", "<NA>", "unknown.unknown",  "(none)")

train %>% mutate_all(funs(ifelse(is_na_val(.), NA, .)))
test %>% mutate_all(funs(ifelse(is_na_val(.), NA, .)))

train %>% summarise_all(funs(sum(is.na(.))/n()*100)) %>% 
  gather(key="feature", value="missing_pct") %>% 
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
  geom_bar(stat="identity", fill="steelblue")+
  labs(y = "missing %", x = "features") +
  coord_flip() +
  theme_minimal()
??ymd

y <- as.numeric(train$transactionRevenue)
train$transactionRevenue <- NULL
summary(y)
y[is.na(y)] <- 0
summary(y)

p1 <- as_tibble(y) %>% 
  ggplot(aes(x = log1p(value))) +
  geom_histogram(bins = 30, fill="steelblue") + 
  labs(x = "transaction revenue") +
  theme_minimal()
p2 <- as_tibble(y[y>0]) %>% 
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill="steelblue") + 
  labs(x = "non-zero transaction revenue") +
  theme_minimal()
??multiplot
multiplot(p1, p2, cols = 2)

as_tibble(log1p(y[y>0] / 1e6)) %>% 
  ggplot(aes(x = value)) +
  geom_histogram(bins = 30, fill="steelblue") + 
  labs(x = "log(non-zero transaction revenue / 1e6)") +
  theme_minimal()

cols = c('pageviews','log_transactionRevenue')
sample_train = train[,cols]
sample_train$pageviews[is.na(sample_train$pageviews)] <- 0
sample_train$log_transactionRevenue[is.na(sample_train$log_transactionRevenue)] <- 0
dim(sample_train)
set.seed(2) 
sample = sample.split(sample_train$log_transactionRevenue, SplitRatio = .8)

sub_train = subset(sample_train, sample == TRUE)
sub_test  = subset(sample_train, sample == FALSE)

X_test <- sub_test %>% select(-log_transactionRevenue)
y_test <- sub_test %>% select(log_transactionRevenue)
paste('training set rows: ',dim(sub_train)[1])
paste('test set rows: ',dim(sub_test)[1])
h2o.init(nthreads=-1,max_mem_size='4G')
sub_train = as.h2o(sub_train)
X_test = as.h2o(X_test)
RF_model = h2o.randomForest(x='pageviews',
                            y='log_transactionRevenue',
                            training_frame=sub_train)
summary(RF_model)
RF_preds = as.data.frame(h2o.predict(RF_model,X_test))
rmse(y_test,RF_preds)
cols = c('pageviews')
sample_test = test[,cols]
sample_test = as.h2o(sample_test)
preds = as.data.frame(h2o.predict(RF_model,sample_test))
submission = data.frame(fullVisitorId, preds)
str(submission)
submission = as.data.table(submission)
submission <- submission[,list(PredictedLogRevenue=sum(predict)),by=c("fullVisitorId")]
fwrite(submission,"submission.csv")
