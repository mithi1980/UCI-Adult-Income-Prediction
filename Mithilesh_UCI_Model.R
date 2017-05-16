Add_Count_Feature<-function(train_df,group_by,aggregate_col,New_Feature_Name,aggregate_type,filter_clause)
{
  sqlstmt<-paste("select",group_by,",",aggregate_type,"(",aggregate_col,")",New_Feature_Name,"from",train_df,filter_clause,"group by",group_by,sep = " ")  
  tempdf<-sqldf(sqlstmt)
  eval(parse(text=paste(train_df,'$',New_Feature_Name,'<-NULL',sep="")))
  train_test<-merge(train_test,tempdf,by=group_by,all.x = T)
  eval(parse(text="train_test[,c(New_Feature_Name)]<-as.numeric(train_test[,c(New_Feature_Name)])"))
  return(train_test)
}

# -------------------------------------------------------------------------
#  1. Data loading with libraries
# -------------------------------------------------------------------------
## setting working directory (edit your path before running script)
#path <- "home/mithilesh/Documents/SP JAIN/Machine Learning/ML103"
#setwd(path)

## loading libraries
library(data.table)
library(plyr)
library(xgboost)
library(dplyr)
library(nnet)
library(mice)
library(car)
library(sqldf)
library(YaleToolkit)
library(randomForest)
rm(list=ls())

#get the data replace empty values with NA
train = read.csv("/home/mithilesh/Documents/SP JAIN/Machine Learning/ML103/train.csv", header=T, na.strings=c("","NA"))
summary(train)
test = read.csv("/home/mithilesh/Documents/SP JAIN/Machine Learning/ML103/test.csv", header=T, na.strings=c("","NA"))
summary(test)
ntrain=nrow(train)
ntest=nrow(test)


boxplot(train,col = c('red'))

col_names=c(
  "Id"
  ,"age"
  ,"workclass"
  ,"fnlwgt"
  ,"education"     
  ,"education_num"
  ,"marital_status"
  ,"occupation"     
  ,"relationship"   
  ,"race"           
  ,"sex"           
  ,"capital_gain"   
  ,"capital_loss"   
  ,"hr_per_week"    
  ,"native_country" 
  ,"income")

test[["income"]] = 0
train_test=rbind(train,test)
train_test[["X"]]=NULL
train_test[["education_num"]]=NULL
train_test[["fnlwgt"]]=NULL
summary(train_test)

train_test$workclass <- as.character(train_test$workclass)
train_test$workclass[train_test$workclass == "?"] <- NA
train_test$workclass[train_test$workclass == " ?"] <- NA
train_test$native_country <- as.character(train_test$native_country)
train_test$native_country[train_test$native_country == "?"] <- NA
train_test$native_country[train_test$native_country == " ?"] <- NA
train_test$workclass <- as.factor(train_test$workclass)
train_test$native_country <- as.factor(train_test$native_country)
summary(train_test)

#3.Missing value treatment
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(train_test,2,pMiss)
md.pattern(train_test)
whatis(train_test)

##-------------------------------------------------------------------------------------------
table(train_test$workclass, useNA = "always")
# Given "Never worked" and "Without-Pay" are both very small groups, and they are
# likely very similar, we can combine them to form a "Not Working" Category.
# In a similar vein, we can combine government employee categories, and self-employed
# categories. This allows us to reduce the number of categories significantly.
train_test$workclass = gsub("^Federal-gov","Federal-Govt",train_test$workclass)
train_test$workclass = gsub("^Local-gov","Other-Govt",train_test$workclass)
train_test$workclass = gsub("^State-gov","Other-Govt",train_test$workclass)
train_test$workclass = gsub("^Private","Private",train_test$workclass)
train_test$workclass = gsub("^Self-emp-inc","Self-Employed",train_test$workclass)
train_test$workclass = gsub("^Self-emp-not-inc","Self-Employed",train_test$workclass)
train_test$workclass = gsub("^Without-pay","Not-Working",train_test$workclass)
train_test$workclass = gsub("^Never-worked","Not-Working",train_test$workclass)
class(train_test$workclass)
train_test$workclass=as.factor(train_test$workclass)
levels(train_test$workclass)

table(train_test$occupation, useNA = "always")
# On occupation, a simple way to block the categories would include blue collar versus white
# collar. Separate out service industry, and other occupations that are not fitting well with
# the other groups into their own group. It's unfortunate that Armed Forces won't fit well
# with any of the other groups. In order to get it properly represented, we can try up-sampling
# it when we train the model.
train_test$occupation = gsub("^Adm-clerical","Admin",train_test$occupation)
train_test$occupation = gsub("^Armed-Forces","Military",train_test$occupation)
train_test$occupation = gsub("^Craft-repair","Blue-Collar",train_test$occupation)
train_test$occupation = gsub("^Exec-managerial","White-Collar",train_test$occupation)
train_test$occupation = gsub("^Farming-fishing","Blue-Collar",train_test$occupation)
train_test$occupation = gsub("^Handlers-cleaners","Blue-Collar",train_test$occupation)
train_test$occupation = gsub("^Machine-op-inspct","Blue-Collar",train_test$occupation)
train_test$occupation = gsub("^Other-service","Service",train_test$occupation)
train_test$occupation = gsub("^Priv-house-serv","Service",train_test$occupation)
train_test$occupation = gsub("^Prof-specialty","Professional",train_test$occupation)
train_test$occupation = gsub("^Protective-serv","Other-Occupations",train_test$occupation)
train_test$occupation = gsub("^Sales","Sales",train_test$occupation)
train_test$occupation = gsub("^Tech-support","Other-Occupations",train_test$occupation)
train_test$occupation = gsub("^Transport-moving","Blue-Collar",train_test$occupation)

table(train_test$native_country, useNA = "always")
train_test$native_country=as.character(train_test$native_country)
# The variable country presents a small problem. Obviously the United States
# represents the vast majority of observations, but some of the groups have
# such small numbers that their contributions might not be significant. A way
# around this would be to block the countries.
# Use a combination of geographical location, political organization, and economic
# zones. Euro_1 is countries within the Eurozone that are considered more affluent,
# and therefore people from there are probably going to be more affluent. Euro_2
# includes countries within the Eurozone that are considered less affluent. These
# included countries that are financially troubled like Spain and Portugal, but also
# the Slavic countries and those formerly influenced by the USSR like Poland. Formerly
# British holdings that are still closely economically aligned with Britain are included
# under the British-Commonwealth.
train_test$native_country[train_test$native_country=="Cambodia"] = "SE-Asia"
train_test$native_country[train_test$native_country=="Canada"] = "British-Commonwealth"  
train_test$native_country[train_test$native_country=="China"] = "China"   
train_test$native_country[train_test$native_country=="Columbia"] = "South-America"  
train_test$native_country[train_test$native_country=="Cuba"] = "Other"    
train_test$native_country[train_test$native_country=="Dominican-Republic"] = "Latin-America"
train_test$native_country[train_test$native_country=="Ecuador"] = "South-America"  
train_test$native_country[train_test$native_country=="El-Salvador"] = "South-America"
train_test$native_country[train_test$native_country=="England"] = "British-Commonwealth"
train_test$native_country[train_test$native_country=="France"] = "Euro_1"
train_test$native_country[train_test$native_country=="Germany"] = "Euro_1"
train_test$native_country[train_test$native_country=="Greece"] = "Euro_2"
train_test$native_country[train_test$native_country=="Guatemala"] = "Latin-America"
train_test$native_country[train_test$native_country=="Haiti"] = "Latin-America"
train_test$native_country[train_test$native_country=="Holand-Netherlands"] = "Euro_1"
train_test$native_country[train_test$native_country=="Honduras"] = "Latin-America"
train_test$native_country[train_test$native_country=="Hong"] = "China"
train_test$native_country[train_test$native_country=="Hungary"] = "Euro_2"
train_test$native_country[train_test$native_country=="India"] = "British-Commonwealth"
train_test$native_country[train_test$native_country=="Iran"] = "Other"
train_test$native_country[train_test$native_country=="Ireland"] = "British-Commonwealth"
train_test$native_country[train_test$native_country=="Italy"] = "Euro_1"
train_test$native_country[train_test$native_country=="Jamaica"] = "Latin-America"
train_test$native_country[train_test$native_country=="Japan"] = "Other"
train_test$native_country[train_test$native_country=="Laos"] = "SE-Asia"
train_test$native_country[train_test$native_country=="Mexico"] = "Latin-America"
train_test$native_country[train_test$native_country=="Nicaragua"] = "Latin-America"
train_test$native_country[train_test$native_country=="Outlying-US(Guam-USVI-etc)"] = "Latin-America"
train_test$native_country[train_test$native_country=="Peru"] = "South-America"
train_test$native_country[train_test$native_country=="Philippines"] = "SE-Asia"
train_test$native_country[train_test$native_country=="Poland"] = "Euro_2"
train_test$native_country[train_test$native_country=="Portugal"] = "Euro_2"
train_test$native_country[train_test$native_country=="Puerto-Rico"] = "Latin-America"
train_test$native_country[train_test$native_country=="Scotland"] = "British-Commonwealth"
train_test$native_country[train_test$native_country=="South"] = "Euro_2"
train_test$native_country[train_test$native_country=="Taiwan"] = "China"
train_test$native_country[train_test$native_country=="Thailand"] = "SE-Asia"
train_test$native_country[train_test$native_country=="Trinadad&Tobago"] = "Latin-America"
train_test$native_country[train_test$native_country=="United-States"] = "United-States"
train_test$native_country[train_test$native_country=="Vietnam"] = "SE-Asia"
train_test$native_country[train_test$native_country=="Yugoslavia"] = "Euro_2"

table(train_test$education, useNA = "always")
# Block all the dropouts together. Block high school graduates and those that
# attended some college without receiving a degree as another group. Those college
# graduates who receive an associates are blocked together regardless of type of
# associates. Those who graduated college with a Bachelors, and those who went on to
# graduate school without receiving a degree are blocked as another group. Most
# everything thereafter is separated into its own group.
train_test$education = gsub("^10th","Dropout",train_test$education)
train_test$education = gsub("^11th","Dropout",train_test$education)
train_test$education = gsub("^12th","Dropout",train_test$education)
train_test$education = gsub("^1st-4th","Dropout",train_test$education)
train_test$education = gsub("^5th-6th","Dropout",train_test$education)
train_test$education = gsub("^7th-8th","Dropout",train_test$education)
train_test$education = gsub("^9th","Dropout",train_test$education)
train_test$education = gsub("^Assoc-acdm","Associates",train_test$education)
train_test$education = gsub("^Assoc-voc","Associates",train_test$education)
train_test$education = gsub("^Bachelors","Bachelors",train_test$education)
train_test$education = gsub("^Doctorate","Doctorate",train_test$education)
train_test$education = gsub("^HS-Grad","HS-Graduate",train_test$education)
train_test$education = gsub("^Masters","Masters",train_test$education)
train_test$education = gsub("^Preschool","Dropout",train_test$education)
train_test$education = gsub("^Prof-school","Prof-School",train_test$education)
train_test$education = gsub("^Some-college","HS-Graduate",train_test$education)

# Similarly
train_test$marital_status=as.character(train_test$marital_status)

train_test$marital_status[train_test$marital_status=="Never-married"] = "Never-Married"
train_test$marital_status[train_test$marital_status=="Married-AF-spouse"] = "Married"
train_test$marital_status[train_test$marital_status=="Married-civ-spouse"] = "Married"
train_test$marital_status[train_test$marital_status=="Married-spouse-absent"] = "Not-Married"
train_test$marital_status[train_test$marital_status=="Separated"] = "Not-Married"
train_test$marital_status[train_test$marital_status=="Divorced"] = "Not-Married"
train_test$marital_status[train_test$marital_status=="Widowed"] = "Widowed"

train_test$race=as.character(train_test$race)

train_test$race[train_test$race=="White"] = "White"
train_test$race[train_test$race=="Black"] = "Black"
train_test$race[train_test$race=="Amer-Indian-Eskimo"] = "Amer-Indian"
train_test$race[train_test$race=="Asian-Pac-Islander"] = "Asian"
train_test$race[train_test$race=="Other"] = "Other"
##-------------------------------------------------------------------------------------------
#Missing value tratement
mean(train_test$age,na.rm=TRUE)
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
train_test$workclass[is.na(train_test$workclass)]=Mode(train_test$workclass)
train_test$native_country[is.na(train_test$native_country)]=Mode(train_test$native_country)

train_test$income<-ifelse(train_test$income==">50K", 1, 0)
train_test$income[is.na(train_test$income)]=-9999

##-------------------------------------------------------------------------------------------
for (f in col_names) {
  if(!f=="income")
  {
    if (class(train_test[[f]])=="character") {
      #cat("VARIABLE : ",f,"\n")
      levels <- unique(train_test[[f]])
      train_test[[f]] <- as.numeric(as.integer(factor(train_test[[f]], levels=levels)))
    }
    if (class(train_test[[f]])=="factor") {
      #cat("VARIABLE : ",f,"\n")
      levels <- unique(train_test[[f]])
      train_test[[f]] <- as.numeric(as.integer(factor(train_test[[f]], levels=levels)))
    }
  }
}
whatis(train_test)
##-------------------------------------------------------------------------------------------

##Start Feature Engg
#train_test$edu_income_ratio<-NULL

#temp<-sqldf("select education,round(sum(case when income=0 then 1 else 0 end)/sum(case when income=1 then 1 else 0 end),1) edu_income_ratio from train_test group by education")
#train_test<-sqldf("select t1.*,t2.edu_income_ratio from train_test t1,temp t2 where t1.education=t2.education")
train_test$rel_income_ratio<-NULL
temp<-sqldf("select relationship,round(sum(case when income=0 then 1 else 0 end)/sum(case when income=1 then 1 else 0 end),1) rel_income_ratio from train_test group by relationship")
train_test<-sqldf("select t1.*,t2.rel_income_ratio from train_test t1,temp t2 where t1.relationship=t2.relationship")

train_test$occup_count<-NULL
train_test<-Add_Count_Feature('train_test','occupation','occupation','occup_count','count','where income<>-9999')


#train_test$hrweek_age_ratio<-ifelse(train_test$education %in% c(1,2) & train_test$marital==3,1,0)
#train_test$hrweek_age_ratio<-NULL

train_test$sex<-NULL
#train_test$type_employer<-NULL## this is required
train_test$country<-NULL
#train_test$relationship<-NULL## this is required
train_test$race<-NULL
#summary(train_test)
#summary(train_test$occup_count)
#summary(as.factor(train_test$occupation))

##End Feature Engg


train_new = train_test[train_test$income!=-9999, ]
test_new = train_test[train_test$income==-9999, ]
y_train = train_new[["income"]] 
y_train
#target = as.numeric(ifelse(y_train==">50K", 1, 0))
target=y_train
test_Id = test_new[["Id"]]

train_new[["Id"]]=NULL
train_new[["income"]]=NULL
test_new[["Id"]]=NULL
test_new[["income"]]=NULL
table(train_new$income)
table(test$income)
# Preparing for xgboost
# dtrain = xgb.DMatrix(as.matrix(train_new), label=y_train)
# dtest = xgb.DMatrix(as.matrix(test_new))


##-------------------------------------------------------------------------------------------
## xgboost : 1 st 
##-------------------------------------------------------------------------------------------

seed <- 235
set.seed(seed)
# cross-validation
model1_xgb_cv <- xgb.cv(data=as.matrix(train_new), label=as.matrix(target), objective="binary:logistic", nfold=6, nrounds=800, eta=0.025, max_depth=6, subsample=0.75, colsample_bytree=0.70, min_child_weight=1, eval_metric="auc")
print(model1_xgb_cv)

# model building
model1_xgb <- xgboost(data=as.matrix(train_new), label=as.matrix(target), objective="binary:logistic", nrounds=650, eta=0.025, max_depth=6, subsample=0.75, colsample_bytree=0.80, min_child_weight=1, eval_metric="auc")
feature.names <- colnames(train_new)#[-13]
importance_matrix <- xgb.importance(feature.names, model = model1_xgb)
summary(model1_xgb)
#importance_matrix
xgb.plot.importance(importance_matrix[1:10,])
# model scoring
pred <- predict(model1_xgb, as.matrix(test_new))

#caret::confusionMatrix(target,)#, mode = "prec_recall")

# Start Random Forest
bestmtry <- tuneRF(train_new, target,
                   ntreeTry=100, stepFactor=1.5, improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)

train_new$income<-as.factor(target)
#test_new$income <- as.factor(y_test)
rf.fit <- randomForest(income ~ ., data=train_new,
                       mtry=3, ntree=700, keep.forest=TRUE, importance=TRUE, test=test_new)
summary(rf.fit)
rf.preds = predict(rf.fit, type="prob", newdata=test_new)[,2]
#rf.pred = prediction(rf.preds, x_test$income)
library(Metrics)
rf.perf = performance(rf.pred, "tpr", "fpr")

importance(rf.fit)
varImpPlot(rf.fit)

# prediction accuracy and other metric in random forest far below than xgboost
###End Random Forest


# submission
submitdata <- data.frame("Id" = paste("A",as.character(test_Id),sep=""), "Prediction" = pred)

write.csv(submitdata, "/home/mithilesh/Documents/SP JAIN/Machine Learning/ML103/Submission.csv", row.names=F)
# LB: 0.92754
str(submitdata)

