#Set working directory. NEEDS TO BE MODIFIED IN OTHER SYSTEMS
setwd("G:\\R\\AlgLearningTheory\\Project\\UCI HAR Dataset\\UCI HAR Dataset")

#Clear All data before starting
rm(list=ls())

#Import required libraries
library("klaR")
library("data.table")
library("glmnet")
library("neuralnet")
library("naivebayes")

#Read column names
features <- read.table("features.txt")
#Read activity labels
act_labels <- read.table("activity_labels.txt")

#Read train and test data files
train_features <- read.table("train\\X_train.txt")
train_labels <- read.table("train\\y_train.txt")
test_features <- read.table("test\\X_test.txt")
test_labels <- read.table("test\\y_test.txt")

#Add column names to datasets
colnames(train_features) <- c(t(features)[2,])
colnames(test_features) <- c(t(features)[2,])

#Prepate datasets for merging
colnames(train_labels) <- c("act")
colnames(test_labels) <- c("act")
colnames(act_labels) <- c("act", "Activity")

#Merge labels with features in a single dataset
train_dataset <- cbind(train_features,train_labels )
test_dataset <- cbind(test_features,test_labels )

#Use merge function to merge feature definitions with feature keys (keys are from 1 to 6)
train_dataset <- merge(train_dataset, act_labels )
test_dataset <- merge(test_dataset,act_labels )

#Post merge, eliminate feature key from the dataset as we now have the feature descriptions
nn_train_data <- train_dataset
nn_test_data <- test_dataset

train_dataset <- subset(train_dataset, select = -c(act))
test_dataset <- subset(test_dataset, select = -c(act))

#Remove bad characters from column names, such as "(",")" and "-"
colnames(nn_train_data) = make.names(colnames(nn_train_data), unique = TRUE, allow_ = TRUE)
colnames(nn_test_data) = make.names(colnames(nn_test_data), unique = TRUE, allow_ = TRUE)

colnames(train_dataset) = make.names(colnames(train_dataset), unique = TRUE, allow_ = TRUE)
colnames(test_dataset) = make.names(colnames(test_dataset), unique = TRUE, allow_ = TRUE)


#########################################################
################################# Start classification.###
## METHOD 1.1 : NAIVE BAYES
NBModel <- naive_bayes(Activity ~ ., data=train_dataset)
test <- test_dataset
test$Activity <- NULL
nb1Pred <- predict(NBModel, test)
table(nb1Pred, test_dataset$Activity)
mean(as.character(nb1Pred) == as.character(test_dataset$Activity))

## METHOD 1.2 : NAIVE BAYES WITH LAPLACE SMOOTHENING
NBModel1 <- naive_bayes(Activity ~ ., data=train_dataset, fL=c(1,10))
nb2Pred <- predict(NBModel, test)
table(nb2Pred, test_dataset$Activity)
mean(as.character(nb2Pred) == as.character(test_dataset$Activity))


## METHOD 2.1 : LOGISTIC REGRESSION
# Convert dataframes into matrices
x <- model.matrix(Activity~., train_dataset)[,-1]
y = train_dataset$Activity
x_test <- model.matrix(Activity~., test_dataset)[,-1]

#Fit Multinomial logistic regression
fit = glmnet(x, y, family = "multinomial")
summary(fit)
pred_class <- predict(fit, newx = x_test, s = 0, type="class" )
#pred_score <- predict(fit, newx = x_test, s = NULL, type="response" )

#Confusion table for logistic regression
table(pred_class, test_dataset$Activity)
mean(as.character(pred_class) == as.character(test_dataset$Activity))
print(fit)
plot(fit)

## METHOD 2.2 : LOGISTIC REGRESSIONS WITH SPLINES
#Cross validation to arrive at LAMBDA. APLPHA = 1 indicates Spline method
cv.out <- cv.glmnet(x,y,alpha=1,family="multinomial",type.measure = "mse" )
plot(cv.out)

lambda_1se <- cv.out$lambda.1se

print("The lambda value for lowest LSE is is")
lambda_1se

#Predict using best lasso fit
lasso_prob <- predict(cv.out,newx = x_test,s=lambda_1se,type="response")
max_Levels <- colnames(lasso_prob)[apply(lasso_prob,1,which.max)]
table(max_Levels, test_dataset$Activity)
mean(as.character(max_Levels) == as.character(test_dataset$Activity))

## METHOD 3 : Neural Networks
#NOTE : This step did not converge for us in R. We were able to perform this in Python
layers=c(30,6)
form <- paste0("act ~ ", colnames(nn_train_data)[2])
for (i in 3:561)
   form<-paste0(form," + ", colnames(nn_train_data)[i])

nnModel1<-neuralnet(formula = formula(form),data=nn_train_data, hidden=layers,linear.output=T, err.fct="sse")


#### Perform PCA
pc <- prcomp(rbind(train_features,test_features), center=TRUE, scale=TRUE)
pc.var <- pc$sdev^2
pc.pvar <- pc.var/sum(pc.var)
plot(cumsum(pc.pvar),xlab="Principal component", ylab="Cumulative Proportion of variance explained",type='b',
     main="Principal Components proportions",col="blue")
abline(h=0.95)
abline(v=100)

#Additional - find best coefficients from Lasso
#Print coeefs
# coeff_list <- coef(cv.out,s=lambda_1se)
# 
# coeff_list[which(coeff_list != 0 ) ] 
# 
# cond <- sapply(coeff_list, function(x) x!= 0)
# cond
# coeff_list[cond]
# str(coeff_list)
# list.filter(coeff_list, x!=0)
# coef(cv.out, s="lambda.1se")[which(coef(cv.out, s="lambda.1se") != 0)] 
# #Print coeffs ends

