df1<-read.csv("C:/Users/byabh/Desktop/ALT/project/train.csv")

df2 <- read.csv("C:/Users/byabh/Desktop/ALT/project/test.csv")

train<-as.data.frame(df1)
test<-as.data.frame(df2)

summary(train$subject)

train$Partition = "Train"
test$Partition = "Test"

library(ggplot2)
library(FSelector) 
all = rbind(train,test)

all$Partition = as.factor(all$Partition)

qplot(data = all, x = subject, fill = Partition)

qplot(data = all , x = subject, fill = Activity)


p <- ggplot(all, aes(x=subject, y = Activity)) + geom_boxplot ()
plot(all$Activity)
