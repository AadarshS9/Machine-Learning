#Bagging
install.packages("ipred")

library(ipred)
library(caret)

data(iris)
head(iris)

set.seed(123)

bagged_model<-bagging(Species ~ ., data = iris, nbagg = 50)
bagged_model

predictions<-predict(bagged_model,newdata = iris)
predictions

table(predictions,iris$Species)
#OR
confusionMatrix(predictions,iris$Species)
