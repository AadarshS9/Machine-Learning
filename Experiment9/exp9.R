install.packages("e1071") 
install.packages("caTools") 
install.packages("class")
install.packages("caret")

library(e1071) 
library(caTools) 
library(class) 
library(caret)

head(iris) 

split <- sample.split(iris, SplitRatio = 0.8) 
train_cl <- subset(iris, split == "TRUE") 
test_cl <- subset(iris, split == "FALSE")

nrow(iris)
nrow(train_cl)
nrow(test_cl)

train_scale <- scale(train_cl[, 1:4]) 
test_scale <- scale(test_cl[, 1:4]) 

#K = 3
classifier_knn <- knn(train = train_scale,test = test_scale,cl = train_cl$Species,k = 3)
classifier_knn
misClassError <- mean(classifier_knn != test_cl$Species) 
print(paste('Accuracy =', 1-misClassError)) 

cm <- table(test_cl$Species, classifier_knn) 
cm 

confusionMatrix(classifier_knn,test_cl$Species)

#K = 5
classifier_knn <- knn(train = train_scale,test = test_scale,cl = train_cl$Species,k = 5)
classifier_knn
misClassError <- mean(classifier_knn != test_cl$Species) 
print(paste('Accuracy =', 1-misClassError)) 

confusionMatrix(classifier_knn,test_cl$Species)
