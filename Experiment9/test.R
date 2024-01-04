
library(caret)
library(class)


data(iris)

set.seed(123)
#since 5 th columnn is Species We need not standardize(centering+scaling) it
#standardize all columns from 1-4
preproc <- preProcess(iris[, -5], method = c("center", "scale"))

iris[, -5] <- predict(preproc, iris[, -5])

split_index <- createDataPartition(iris$Species, p = 0.8, list = FALSE)

training_data <- iris[split_index, ]
testing_data <- iris[-split_index, ]

classifier_knn <- knn(train = training_data[, -5], test = testing_data[, -5], cl = training_data$Species, k = 3)

misClassError <- mean(classifier_knn != testing_data$Species) 

print(paste('Accuracy =', 1 - misClassError)) 

confusionMatrix(testing_data$Species,classifier_knn)
