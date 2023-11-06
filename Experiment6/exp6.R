install.packages("mlbench")
install.packages("caret")
install.packages("ggplot2")
install.packages("e1071")

library(mlbench)
library(caret)
library(e1071)

soybean_df <- read.csv("D:\\S5\\ML\\exp6\\Soybean.csv")
nrow(soybean_df)
head(soybean_df)

#number of missing value cells
sum(is.na(soybean_df))
#omit rows with NA value cells
soybean <- na.omit(soybean_df)
nrow(soybean)
head(soybean)
soybean

#normalization or scaling
preproc <- preProcess(soybean[, -1], method = c("center", "scale"))
soybean[, -1] <- predict(preproc, soybean[, -1])
set.seed(123)  # For reproducibility

#split 80% to training and 20% to test data
splitIndex <- createDataPartition(soybean$Class, p = 0.8, list = FALSE)
training_data <- soybean[splitIndex, ]
nrow(training_data)
head(training_data)
training_data
testing_data <- soybean[-splitIndex, ]
nrow(testing_data)
head(testing_data)
testing_data

# Train the Naive Bayes classifier
nb_model <- naiveBayes(Class ~ ., data = training_data)
nb_model
# Make predictions on the testing data
predictions <- predict(nb_model, newdata = testing_data)
predictions
# Create a confusion matrix
predictions <- as.factor(predictions)
testing_data$Class<-as.factor(testing_data$Class)
confusion_matrix <- confusionMatrix(predictions, testing_data$Class)
# View the confusion matrix and associated metrics
print(confusion_matrix)
