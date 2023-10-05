install.packages("mlbench")
install.packages("caret")
install.packages("ggplot2")
install.packages("randomForest")

library(mlbench)
library(caret)

soybean_df <- read.csv("C:\\Users\\student\\Documents\\Aadarsh\\Experiment5and6\\Soybean.csv")
soybean_df

#number of missing value cells
sum(is.na(soybean_df))
#omit rows with NA value cells
soybean <- na.omit(soybean_df)
soybean

preproc <- preProcess(soybean[, -1], method = c("center", "scale"))
soybean[, -1] <- predict(preproc, soybean[, -1])
set.seed(123)  # For reproducibility
#split 80% to training and 20% to test data
splitIndex <- createDataPartition(soybean$Class, p = 0.8, list = FALSE)
training_data <- soybean[splitIndex, ]
training_data
testing_data <- soybean[-splitIndex, ]
testing_data



# Load required packages
library(randomForest)
# Train a Random Forest classifier
model <- train(Class ~ ., data = training_data, method = "rf")
model
# Make predictions on the testing data
predictions <- predict(model, newdata = testing_data)
predictions <- as.factor(predictions)
predictions
testing_data$Class<-as.factor(testing_data$Class)
typeof(testing_data$Class)
typeof(predictions)
# Evaluate the model's performance
confusionMatrix(predictions, testing_data$Class)







install.packages("e1071")
library(e1071)
# Train the Naive Bayes classifier
nb_model <- naiveBayes(Class ~ ., data = training_data)
# Make predictions on the testing data
predictions <- predict(nb_model, newdata = testing_data)
# Load the caret package (if not already loaded)
library(caret)
# Create a confusion matrix
confusion_matrix <- confusionMatrix(predictions, testing_data$Class)
# View the confusion matrix and associated metrics
print(confusion_matrix)

