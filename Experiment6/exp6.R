install.packages("rpart")
install.packages("rpart.plot")

library(rpart)

titanic <- read.csv("C:\\Users\\student\\Documents\\AadarshSuresh\\exp6\\titanic.csv")  
titanic

titanic <- titanic[, c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived")]

titanic <- na.omit(titanic)

titanic$Sex <- as.factor(titanic$Sex)

tree_model <- rpart(Survived ~ ., data = titanic, method = "class")

print(tree_model)

library(rpart.plot)
prp(tree_model)

predictions <- predict(tree_model, newdata = titanic, type = "class")

confusion_matrix <- table(Actual = titanic$Survived, Predicted = predictions)
print(confusion_matrix)