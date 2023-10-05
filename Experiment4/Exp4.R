library(ggplot2)

setwd(file.path("C:\\Users\\student\\Documents\\Aadarsh\\Experiment3and4"))
titanic_df <- read.csv("titanic.csv")
ggplot(data = titanic_df, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  labs(
    title = "Histogram of Age in Titanic Dataset",
    x = "Age",
    y = "Frequency"
  ) +
  theme_minimal()