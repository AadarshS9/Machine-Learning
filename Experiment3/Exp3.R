library(ggplot2)
ggplot(iris,aes(x=Petal.Length,y=Petal.Width))+
  geom_point(aes(color=factor(Species)))+
  labs(title="Scatter Plot of Petal Length VS. Petal Width")
print(iris)