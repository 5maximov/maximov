library('GGally')
library('lmtest')
library('FNN')
library('ISLR')
data(Auto)

pairs(Auto)

# константы
my.seed <- 12345
train.percent <- 0.85


Auto.1 <- data.frame(mpg = Auto$mpg, 
                     weight = Auto$weight, 
                     displacement = Auto$displacement,
                     horsepower = Auto$horsepower, 
                     cylinders = Auto$cylinders) 

Auto.1$cylinders <- as.factor(Auto.1$cylinders)

# обучающая выборка
set.seed(my.seed)
inTrain <- sample(seq_along(Auto.1$mpg), 
                  nrow(Auto.1) * train.percent)
df.train <- Auto.1[inTrain, c(colnames(Auto.1)[-1], colnames(Auto.1)[1])]
df.test <- Auto.1[-inTrain, -1]


# описательные статистики по переменным
summary(df.train)

# совместный график разброса переменных
ggpairs(df.train)

# цвета по фактору 
ggpairs(df.train[, c('weight', 'cylinders', 'displacement', 'horsepower')], 
        mapping = ggplot2::aes(color = cylinders))

# модели
model.1 <- lm(mpg ~ . + cylinders:weight + cylinders:displacement + cylinders:horsepower,
              data = df.train)
summary(model.1)


model.2 <- lm(mpg ~ . + cylinders:displacement + cylinders:horsepower,
              data = df.train)
summary(model.2)


model.3 <- lm(mpg ~ . + cylinders:horsepower,
              data = df.train)
summary(model.3)


model.4 <- lm(mpg ~ weight + displacement + horsepower,
              data = df.train)
summary(model.4)


model.5 <- lm(mpg ~ weight + horsepower,
              data = df.train)
summary(model.5)

# тест Бройша-Пагана
bptest(model.5)

# статистика Дарбина-Уотсона
dwtest(model.5)

# графики остатков
par(mar = c(4.5, 4.5, 2, 1))
par(mfrow = c(1, 3))
plot(model.5, 1)
plot(model.5, 4)
plot(model.5, 5)
par(mfrow = c(1, 1))


# фактические значения y на тестовой выборке
y.fact <- Auto.1[-inTrain, 1]
y.model.lm <- predict(model.5, df.test)
MSE.lm <- sum((y.model.lm - y.fact)^2) / length(y.model.lm)

# kNN требует на вход только числовые переменные
df.train.num <- as.data.frame(apply(df.train, 2, as.numeric))
df.test.num <- as.data.frame(apply(df.test, 2, as.numeric))

for (i in 2:50){
  model.knn <- knn.reg(train = df.train.num[, !(colnames(df.train.num) %in% 'mpg')], 
                       y = df.train.num[, 'mpg'], 
                       test = df.test.num, k = i)
  y.model.knn <- model.knn$pred
  if (i == 2){
    MSE.knn <- sum((y.model.knn - y.fact)^2) / length(y.model.knn)
  } else {
    MSE.knn <- c(MSE.knn, 
                 sum((y.model.knn - y.fact)^2) / length(y.model.knn))
  }
}

# график
par(mar = c(4.5, 4.5, 1, 1))
plot(2:50, MSE.knn, type = 'b', col = 'darkgreen',
     xlab = 'значение k', ylab = 'MSE на тестовой выборке')
lines(2:50, rep(MSE.lm, 49), lwd = 2, col = grey(0.2), lty = 2)
legend('bottomright', lty = c(1, 2), pch = c(1, NA), 
       col = c('darkgreen', grey(0.2)), 
       legend = c('k ближайших соседа', 'регрессия (все факторы)'), 
       lwd = rep(2, 2))

# Тест Бройша-Пагана: p-value = 1.782e-05<0.05 - проявилась гетероскедастичность.
# Статистика Дарбина-Уотсона: p-value = 0.9506 - нулевая гипотеза не отвергается.
# Так как т.Бройша-Пагана и статистика Дарбина-Уотсона удовлетворительны, то построенная модель регрессии пригодна для прогнозирования.