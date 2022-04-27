set.seed(123)

## Importation et organisation des données ####

repertoire = "" # Chemin de votre dossier de travail
setwd(repertoire)
cars = read.csv("cars_complet.csv")

cars$model = as.factor(cars$model)
cars$transmission = as.factor(cars$transmission)
cars$fuelType = as.factor(cars$fuelType)
cars$brand = as.factor(cars$brand)

library(tidyverse)

train = slice_sample(cars, prop = 0.8)
test = anti_join(cars, train)

## Arbres CART ####

library(rpart)
library(rpart.plot)

# Arbre maximal

arbreMax = rpart(price ~ ., data = train, cp = 0)
plot(arbreMax)
plotcp(arbreMax)

# Arbre optimal

cpOpt = arbreMax$cptable[which.min(arbreMax$cptable[,4]),1]
arbreOpt = prune(arbreMax, cp = cpOpt)
plot(arbreOpt)

# Arbre simplifié à 1 écart-type

seuil1SE <- sum(arbreMax$cptable[which.min(arbreMax$cptable[, 4]), 4:5])
cp1SE <- arbreMax$cptable[min(which(arbreMax$cptable[, 4] <= seuil1SE)), 1]
arbre1SE <- prune(arbreMax, cp = cp1SE)
plot(arbre1SE)

# Prédictions

library(caret)

test$predicted = predict(arbreMax, test, type = "vector")
postResample(pred = test$predicted, obs = test$price)

test$predicted = predict(arbreOpt, test, type = "vector")
postResample(pred = test$predicted, obs = test$price)

test$predicted = predict(arbre1SE, test, type = "vector")
postResample(pred = test$predicted, obs = test$price)

## Forêts aléatoires ####

library(randomForest)

# Adaptation des données : suppression des modèles de voitures

cars$model = NULL
train$model = NULL
test$model = NULL

# Forêt par défaut

ntree = 50
model = randomForest(price ~ ., data = train, ntree = ntree, do.trace = TRUE) # très long à exécuter

# Prédictions avec la première forêt

test$predicted = predict(model, test)
postResample(pred = test$predicted, obs = test$price)

# Erreur OOB / score r^2 et choix du nombre d'arbres

errOOB = data.frame(ntree = 1:ntree, mse = model$mse)
p = ggplot(data = errOOB, aes(x = ntree, y = mse)) + geom_line()
p = p + labs(title = "Erreur OOB selon le nombre d'arbres", x = "Nombre d'arbres", y = "Erreur OOB")
p + theme(
  plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
  axis.title.x = element_text(size = 14),
  axis.title.y = element_text(size = 14)
)

score = data.frame(ntree = 1:ntree, rsq = model$rsq)
p = ggplot(data = score, aes(x = ntree, y = rsq)) + geom_line()
p = p + labs(title = "Score r^2 selon le nombre d'arbres", x = "Nombre d'arbres", y = "r^2")
p + theme(
  plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
  axis.title.x = element_text(size = 14),
  axis.title.y = element_text(size = 14)
) # Satisfaisant à partir de 30 arbres

ntree = 30

# Prédictions avec la forêt optimale

model = randomForest(price ~ ., data = train, ntree = ntree, do.trace = TRUE) # très long à exécuter
test$predicted = predict(model, test)
postResample(pred = test$predicted, obs = test$price)

## Importance des variables ####

# Au sens des arbres

varImp = data.frame(x = names(arbreOpt$variable.importance), y = arbreOpt$variable.importance)
p = ggplot(varImp, aes(x, y)) + geom_bar(stat = "identity")
p = p + labs(title = "Diagramme à bâtons de l'importance des variables", x = "Variables", y = "Mesure de l'importance")
p + theme(
  plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
  axis.title.x = element_text(size = 14),
  axis.title.y = element_text(size = 14)
)

# Au sens des forêts

model$importance
varImpPlot(model, main = "Importance des variables (forêts aléatoires)")
