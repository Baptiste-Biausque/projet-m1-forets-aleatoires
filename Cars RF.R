set.seed(123)

## Importation et organisation des donn�es ####

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

# Arbre simplifi� � 1 �cart-type

seuil1SE <- sum(arbreMax$cptable[which.min(arbreMax$cptable[, 4]), 4:5])
cp1SE <- arbreMax$cptable[min(which(arbreMax$cptable[, 4] <= seuil1SE)), 1]
arbre1SE <- prune(arbreMax, cp = cp1SE)
plot(arbre1SE)

# Pr�dictions

library(caret)

test$predicted = predict(arbreMax, test, type = "vector")
postResample(pred = test$predicted, obs = test$price)

test$predicted = predict(arbreOpt, test, type = "vector")
postResample(pred = test$predicted, obs = test$price)

test$predicted = predict(arbre1SE, test, type = "vector")
postResample(pred = test$predicted, obs = test$price)

## For�ts al�atoires ####

library(randomForest)

# Adaptation des donn�es : suppression des mod�les de voitures

cars$model = NULL
train$model = NULL
test$model = NULL

# For�t par d�faut

ntree = 50
model = randomForest(price ~ ., data = train, ntree = ntree, do.trace = TRUE) # tr�s long � ex�cuter

# Pr�dictions avec la premi�re for�t

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
) # Satisfaisant � partir de 30 arbres

ntree = 30

# Pr�dictions avec la for�t optimale

model = randomForest(price ~ ., data = train, ntree = ntree, do.trace = TRUE) # tr�s long � ex�cuter
test$predicted = predict(model, test)
postResample(pred = test$predicted, obs = test$price)

## Importance des variables ####

# Au sens des arbres

varImp = data.frame(x = names(arbreOpt$variable.importance), y = arbreOpt$variable.importance)
p = ggplot(varImp, aes(x, y)) + geom_bar(stat = "identity")
p = p + labs(title = "Diagramme � b�tons de l'importance des variables", x = "Variables", y = "Mesure de l'importance")
p + theme(
  plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
  axis.title.x = element_text(size = 14),
  axis.title.y = element_text(size = 14)
)

# Au sens des for�ts

model$importance
varImpPlot(model, main = "Importance des variables (for�ts al�atoires)")
