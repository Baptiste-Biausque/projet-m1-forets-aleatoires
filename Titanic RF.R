set.seed(seed = 123)

## Importation et organisation des données ####

repertoire = "" # Chemin de votre dossier de travail
setwd(repertoire)
titanic = read.csv("titanic_complet.csv")

titanic$Survived = as.factor(titanic$Survived == 1)
titanic$Sex = as.factor(titanic$Sex)
titanic$Embarked = as.factor(titanic$Embarked)
titanic$Deck = as.factor(titanic$Deck)

library(tidyverse)

train = slice_sample(titanic, prop = 0.8)
test = anti_join(titanic, train)

## Arbres de décision ####

library(rpart)
library(rpart.plot)

# Arbre maximal

arbreMax = rpart(Survived ~ ., data = train, minsplit = 2, cp = 0)
plot(arbreMax)
text(arbreMax, xpd = TRUE, cex = 0.8)
plotcp(arbreMax)

# Arbre optimal

cpOpt = arbreMax$cptable[which.min(arbreMax$cptable[,4]),1]
arbreOpt = prune(arbreMax, cp = cpOpt)
plot(arbreOpt)
text(arbreOpt, xpd = TRUE, cex = 0.8)

# Arbre simplifié à 1 écart-type

seuil1SE <- sum(arbreMax$cptable[which.min(arbreMax$cptable[, 4]), 4:5])
cp1SE <- arbreMax$cptable[min(which(arbreMax$cptable[, 4] <= seuil1SE)), 1]
arbre1SE <- prune(arbreMax, cp = cp1SE)
rpart.plot(arbre1SE)

# Découpes de substitution

arbreSub = rpart(Survived ~ ., data = train, maxdepth = 1) # Pour montrer les découpes concurrentes du premier noeud
summary(arbreSub)

# Prédictions

library(caret)

test$Predicted = predict(arbreMax, test, type = "class")
confusionMatrix(data = test$Predicted, reference = test$Survived)

test$Predicted = predict(arbreOpt, test, type = "class")
confusionMatrix(data = test$Predicted, reference = test$Survived)

test$Predicted = predict(arbre1SE, test, type = "class")
confusionMatrix(data = test$Predicted, reference = test$Survived)

## Forêts aléatoires ####

library(randomForest)

# Forêt par défaut

ntree = 5000
model = randomForest(Survived ~ ., data = train, ntree = ntree, na.action = na.omit)

# Prédictions avec forêt initiale

test$Predicted = predict(model, test)
confusionMatrix(data = test$Predicted, reference = test$Survived)

# Erreur OOB et choix du nombre d'arbres

p = ggplot(data.frame(oob.times = model$oob.times), aes(x = oob.times)) + geom_histogram(colour = "black", fill = "white")
p = p + labs(title = "Histogramme du nombre de fois qu'une observation est \"out of bag\"", x = "Nombre de placements \"out of bag\"", y = "Nombre d'observations concernées")
p + theme(
  plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
  axis.title.x = element_text(size = 14),
  axis.title.y = element_text(size = 14)
)

errOOB = data.frame(ntree = 1:ntree, err.rate = model$err.rate[,1])
p = ggplot(data = errOOB, aes(x = ntree, y = err.rate)) + geom_line()
p = p + labs(title = "Erreur OOB selon le nombre d'arbres", x = "Nombre d'arbres", y = "Erreur OOB")
p + theme(
  plot.title = element_text(size = 18, hjust = 0.5, face = "bold"),
  axis.title.x = element_text(size = 14),
  axis.title.y = element_text(size = 14)
) # Stabilisation à partir de 2000 arbres

ntree = 2000

# Recherche du meilleur mtry

Nrep = 10
erreurs = rep(NA, 10)
for (i in 1:11) {
  erreurs[i] = mean(replicate(2, randomForest(Survived ~ ., data = train, ntree = ntree, mtry = i, na.action = na.omit)$err.rate[2000,1]))
}
mtry = which.min(erreurs)

# Prédictions avec la meilleure forêt

model = randomForest(Survived ~ ., data = train, ntree = ntree, mtry = mtry, na.action = na.omit)

test$Predicted = predict(model, test)
confusionMatrix(data = test$Predicted, reference = test$Survived)

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
varImpPlot(model, main = "Importance des variables (forêt aléatoire)")
