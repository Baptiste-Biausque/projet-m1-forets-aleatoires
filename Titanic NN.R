set.seed(123) # Pour obtenir les m�mes �chantillons train et test

## Importation et organisation des donn�es ####

repertoire = "" # Chemin de votre dossier de travail
setwd(repertoire)
titanic = read.csv("titanic_complet.csv")

titanic$Survived = as.factor(titanic$Survived == 1)
titanic$Sex = as.factor(titanic$Sex)
titanic$Embarked = as.factor(titanic$Embarked)
titanic$Deck = as.factor(titanic$Deck)

# Encodage des donn�es cat�gorielles

titanic$Survived = as.numeric(titanic$Survived) - 1
titanic$Died = 1 - titanic$Survived
titanic$Sex = as.numeric(titanic$Sex) - 1
titanic$Embarked = as.numeric(titanic$Embarked)
titanic$Deck = as.numeric(titanic$Deck)

# S�paration apprentissage et test

library(tidyverse)

train = slice_sample(titanic, prop = 0.8)
test = anti_join(titanic, train)

# Mise � l'�chelle des donn�es

maxi = apply(titanic, 2, max)
mini = apply(titanic, 2, min)

train = as.data.frame(scale(train, center = mini, scale = (maxi - mini)))
test = as.data.frame(scale(test, center = mini, scale = (maxi - mini)))

# S�paration x et y

X_train = as.matrix(train[, !colnames(train) %in% c("Survived", "Died")])
y_train = as.matrix(train[, c("Survived", "Died")])
X_test = as.matrix(test[, !colnames(train) %in% c("Survived", "Died")])
y_test = as.matrix(test[, c("Survived", "Died")])

## R�seau de neurones ####

library(tensorflow)
library(keras)

model = keras_model_sequential()
model %>%
  layer_dense(units = 15, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dense(units = ncol(y_train), activation = "sigmoid")

model %>%
  compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = c("categorical_accuracy"))

fit = fit(model, x = X_train, y = y_train, epochs = 10, batch_size = 1, verbose = 1, validation_data = list(x_val = X_test, y_val = y_test))

# Pr�dictions

library(caret)

y_pred = as.factor(1 - as.numeric(model %>% predict(X_test) %>% k_argmax()))
confusionMatrix(data = y_pred, reference = as.factor(y_test[,1]))
