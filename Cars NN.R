set.seed(123)

## Importation et organisation des données ####

repertoire = "" # Chemin de votre dossier de travail
setwd(repertoire)
cars = read.csv("cars_complet.csv")

cars$model = as.factor(cars$model)
listeModel = levels(cars$model)

cars$transmission = as.factor(cars$transmission)
listeTransmission = levels(cars$transmission)
listeTransmission[listeTransmission == "Other"] = "Other transmission"

cars$fuelType = as.factor(cars$fuelType)
listeFuelType = levels(cars$fuelType)
listeFuelType[listeFuelType == "Other"] = "Other fuel"

cars$brand = as.factor(cars$brand)
listeBrand = levels(cars$brand)

# Encodage des données catégorielles

library(tensorflow)
library(keras)

model = as.data.frame(to_categorical(as.numeric(cars$model))[, -1])
names(model) = listeModel
model$index = as.numeric(rownames(model))

transmission = as.data.frame(to_categorical(as.numeric(cars$transmission))[, -1])
names(transmission) = listeTransmission
transmission$index = as.numeric(rownames(transmission))

fuelType = as.data.frame(to_categorical(as.numeric(cars$fuelType))[, -1])
names(fuelType) = listeFuelType
fuelType$index = as.numeric(rownames(fuelType))

brand = as.data.frame(to_categorical(as.numeric(cars$brand))[, -1])
names(brand) = listeBrand
brand$index = as.numeric(rownames(brand))

cars = cars[, !colnames(cars) %in% c("model", "transmission", "fuelType", "brand")]
cars$index = as.numeric(rownames(cars))
cars = merge(cars, transmission)
cars = merge(cars, fuelType)
cars = merge(cars, brand)
cars = merge(cars, model)
cars$index = NULL

remove(brand, listeBrand, fuelType, listeFuelType, model, listeModel, transmission, listeTransmission)

# Séparation apprentissage et test

library(tidyverse)

train_set = slice_sample(cars, prop = 0.8)
test_set = anti_join(cars, train_set)

# Mise à l'échelle des données

maxi = apply(cars, 2, max)
mini = apply(cars, 2, min)

train = scale(as.matrix(train_set), center = mini, scale = (maxi - mini))
test = scale(as.matrix(test_set), center = mini, scale = (maxi - mini))

# Séparation x et y

X_train = train[, colnames(train) != "price"]
y_train = train[, "price"]
X_test = test[, colnames(test) != "price"]
y_test = test[, "price"]

## Réseau de neurones ####

model = keras_model_sequential()
model %>%
  layer_dense(units = 25, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

model %>%
  compile(loss = "mean_squared_error", optimizer = "adam")

fit = fit(model, x = X_train, y = y_train, epochs = 30, batch_size = 128, verbose = 1, validation_data = list(x_val = X_test, y_val = y_test))

# Prédictions

y_pred = model %>% predict(X_test)

library(DMwR)
y_pred = unscale(y_pred, norm.data = test, col.ids = 2)

library(caret)
postResample(pred = y_pred, obs = test_set[, "price"])
