# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 14:53:16 2020

@author: alarc
"""

from scipy import stats
import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


# Datos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train = pd.read_csv(r"C:\Users\alarc\OneDrive\Documents\Programming\Python\Jupyter\Inteligencia Artificial\Proyecto 1\house-prices-advanced-regression-techniques/train.csv",
                    index_col="Id")
# train.head()
test = pd.read_csv(r"C:\Users\alarc\OneDrive\Documents\Programming\Python\Jupyter\Inteligencia Artificial\Proyecto 1\house-prices-advanced-regression-techniques/test.csv",
                    index_col="Id")
sample = pd.read_csv(r"C:\Users\alarc\OneDrive\Documents\Programming\Python\Jupyter\Inteligencia Artificial\Proyecto 1\house-prices-advanced-regression-techniques/sample_submission.csv",
                    index_col="Id")
# test.head()
# type(test)

# Matriz de Correlación ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Gráfico con leyendas
figure = plt.figure() 
axes = figure.add_subplot(111) 
# using the matshow() function  
caxes = axes.matshow(train.corr()) 
figure.colorbar(caxes) 
# Hacemos el gráfico  
plt.show() 

MC=train.corr()
#type(MC)
SaleCorr=abs(MC.loc[:,"SalePrice"])
#type(SaleCorr)
aux=SaleCorr.sort_values(ascending=False)
Explicativas=aux.index[aux>0.6]
#type(aux.index)

# Con base en la correlación, vamos a seleccionar las que tengan
# una > 0.6 con la variable respuesta.
train_selec = train[Explicativas]

# Vamos a ver la correlación de las variables que deseamos.

# Gráfico con leyendas
figure = plt.figure() 
axes = figure.add_subplot(111) 
# using the matshow() function  
caxes = axes.matshow(train_selec.corr()) 
figure.colorbar(caxes) 
# Ponemos las leyendas
#axes.set_xticklabels(['']+train_selec.columns) 
# Le insertamos un cero al principio para que lea las leyendas
axes.set_yticklabels(train_selec.columns.insert(0,0)) 
# Hacemos el gráfico  
plt.show() 

# Vemos que hay mucha multicolinealidad entre las parejas de variables
# "GarageCars"-"GarageArea" y "TotalBsmtSF"-"1stFlrSF". Entonces las
# vamos a quitar como sigue
train_selec=train_selec.drop(columns=['GarageCars', '1stFlrSF'])

# Gráfico con leyendas
figure = plt.figure() 
axes = figure.add_subplot(111) 
# using the matshow() function  
caxes = axes.matshow(train_selec.corr()) 
figure.colorbar(caxes) 
# Ponemos las leyendas
#axes.set_xticklabels(['']+train_selec.columns) 
# Le insertamos un cero al principio para que lea las leyendas
axes.set_yticklabels(train_selec.columns.insert(0,0)) 
# Hacemos el gráfico  
plt.show() 

# Ponemos las variables respusta en X
X = train_selec.drop(columns=['SalePrice'])
y = train_selec['SalePrice']

# Creación de Modelos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Lineal
linear_model = LinearRegression(normalize=False,fit_intercept=True)
linear_model.fit(X,y)
y_linear=linear_model.predict(X)

# Lasso
lasso_model = Lasso(alpha=0.5,normalize=True, max_iter=1e6)
lasso_model.fit(X,y)
y_lasso=lasso_model.predict(X)

# Ridge
ridge_model = Ridge(alpha=0.5)
ridge_model.fit(X,y) 
y_ridge=ridge_model.predict(X)

# Preparamos la información que deseamos predecir. ~~~~~~~~~~~~~~~~~~~~~~

# Filtramos las covariables que vamos a utilizar
test_selec = test[X.columns]
# Procedemos a quitar algunos datos faltantes.
is_NaN = test_selec.isnull()
row_has_NaN = is_NaN.any(axis=1)
# Nos quedamos con las que NO son NA
test_selec = test_selec[~row_has_NaN]
Xs = test_selec[~row_has_NaN]
ys = sample[~row_has_NaN]
ysd=ys.describe()

# Esta es otra opción pero no la usaremos
#test_selec = test_selec.dropna()
#test_selec.describe()


# Modelo lineal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# coeficientes
linear_model.coef_
# Veamos entonces la comparación de la predicción con la real
plt.scatter(y,y_linear)
plt.plot(y, y, '-',color="red",label="Recta Identidad")
plt.title('Training - Modelo lineal')
plt.xlabel('y')
plt.ylabel('y.gorro')
plt.legend(loc='best', frameon=False)
plt.show()

# Ahora vamos a predecir con el conjunto de prueba.

# Luego hacemos la predicción del modelo
ys_hat=linear_model.predict(Xs)
ys_hat=pd.DataFrame(data=ys_hat)

# Pequeño resumen de los datos
ysd
ys_hat.describe()

# Graficamos
plt.scatter(ys,ys_hat)
plt.plot(ys, ys, '-',color="red",label="Recta Identidad")
plt.title('Test - Modelo lineal')
plt.xlabel('y')
plt.ylabel('y.gorro')
plt.legend(loc='best', frameon=False)
plt.show()

# Modelo Lasso ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# coeficientes
lasso_model.coef_
# Veamos entonces la comparación de la predicción con la real
plt.scatter(y,y_lasso)
plt.plot(y, y, '-',color="red",label="Recta Identidad")
plt.title('Training - Modelo lasso')
plt.xlabel('y')
plt.ylabel('y.gorro')
plt.legend(loc='best', frameon=False)
plt.show()

# Ahora vamos a predecir con el conjunto de prueba.

# Luego hacemos la predicción del modelo
ys_hat=lasso_model.predict(Xs)
ys_hat=pd.DataFrame(data=ys_hat)

# Pequeño resumen de los datos
ysd
ys_hat.describe()

# Graficamos
plt.scatter(ys,ys_hat)
plt.plot(ys, ys, '-',color="red",label="Recta Identidad")
plt.title('Test - Modelo lasso')
plt.xlabel('y')
plt.ylabel('y.gorro')
plt.legend(loc='best', frameon=False)
plt.show()

# Modelo Ridge ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# coeficientes
ridge_model.coef_
# Veamos entonces la comparación de la predicción con la real
plt.scatter(y,y_ridge)
plt.plot(y, y, '-',color="red",label="Recta Identidad")
plt.title('Training - Modelo ridge')
plt.xlabel('y')
plt.ylabel('y.gorro')
plt.legend(loc='best', frameon=False)
plt.show()

# Ahora vamos a predecir con el conjunto de prueba.

# Luego hacemos la predicción del modelo
ys_hat=ridge_model.predict(Xs)
ys_hat=pd.DataFrame(data=ys_hat)

# Pequeño resumen de los datos
ysd
ys_hat.describe()

# Graficamos
plt.scatter(ys,ys_hat)
plt.plot(ys, ys, '-',color="red",label="Recta Identidad")
plt.title('Test - Modelo ridge')
plt.xlabel('y')
plt.ylabel('y.gorro')
plt.legend(loc='best', frameon=False)
plt.show()



# Graficamos
plt.scatter(y_linear,y_lasso)
plt.plot(y_linear, y_linear, '-',color="red",label="Recta Identidad")
plt.title('Test - Modelo ridge')
plt.xlabel('y')
plt.ylabel('y.gorro')
plt.legend(loc='best', frameon=False)
plt.show()
