#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# # 80/20 - pareto 
# Essas métricas vão resolver a maioria dos problemas

# In[ ]:


np.random.seed(50)
p = np.random.uniform(size=10)

y = np.random.uniform(size=10)
y_outlier = y.copy()
y_outlier[0] = 100


# # (R)MSE - (Root) Mean Squared Error - (Raiz Quadrada) do Erro Médio Quadrado
# - mesma unidade de medida original
# - maior penalidade para erros grandes, tende a ser mais impactado por outliers
# - minimizador = média - se eu usar p = média o erro é menor que p = mediana

# In[ ]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, p)
mse_outlier = mean_squared_error(y_outlier, p)
print("Sem outlier = {} | Com outlier = {}".format(np.sqrt(mse), np.sqrt(mse_outlier)))


# # (R)MSLE - (Root) Mean Squared Logarithmic Error - Raiz Quadrada do Erro Médio Logarítmico Quadrado
# - não pode ser negativo
# - mse se importa com a diferença "absoluta", msle se importa com a diferença "relativa"

# In[ ]:


from sklearn.metrics import mean_squared_log_error

print(np.log(10) - np.log(11))
print(np.log(110) - np.log(111))

msle = mean_squared_log_error(y, p)
msle_outlier = mean_squared_log_error(y_outlier, p)
print("Sem outlier = {} | Com outlier = {}".format(np.sqrt(msle), np.sqrt(msle_outlier)))


# In[ ]:


from matplotlib import pyplot as plt

y_ = np.linspace(0.5, 1, 1000)
sle = np.sqrt((np.log1p(1) - np.log1p(y_))**2)
ape = np.abs((1 - y_) / y_)

plt.plot(1-y_, ape), plt.plot(1-y_, sle)


# # MAE - Mean Absolute Error - Erro Médio Absoluto
# - minimizador = mediana
# - menos preocupado com outliers

# In[ ]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, p)
mae_outlier = mean_absolute_error(y_outlier, p)
print("Sem outlier = {} | Com outlier = {}".format(mae,mae_outlier))


# # MedAE - Median Absolute Error - Erro Mediano Absoluto
# - erro no percentil - percentil 50
# - preço de casas na Zillow - https://www.zillow.com/research/putting-accuracy-in-context-3255/
# - metade dos erros do modelo é menor que o valor deste erro

# In[ ]:


from sklearn.metrics import median_absolute_error

medae = median_absolute_error(y, p)
medae_outlier = median_absolute_error(y_outlier, p)
print("Sem outlier = {} | Com outlier = {}".format(medae, medae_outlier))


# # MAPE - Mean Absolute Percentage Error - Erro Médio Percentual Absoluto
# - https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

# In[ ]:


def mape(y_true, y_pred):
  return np.mean( np.abs( (y_true - y_pred) / y_true ) ) 

mape_ = mape(y, p)
mape_outlier = mape(y_outlier, p)
print("Sem outlier = {} | Com outlier = {}".format(mape_, mape_outlier))


# # R2 - R-squared - R-quadrado
# - https://data.library.virginia.edu/is-r-squared-useless/

# In[ ]:


# winsorizar


# # winsorizar

# In[ ]:




