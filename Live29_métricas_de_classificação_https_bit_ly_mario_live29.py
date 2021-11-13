#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# 80/20 - pareto
# - Essas métricas vão resolver a maioria dos problemas

# In[ ]:


np.random.seed(50)
p_binary = np.random.uniform(size=10) #mdl.predict_proba
p_multi = np.random.uniform(size=(10,3))

y_binary = (np.random.uniform(size=10) > 0.5).astype(int)
y_multi = np.random.uniform(size=(10, 3)).argmax(axis=1)

p_binary_threshold = (p_binary > 0.5).astype(int)
p_multi_argmax = p_multi.argmax(axis=1)


# # Precisam de ponto de corte
# - mdl.predict

# ## Acurácia
# - não use "oficialmente", apenas "preguiçosamente"
# - inadequada para dados desequilibrados
# - 99,76% - não são spam

# In[ ]:


from sklearn.metrics import accuracy_score

print("P = {}\nY = {}".format(p_binary_threshold, y_binary))

accuracy_score(y_binary, p_binary_threshold)


# # Precision
# - dos casos que eu previ como positivos (para uma classe) quantos realmente são?
# - Envio de cupons de desconto, custos diferentes para cada erro.
# - Ex: se custa caro mandar a promoção, das pessoas que eu previ que iam comprar, quantas compraram?
# 

# In[ ]:


from sklearn.metrics import precision_score
print("P = {}\nY = {}".format(p_binary_threshold, y_binary))

precision_score(y_binary, p_binary_threshold)


# # Recall
# - dos que eram realmente positivos (para uma classe) quantos eu detectei?
# - taxa de detecção
# - https://en.wikipedia.org/wiki/Confusion_matrix

# In[ ]:


from sklearn.metrics import recall_score
print("P = {}\nY = {}".format(p_binary_threshold, y_binary))

recall_score(y_binary, p_binary_threshold)


# # F1 Score
# - média harmônica entre os dois  
# ( 2 * precision * recall ) / (precision + recall) 
# https://en.wikipedia.org/wiki/F1_score 
# 

# In[ ]:


from sklearn.metrics import f1_score

print("P = {}\nY = {}".format(p_binary_threshold, y_binary))
#(2 * .67 * .2857) / (.67 + .2857)

f1_score(y_binary, p_binary_threshold)


# In[ ]:





# # Kappa
# - It is generally thought to be a more robust measure than simple percent agreement calculation, as κ takes into account the possibility of the agreement occurring by chance
# - leia mais sobre ela

# In[ ]:


from sklearn.metrics import cohen_kappa_score

print("P = {}\nY = {}".format(p_multi_argmax, y_multi))

cohen_kappa_score(y_multi, p_multi_argmax)


# In[ ]:





# # Avalia a probabilidadade diretamente (sem ponto de corte)
# - mdl.predict_proba

# # Log Loss
# - calculada para a probabilidade empírica do evento. Proporção que o evento ocorre na vida real
# - Se o time A jogar contra o time B e tiver 40% de chances de ganhar, se jogarem 10 vezes, 4 vezes o time A vai ganhar.
# - A log loss estará na mínima quando o modelo prever 0.4

# In[ ]:


from sklearn.metrics import log_loss
print("P = {}\nY = {}".format(p_binary, y_binary))

p_random = np.ones(10) * 0.5

log_loss(y_binary, p_binary)


# # ROC AUC
# - qual é a chance de um exemplo positivo ter um score (previsão) maior do que um negativo?
# - bom quando garantir que positivos sejam rankeados acima dos negativos é mais importante do que prever a probabilidade real do evento

# In[ ]:


from sklearn.metrics import roc_auc_score

print("P = {}\nY = {}".format(p_binary, y_binary))

roc_auc_score(y_binary, p_binary)


# In[ ]:


sum_over = 0
total = 100000

for i in range(total):
  caixa_de_positivos = p_binary[y_binary == 1]
  caixa_de_negativos = p_binary[y_binary == 0]

  positivo = np.random.choice(caixa_de_positivos, size=1, replace=False)
  negativo = np.random.choice(caixa_de_negativos, size=1, replace=False)

  if positivo > negativo:
    sum_over += 1
sum_over / total


# # AUPRC - Area Under the Precision-Recall Curve
# - acho mais estável e mais fácil de interpretar
# - AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight:

# In[ ]:


from sklearn.metrics import average_precision_score
print("P = {}\nY = {}".format(p_binary, y_binary))

average_precision_score(y_binary, p_binary)


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_multi, p_multi_argmax))


# In[ ]:




