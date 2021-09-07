#!/usr/bin/env python
# coding: utf-8

# **[Introduction to Machine Learning Home Page](https://www.kaggle.com/learn/intro-to-machine-learning)**
# 
# ---
# 

# ## Recap
# You've built your first model, and now it's time to optimize the size of the tree to make better predictions. Run this cell to set up your coding environment where the previous step left off.

# In[1]:


# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = 'dados/Iowa House Prices/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Crie a variável alvo e chame de  y
y = home_data.SalePrice
# Selecione as outras features e chame de X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Quebre os dados em validação e treinamento
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Especifique o Modelo (uma árvore de decisão regressora)
iowa_model = DecisionTreeRegressor(random_state=1)
# Faz o fit no Modelo
iowa_model.fit(train_X, train_y)

# Faz previsões na validação e calcula o mean absolute error (erro médio absouto pra ver se está prestando)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))


# # Exercises
# Escreva uma função `get_mae` pra chamar de sua. Por enquanto vamos fornecer uma. Esta é a mesma função que vc ouviu falar na aula anterior. Simplesmnte rode a celula abaixo.

# In[2]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# ## Step 1: Compare Different Tree Sizes
# Write a loop that tries the following values for *max_leaf_nodes* from a set of possible values.
# 
# Call the *get_mae* function on each value of max_leaf_nodes. Store the output in some way that allows you to select the value of `max_leaf_nodes` that gives the most accurate model on your data.

# In[3]:


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes

pontuacao = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

# um modo didático tirado daqui https://evansyangs.codes/kaggle-wei-ke-cheng/02-05-underfitting-and-overfitting.html#step-1-compare-different-tree-sizes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Faz loop nas seis quantidades de nodes, a cada quantidade mede desempenho
# e selecionar o melhor
mae_list=[]
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae_list.append(get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y))
min_mae=min(mae_list)
min_mae_index=mae_list.index(min_mae)


# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = candidate_max_leaf_nodes[min_mae_index]
#print(best_tree_size)
# oficial da kaggle
#scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
#Here you put in a dictionary the mean absolute error of your model for each number of leaf in candidate_max_leaf_nodes.
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
# oficial
#best_tree_size = min(scores, key=scores.get)


# In[4]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes

pontuacao = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}


#um modo didático tirado daqui https://evansyangs.codes/kaggle-wei-ke-cheng/02-05-underfitting-and-overfitting.html#step-1-compare-different-tree-sizes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Vamos fazer loop para as 6 quantidades de nodes, a cada quantidade vamos medir o desempenho e selecionar o melhor
#
#cria lista vazia
mae_list=[]
#intera entre os 6 candidatos 
for max_leaf_nodes in candidate_max_leaf_nodes:
    #faz append no resultado do get_mae (é um mean absolute error)
    mae_list.append(get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y))
#pega o menor item da lista de erros
min_mae=min(mae_list)
#aqui ele pede o index la do mae_list referente ao min_mae e vem que o terceiro item é o cara
min_mae_index=mae_list.index(min_mae)




# Aqui ele descobre quem é o terceiro cara  5, 25, 50, 100, 250 ou 500, como o index começa em zero ele é o 100
best_tree_size = candidate_max_leaf_nodes[min_mae_index]
print (best_tree_size)
#print (min_mae_index)


# In[5]:


# The lines below will show you a hint or the solution.
# step_1.hint() 
# step_1.solution()


# ## Step 2: Fit Model Using All Data
# You know the best tree size. If you were going to deploy this model in practice, you would make it even more accurate by using all of the data and keeping that tree size.  That is, you don't need to hold out the validation data now that you've made all your modeling decisions.

# In[6]:


# Fill in argument to make optimal size and uncomment# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)



# final_model = DecisionTreeRegressor(____)

# fit the final model and uncomment the next two lines
# final_model.fit(____, ____)


# In[7]:





# 
# # Foward selection
# 
# testando features uma por uma e vendo qual a melhor
# 
# codigo tirado das series temporais do mario filho

# 

# In[1]:


var_menor_erro = None
valor_menor_erro = 1000.
# para cada variável, vamos treinar o modelo somente com esta variável e ver o que acontece.
for var in Xtr.columns:
    mdl = RandomForestRegressor(n_jobs=-1, random_state=0, n_estimators=500)
    mdl.fit(Xtr[[var]], ytr)
    p = mdl.predict(Xval[[var]])
    
    p_final = Xval['PRECO_MEDIO_REVENDA_ATUAL'] + p
    yval_final = Xval['PRECO_MEDIO_REVENDA_ATUAL'] + yval

    erro = np.sqrt(mean_squared_log_error(yval_final, p_final)) * 100
    
    print("Variável: {} - Erro: {:.4f}\n".format(var, erro))
    
    if erro < valor_menor_erro:
        var_menor_erro = var
        valor_menor_erro = erro
        
print("Melhor Variável: {} - Erro: {:.4f}\n".format(var_menor_erro, valor_menor_erro))
    


# In[ ]:




