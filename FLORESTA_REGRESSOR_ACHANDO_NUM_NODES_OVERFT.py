#!/usr/bin/env python
# coding: utf-8

# **[Introduction to Machine Learning Home Page](https://www.kaggle.com/learn/intro-to-machine-learning)**
# 
# ---
# 

# ## Recap
# Here's the code you've written so far.

# In[1]:


# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = 'dados/Iowa House Prices/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))


# # Exercises
# Data science isn't always this easy. But replacing the decision tree with a Random Forest is going to be an easy win.

# ## Step 1: Use a Random Forest

# In[2]:


from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X,train_y)
pred=rf_model.predict(val_X)


# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y,pred)




print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


# In[3]:


# Agora vou tentar fazer seleção de arvores aqui

#FUNCAO QUE CALCULA O ERRO
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    #model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model = RandomForestRegressor(random_state=1,n_estimators=max_leaf_nodes)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
#COMPARANDO
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


# In[4]:


print(best_tree_size)


# In[5]:


#RODANDO A MELHOR FLORESTA DE 25 ARVORES

model = RandomForestRegressor(random_state=1,n_estimators=best_tree_size)

# fit your model
rf_model.fit(train_X,train_y)
pred=rf_model.predict(val_X)


# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y,pred)




print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


# So far, you have followed specific instructions at each step of your project. This helped learn key ideas and build your first model, but now you know enough to try things on your own. 
# 
# Machine Learning competitions are a great way to try your own ideas and learn more as you independently navigate a machine learning project. 
# 
# # Keep Going
# 
# You are ready for **[Machine Learning Competitions](https://www.kaggle.com/kernels/fork/1259198).**
# 

# ---
# **[Introduction to Machine Learning Home Page](https://www.kaggle.com/learn/intro-to-machine-learning)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161285) to chat with other Learners.*

# In[ ]:




