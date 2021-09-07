#!/usr/bin/env python
# coding: utf-8

# **[Intermediate Machine Learning Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# ---
# 

# # **pipelines** servem para aumentar a eficiencia do código de machine learning.
# 
# # Setup
# 
# # Introducão
# ## Pipelines são forma de organizar o código para prepocessar e modelar dados 
# Ele empacota os passos de preprocessamento e modelagem, então vc pode usar o pacote inteiro como se fosse um passo só
# 
# ## Benefícios do pipeline
# 
# codigo limpo, afinal codigo de preprocessamento pode ficar bagunçado
# nao precisara rastrear manualmente seus dados de validaçao e treinamento a cada passo
# 
# ## Poucos Bugs: pouca chance de aplicar mal um passo ou esquecer um passo de preprocessamento
# 
# Facilidade de produtizar: facilita transicionar para algo deployable, ao contrario de modelos. Ou seja, pipeline é fácil de botar em outra escala.
# 
# ## Vamos lidar abaixo com categoricas ou faltantes
# 
# 
# 

# In[21]:


# Set up code checking
import os
if not os.path.exists("Dados/Housing Prices Competition for Kaggle Learn Users/train.csv"):
    print("arquivo nao existe")
    os.symlink("/Dados/Housing Prices Competition for Kaggle Learn Users/train.csv", "../input/train.csv")  
    os.symlink("/Dados/Housing Prices Competition for Kaggle Learn Users/test.csv", "../input/test.csv") 


# You will work with data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course). 
# 
# ![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)
# 
# Run the next code cell without changes to load the training and validation sets in `X_train`, `X_valid`, `y_train`, and `y_valid`.  The test set is loaded in `X_test`.

# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Importa os dois CSVs, teste e treino
X_full = pd.read_csv('Dados/Housing Prices Competition for Kaggle Learn Users/train.csv', index_col='Id')
X_test_full = pd.read_csv('Dados/Housing Prices Competition for Kaggle Learn Users/test.csv', index_col='Id')


# In[23]:


# contagem de isnull
print(X_full.SalePrice.isnull().sum())


# In[24]:


# contagem de isna
print(X_full.SalePrice.isna().sum())


# In[25]:


#contagem de nao nulos
print(X_full.SalePrice.count())


# In[4]:


# Remove linhas com a variável alvo faltando, separa alvo e preditores
# neste caso nao tinha salesprice nulo então não dá pra ver a função agindo mas se desse os contadores acima ajudariam
# dissecando o comando df.dropna(axis=0, subset=['ColunaAlvo'], inplace=True)
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Quebrando validação do treinamento
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# In[ ]:





# In[10]:


X_train.head()


# The next code cell uses code from the tutorial to preprocess the data and train a model.  Run this code without changes.

# In[11]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))


# The code yields a value around 17862 for the mean absolute error (MAE).  In the next step, you will amend the code to do better.
# 
# # Step 1: Improve the performance
# 
# ### Part A
# 
# Now, it's your turn!  In the code cell below, define your own preprocessing steps and random forest model.  Fill in values for the following variables:
# - `numerical_transformer`
# - `categorical_transformer`
# - `model`
# 
# To pass this part of the exercise, you need only define valid preprocessing steps and a random forest model.

# In[13]:


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)
# Your code here


# In[14]:


# Lines below will give you a hint or solution code
#step_1.a.hint()
#step_1.a.solution()


# ### Part B
# 
# Run the code cell below without changes.
# 
# To pass this step, you need to have defined a pipeline in **Part A** that achieves lower MAE than the code above.  You're encouraged to take your time here and try out many different approaches, to see how low you can get the MAE!  (_If your code does not pass, please amend the preprocessing steps and model in Part A._)

# In[16]:


# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)


# In[17]:


# Line below will give you a hint
#step_1.b.hint()


# # Step 2: Generate test predictions
# 
# Now, you'll use your trained model to generate predictions with the test data.

# In[19]:


# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test)


# In[20]:


# Lines below will give you a hint or solution code
#step_2.hint()
#step_2.solution()


# Run the next code cell without changes to save your results to a CSV file that can be submitted directly to the competition.

# In[21]:


# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)


# # Step 3: Submit your results
# 
# Once you have successfully completed Step 2, you're ready to submit your results to the leaderboard!  If you choose to do so, make sure that you have already joined the competition by clicking on the **Join Competition** button at [this link](https://www.kaggle.com/c/home-data-for-ml-course).  
# 1. Begin by clicking on the blue **Save Version** button in the top right corner of the window.  This will generate a pop-up window.  
# 2. Ensure that the **Save and Run All** option is selected, and then click on the blue **Save** button.
# 3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 4. Click on the **Output** tab on the right of the screen.  Then, click on the blue **Submit** button to submit your results to the leaderboard.
# 
# You have now successfully submitted to the competition!
# 
# If you want to keep working to improve your performance, select the blue **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.
# 
# 
# # Keep going
# 
# Move on to learn about [**cross-validation**](https://www.kaggle.com/alexisbcook/cross-validation), a technique you can use to obtain more accurate estimates of model performance!

# ---
# **[Intermediate Machine Learning Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161289) to chat with other Learners.*
