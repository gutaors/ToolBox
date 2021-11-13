#!/usr/bin/env python
# coding: utf-8

# # IMPORTANDO ARQUIVOS CSV, TRATANDO NULL E DATA
# 

# In[2]:


import pandas as pd

# Import the data
nasdaq = pd.read_csv('../DataCamp/Importing-and-Managing-Financial-Data-in-Python-master/nasdaq-listings.csv', na_values='NAN', parse_dates=['Last Update'])

# Display the head of the data
display(nasdaq.head())

# Inspect the data
nasdaq.info()


# # importando planilha e colocando em uma lista
# 

# In[1]:


# Import the data
nyse = pd.read_excel('listings.xlsx', sheet_name='nyse', na_values='n/a')

# Display the head of the data
display(nyse.head())

# Inspect the data
nyse.info()


# # IMPORTANDO BOVESPA
# #### copiei de https://medium.com/@cesar.vieira/analisando-a%C3%A7%C3%B5es-da-bovespa-parte-i-500107703688

# In[2]:


import numpy as np
import pandas as pd
from pandas_datareader import data as wb
TickerA='ITSA4.SA'
TickerB='FLRY3.SA'
TickerC='LREN3.SA'
prices=pd.DataFrame()
tickers = [TickerA, TickerB, TickerC]
for t in tickers:
    prices[t]=wb.DataReader(t, data_source='yahoo', start='2010-1-1')['Adj Close']


# In[3]:


import matplotlib.pyplot as plt
(prices/prices.iloc[0]*100).plot(figsize=(15,5))
plt.ylabel('NORMALIZED PRICES')
plt.xlabel('DATE')
plt.show()


# In[4]:


log_returns=np.log(prices/prices.shift(1))
log_returns.plot(figsize=(15,5))
plt.ylabel('LOG DAILY RETURNS')
plt.xlabel('DATE')
plt.show()


# In[5]:


log_returns.mean()


# In[6]:


log_returns.std()


# ### Dicionário e uma dica de dicionário no airflow

# In[2]:


#declaro o dicionário da seguinte forma:
ZIP_FILES_DICT = {
        scdp_bilhetes_sql:SCDP_BILHETES,
        scdp_outliers_aereos_nacionais_sql:SCDP_OUTLIERS_NAC,
        scdp_outliers_aereos_internacionais_sql:SCDP_OUTLIERS_INTERNAC,
        scdp_classe_valor:SCDP_CLASSE_VALOR,
        scdp_meio_transporte_valor:SCDP_MEIO_TRANSPORTE,
        scdp_modalidade_valor:SCDP_MODALIDADE_VALOR,
        scdp_modalidade_valor_medio:SCDP_MODALIDADE_VALOR_MEDIO,
        scdp_outliers_aereos_nacionais_orig_dest_sql:SCDP_OUTLIERS_NAC_ORIG_DEST,
        scdp_outliers_aereos_internacionais_orig_dest_sql:SCDP_OUTLIERS_INTERNAC_ORIG_DEST
}


# estes itens à direita dos dois pontos são declarados lá em constants, da seguinte forma:
SCDP_BILHETES = "bilhetes.ods"

#daí na task eu chamo da seguinte forma:
## Tasks
t0 = PythonOperator(
    task_id="download_data",
    python_callable=_download_data,
    op_kwargs={"files_list": list(ZIP_FILES_DICT.values()), "files_dict": ZIP_FILES_DICT},
    provide_context=True,
    dag=dag
)


# ### Substituindo somente algumas letras por maiúsculas (duas formas de fazer)

# In[11]:


texto = 'Python'
nova_string = ''
for letra in texto:
    if letra == 't':
        nova_string = nova_string + letra.upper()
    elif letra == 'h':
        nova_string += letra.upper()
    else:
        nova_string += letra
print(nova_string)


# 

# 
