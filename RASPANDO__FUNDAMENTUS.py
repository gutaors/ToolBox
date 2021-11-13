#!/usr/bin/env python
# coding: utf-8

# #Ricos pelo Acaso

# Link para o vídeo: https://youtu.be/e_ZRDG4F4ZA
# 
# também tem um acesso ao fundamentus no MINHA VERSAO 15

# # Importando Bibliotecas

# In[1]:


import numpy as np
import pandas as pd
import string
import warnings
import requests
warnings.filterwarnings('ignore')


# # Obtendo e tratando os dados

# In[2]:


#url = 'https://www.fundamentus.com.br/resultado.php'
#df = pd.read_html(url, decimal=',', thousands='.')[0]
#A url que você quer acesssar
#url = "https://www.portalbrasil.net/igpm.htm"
url = 'https://www.fundamentus.com.br/resultado.php'

#Informações para fingir ser um navegador
header = {
  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
  "X-Requested-With": "XMLHttpRequest"
}
#juntamos tudo com a requests
r = requests.get(url, headers=header)
#E finalmente usamos a função read_html do pandas
lista = pd.read_html(r.text)
print(lista)
df = pd.concat(lista)
type(df)
df.head()


# In[3]:


df.head(5)


# In[4]:


#type(lista)
#print(lista)


# In[ ]:





# In[5]:


for coluna in ['Div.Yield', 'Mrg Ebit', 'Mrg. Líq.', 'ROIC', 'ROE', 'Cresc. Rec.5a','EV/EBIT']:
  df[coluna] = df[coluna].str.replace('.', '')
  df[coluna] = df[coluna].str.replace(',', '.')
  df[coluna] = df[coluna].str.rstrip('%').astype('float') / 100


# # Analisando os dados

# In[6]:


#engenhosidade, estava dando erro entao eu fiz a conversao logo abaixo
#df['Liq.2meses'] = pd.to_numeric(df['Liq.2meses'], errors='coerce')

#df = df[df['Liq.2meses'] > 1000000]
df.head()


# In[7]:


# Vamos tentar converter pra não dar erro nos campos da proxima célula
#for coluna in [ 'EV/EBIT',  'ROIC']:
#  df[coluna] = df[coluna].str.replace('.', '')
#  df[coluna] = df[coluna].astype('float')


# In[8]:



ranking = pd.DataFrame()
ranking['pos'] = range(1,151)
ranking['EV/EBIT'] = df[ df['EV/EBIT'] > 0 ].sort_values(by=['EV/EBIT'])['Papel'][:150].values
ranking['ROIC'] = df.sort_values(by=['ROIC'], ascending=False)['Papel'][:150].values


# In[9]:


ranking


# In[10]:


a = ranking.pivot_table(columns='EV/EBIT', values='pos')


# In[11]:


b = ranking.pivot_table(columns='ROIC', values='pos')


# In[12]:


t = pd.concat([a,b])
t


# In[13]:


rank = t.dropna(axis=1).sum()
rank


# In[14]:


rank.sort_values()[:15]


# In[ ]:




