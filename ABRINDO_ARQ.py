#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn 

## Introdução ao Preprocessamento
# # Importando csv gerados pelo quartzo

# In[1]:


import os
os.chdir("D:/DEV/MINISTERIO/LANCES/tipo1")
#df  =  pd.read_csv('tipo1.csv', parse_dates=['lanData'])
#df  =  pd.read_csv('tipo1.csv', parse_dates= ['lanData'],encoding='utf-8-sig', usecols= ['lanData', 'lanCod'],)
df  =  pd.read_csv('tipo1.csv', parse_dates= ['lanData'],encoding='utf-8-sig')


# # Alguns erros abrindo arquivos
# 

# In[2]:



#erro
pandas UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed in position 7: invalid continuation byte

#solucao
df = pd.read_csv('file_name.csv', engine='python')


# In[3]:


#erro
SyntaxError: can't assign to operator

#solucao
o pd.read_csv atribuia a uma variavel com hifen no nome, python nao aceita hifen no nome da variável


# # Colando CSVs de um diretorio em um só csv
# ## arquivos com mesma estrutura que vao ser concatenados em um só

# In[ ]:


import os
import glob
import pandas as pd

## CRIANDO ARQUIVO TIPO 1
os.chdir("D:/DEV/MINISTERIO/LANCES/tipo1")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "tipo1.csv", index=False, encoding='utf-8-sig')

#em seguida abre o csv
os.chdir("D:/DEV/MINISTERIO/LANCES/tipo1")
df  =  pd.read_csv('tipo1.csv', parse_dates=['lanData'])

## CRIANDO ARQUIVO TIPO 3
#os.chdir("D:/DEV/MINISTERIO/LANCES/tipo3")
#extension = 'csv'
#all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
##combine all files in the list
#combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
##export to csv
#combined_csv.to_csv( "tipo3.csv", index=False, encoding='utf-8-sig')


# # Abrindo arquivo csv com codificação desconhecida (erro utf8)

# In[ ]:


import chardet
with open('Dados/LANCES_FORN_PREGAO_ABERTO.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

lances_fornecedor_pregao_aberto  =  pd.read_csv('Dados/LANCES_FORN_PREGAO_ABERTO.csv', encoding=result['encoding'])


# # Descobrindo a codificacao

# In[ ]:


# Vou inspecionar para tentar achar a codificacao
file = 'DADOS_BRUTOS/consolidado.csv'
import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


# # Abrindo arquivo csv com separador tab (sem isto ele junta tudo sem separar as colunas e coloca \t no meio do texto)
# 

# In[9]:


dados = pd.read_csv("../MARIO FILHO/seriestemporaisyt-master/2004-2019.tsv", sep = '\t')


# In[10]:


dados.head(4)


# # Para exibir todas as colunas do dataframe
# 

# In[1]:


# Para exibir todas as colunas do dataframe
pd.pandas.set_option('display.max_columns', None)


# In[ ]:




