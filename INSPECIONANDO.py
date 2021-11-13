#!/usr/bin/env python
# coding: utf-8

# ## Comandos uteis para inspecionar

# In[ ]:


#colunas únicas de uma coluna
print(df['nomecol'].unique())
#tamanho do dataframe fundamentus
len(fundamentus.index)
#análise do fundamentus
fundamentus.describe()
#lista de colunas do dataframe
fundamentus.columns




#quantas linhas tem o dataframe fundamentus?
len(fundamentus.index)
#head
fundamentus.head(5)
#apaga linhas duplicadas
fundamentus=fundamentus.drop_duplicates(keep="first") 
#concatena dois dataframes em um terceiro
fundamentus = pd.concat([fundamentus,fundamentus_de_hoje])
#outer join
newdf = pd.merge(fundamentus, fundamentus2 ,on=['Papel','data'], how='outer')
#formata coluna data
newdf['data'] = pd.to_datetime(newdf['data'], format="%Y-%m-%d")


# In[ ]:


## Aula 1


# In[17]:


import pandas as pd

#fonte = "https://github.com/alura-cursos/imersao-dados-2-2020/blob/master/MICRODADOS_ENEM_2019_SAMPLE_43278.csv?raw=true"
#dados = pd.read_csv(fonte)
#dados.head()

dados = pd.read_csv('Dados/dados.csv')
dados.head()


# In[5]:


#dados.to_csv('dados.csv')


# In[8]:


dados.shape


# In[10]:




# Number of missing values in each column of training data
missing_val_count_by_column = (dados.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split




# X_full eh o train.csv 
# Se o dataset não tem o y (alvo => subset=['NU_INSCRICAO']), remove a linha dropna 
dados.dropna(axis=0, subset=['NU_INSCRICAO'], inplace=True)


# In[18]:


#joga coluna NU_INSCRICAO no y
y = dados.NU_INSCRICAO
#dropa SalePrice do dataset (ele já está no y)
#dados.drop(['NU_INSCRICAO'], axis=1, inplace=True)


# In[19]:



# "Cardinalidade" significa o numero de valores unicos em uma columna
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in dados.columns if
                    dados[cname].nunique() < 10 and 
                    dados[cname].dtype == "object"]


# In[20]:



# Select numerical columns
numerical_cols = [cname for cname in dados.columns if 
                dados[cname].dtype in ['int64', 'float64']]


# In[22]:



# Keep selected columns only
# aqui ele joga pra X as numericas e categoricas somente
my_cols = categorical_cols + numerical_cols
x = dados[my_cols].copy()
x = dados[my_cols].copy()
x = dados[my_cols].copy()


# In[ ]:



# "Cardinalidade" significa o numero de valores unicos em uma columna
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


# In[7]:


dados["SG_UF_RESIDENCIA"]


# In[8]:


dados.columns.values


# In[9]:


dados[["SG_UF_RESIDENCIA", "Q025"]]


# In[10]:


dados["SG_UF_RESIDENCIA"]


# In[11]:


dados["SG_UF_RESIDENCIA"].unique()


# In[12]:


len(dados["SG_UF_RESIDENCIA"].unique())


# In[13]:


dados["SG_UF_RESIDENCIA"].value_counts()


# In[14]:


dados["NU_IDADE"].value_counts()


# In[15]:


dados["NU_IDADE"].value_counts().sort_index()


# In[16]:


dados["NU_IDADE"].hist()


# In[17]:


dados["NU_IDADE"].hist(bins = 20, figsize = (10,8))


# In[ ]:





# In[18]:


dados.query("IN_TREINEIRO == 1")["NU_IDADE"].value_counts().sort_index()


# In[19]:


dados["NU_NOTA_REDACAO"].hist(bins = 20, figsize=(8, 6))


# In[20]:


dados["NU_NOTA_LC"].hist(bins = 20, figsize=(8, 6))


# In[21]:


dados["NU_NOTA_REDACAO"].mean()


# In[22]:


dados["NU_NOTA_REDACAO"].std()


# In[23]:


provas = ["NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_MT","NU_NOTA_LC","NU_NOTA_REDACAO"]

dados[provas].describe()


# In[24]:


dados["NU_NOTA_LC"].quantile(0.1)


# In[25]:


dados["NU_NOTA_LC"].plot.box(grid = True, figsize=(8,6))


# In[26]:


dados[provas].boxplot(grid=True, figsize= (10,8))


# Desafio01: Proporção dos inscritos por idade.
# 
# Desafio02: Descobrir de quais estados são os inscritos com 13 anos.
# 
# Desafio03: Adicionar título no gráfico
# 
# Desafio04: Plotar os Histogramas das idades dos do treineiro e não treineiros.
# 
# Desafio05: Comparar as distribuições das provas em inglês espanhol
# 
# Desafio06: Explorar a documentações e visualizações com matplotlib ou pandas e gerar novas visualizações.

# ## Aula 02

# In[27]:


dados.query("NU_IDADE == 13")


# In[28]:


dados.query("NU_IDADE <= 14")["SG_UF_RESIDENCIA"].value_counts()


# In[29]:


dados.query("NU_IDADE <= 14")["SG_UF_RESIDENCIA"].value_counts(normalize=True)


# In[30]:


alunos_menor_quartoze = dados.query("NU_IDADE <= 14")
alunos_menor_quartoze["SG_UF_RESIDENCIA"].value_counts().plot.pie(figsize=(10,8))


# In[31]:


alunos_menor_quartoze["SG_UF_RESIDENCIA"].value_counts(normalize = True).plot.bar(figsize=(10,8))


# In[32]:


len(alunos_menor_quartoze)


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.boxplot(x="Q006", y = "NU_NOTA_MT", data = dados)
plt.title("Boxplot das notas de matemática pela renda")


# In[34]:


renda_ordenada = dados["Q006"].unique()
renda_ordenada.sort()


# In[35]:


renda_ordenada


# In[36]:


plt.figure(figsize=(10, 6))
sns.boxplot(x="Q006", y = "NU_NOTA_MT", data = dados, order = renda_ordenada)
plt.title("Boxplot das notas de matemática pela renda")


# In[37]:


dados[provas].sum()


# In[38]:



dados["NU_NOTA_TOTAL"] = dados[provas].sum(axis=1)
dados.head()


# In[39]:


plt.figure(figsize=(10, 6))
sns.boxplot(x="Q006", y = "NU_NOTA_TOTAL", data = dados, order = renda_ordenada)
plt.title("Boxplot das notas de total pela renda")


# In[40]:


get_ipython().system('pip3 install seaborn==0.11.0')


# In[41]:


#sns.displot(dados, x ="NU_NOTA_TOTAL")


# In[42]:


provas = ["NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_MT","NU_NOTA_LC","NU_NOTA_REDACAO"]
provas.append("NU_NOTA_TOTAL")
dados[provas].query("NU_NOTA_TOTAL == 0")


# In[43]:


dados_sem_notas_zero = dados.query("NU_NOTA_TOTAL != 0")
dados_sem_notas_zero.head()


# In[44]:


plt.figure(figsize=(10, 6))
sns.boxplot(x="Q006", y = "NU_NOTA_TOTAL", data = dados_sem_notas_zero, order = renda_ordenada)
plt.title("Boxplot das notas de total pela renda")


# In[45]:


plt.figure(figsize=(14, 8))
sns.boxplot(x="Q006", y = "NU_NOTA_TOTAL", data = dados_sem_notas_zero, 
            hue = "IN_TREINEIRO", order = renda_ordenada)
plt.title("Boxplot das notas de total pela renda")


# MEGA DESAFIO DA THAINÁ: Pegar a amostra completa dos alunos de 13 e 14 anos
# 
# Desafio do Gui bonzinho: aumentar a amostra para alunos menor de idade e compara a proporção por estado.
# 
# Desafio 3: Criar uma função para plotar o boxplot do seaborn
# 
# Desafio 4: Verificar se quem zerou a prova foi eliminado ou não estava presente
# 
# Desafio 5: Quem é eliminado tira zero ou será NaN (não teve registro de notas)
# 
# DEsafio 6: Verificar a proporção dos participantes de rendas mais altas e mais baixas como treineiro e não treineiro.
# 
# Desafio 7: Fazer o mesmo boxplot olhando para a questão 25 (tem internet ou não) e fazer uma reflexão sobre o assunto e o contexto de pandemia.

# ## Aula 03

# In[46]:


plt.figure(figsize=(12,8))
sns.histplot(dados_sem_notas_zero, x = "NU_NOTA_TOTAL")


# In[106]:


plt.figure(figsize=(12,8))
sns.histplot(dados_sem_notas_zero, x = "NU_NOTA_MT")


# In[107]:


plt.figure(figsize=(12,8))
sns.histplot(dados_sem_notas_zero, x = "NU_NOTA_LC")


# In[108]:


plt.figure(figsize=(12,8))
sns.histplot(dados_sem_notas_zero, x = "NU_NOTA_TOTAL", hue="Q025", kde=True)


# In[109]:


plt.figure(figsize=(12,8))
sns.histplot(dados_sem_notas_zero, x = "NU_NOTA_TOTAL", hue="Q025", kde=True, stat="probability")


# In[110]:


plt.figure(figsize=(12,8))
sns.histplot(dados_sem_notas_zero, x = "NU_NOTA_TOTAL", hue="Q025", kde=True, stat="density", cumulative=True)


# In[111]:


plt.figure(figsize=(10, 10))
sns.scatterplot(data = dados_sem_notas_zero, x="NU_NOTA_MT", y="NU_NOTA_LC", hue="Q025")
plt.xlim((-50, 1050))
plt.ylim((-50, 1050))


# In[112]:


provas


# In[113]:


sns.pairplot(dados_sem_notas_zero[provas])


# In[ ]:


correlacao = dados_sem_notas_zero[provas].corr()
correlacao


# In[ ]:


sns.heatmap(correlacao, cmap="Blues", center=0, annot=True)


# Desafio: Plotar as médias, medianas e moda nas notas de LC e MT (matiplotlib linha vertical)
# 
# Desafio2: Melhorar a visualização da matriz de correlação e analisar mais detalhadamente.
# 
# Desafio3: Filtrar as notas por seu estado ou sua cidade e refazer as análises, verificando se são semelhantes ao geral.
# 
# Desafio4: Pensar sobre a correlação entre matemática e Linguagens.

# ## Aula 04

# In[ ]:


provas


# In[ ]:


provas_entrada = ["NU_NOTA_CH","NU_NOTA_LC", "NU_NOTA_CN","NU_NOTA_REDACAO"]
prova_saida = "NU_NOTA_MT"
dados_sem_notas_zero = dados_sem_notas_zero[provas].dropna()
notas_entrada = dados_sem_notas_zero[provas_entrada]
notas_saida = dados_sem_notas_zero[prova_saida]


# In[ ]:


notas_entrada


# In[ ]:


x = notas_entrada
y = notas_saida 


# In[ ]:


from sklearn.model_selection import train_test_split

SEED = 4321

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.25,
                                                        random_state=SEED)


# In[ ]:


from sklearn.svm import LinearSVR

modelo = LinearSVR(random_state = SEED)
modelo.fit(x_treino, y_treino)


# In[ ]:


predicoes_matematica = modelo.predict(x_teste)


# In[ ]:


y_teste[:5]


# In[ ]:


plt.figure(figsize=(8, 8))
sns.scatterplot(x=predicoes_matematica, y=y_teste)
plt.xlim((-50, 1050))
plt.ylim((-50, 1050))


# In[ ]:


plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_teste, y=y_teste - predicoes_matematica)


# In[ ]:


plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_teste, y=x_teste.mean(axis=1))
plt.xlim((-50, 1050))
plt.ylim((-50, 1050))


# In[ ]:


resultados = pd.DataFrame()
resultados["Real"] = y_teste
resultados["Previsao"] = predicoes_matematica
resultados["diferenca"] = resultados["Real"] - resultados["Previsao"]
resultados["quadrado_diferenca"] = (resultados["Real"] - resultados["Previsao"])**2


# In[ ]:


resultados


# In[ ]:


resultados["quadrado_diferenca"].mean()


# In[ ]:


resultados["quadrado_diferenca"].mean()**(1/2)


# In[ ]:


from sklearn.dummy import DummyRegressor

modelo_dummy = DummyRegressor()
modelo_dummy.fit(x_treino, y_treino)
dummy_predicoes = modelo_dummy.predict(x_teste)


# In[ ]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_teste, dummy_predicoes)


# In[ ]:


mean_squared_error(y_teste, predicoes_matematica)


# Desafio01: Procurar outro modelo de ML para treinar e comparar com os modelos criados em aula
# 
# Desafio02: Ler a documentação do Dummy e alterar o método de regressão
# 
# Desafio03: Buscar outra métrica para avaliar modelos de regressão
# 

# ## Aula 05

# In[ ]:


from sklearn.svm import LinearSVR

modelo = LinearSVR(random_state=SEED)
modelo.fit(x_treino, y_treino)
predicoes_matematica = modelo.predict(x_teste)
mean_squared_error(y_teste, predicoes_matematica)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25)
modelo_arvore = DecisionTreeRegressor(max_depth = 3)
modelo_arvore.fit(x_treino, y_treino)
predicoes_matematica_arvore = modelo_arvore.predict(x_teste)
mean_squared_error(y_teste, predicoes_matematica_arvore)


# In[ ]:


from sklearn.model_selection import cross_validate

modelo_arvore = DecisionTreeRegressor(max_depth=2)
resultados = cross_validate(modelo_arvore, x, y, cv= 10, scoring="neg_mean_squared_error")
media = (resultados["test_score"]*-1).mean()


# ### Aqui embaixo ele mostra modelos de 5 kfolds, por isto ele fala o seguinte, se fizer media dos scores, pode ser enganoso. Em uma sala com notas zero e dez a media será 5 e ninguém tirou 5. Por isto ele calcula o intervalo de confiança.<br>
# Intervalo de confiança é o seguinte, se pegar o procedimento que gerou o intervalo e rodar este procedimento muitas e muitas vezes, 95% deles vai gerar um valor dentro deste intervalo. Então se rodar 100 vezes 95 delas vai dar resultado dentro deste intervalo. as outras 5 vao ficar fora.
# 

# In[ ]:


from sklearn.model_selection import cross_validate

modelo_arvore = DecisionTreeRegressor(max_depth=2)
resultados = cross_validate(modelo_arvore, x, y, cv= 10, scoring="neg_mean_squared_error")
media = (resultados["test_score"]*-1).mean()
desvio_padrao = (resultados["test_score"]*-1).std()
lim_inferior = media - (2*desvio_padrao)
lim_superior = media + (2*desvio_padrao)

print(f"Intervalo de confiança {lim_inferior} - {lim_superior}")


# In[ ]:


resultados["test_score"]*-1


# In[ ]:


def calcula_mse(resultados):
    media = (resultados["test_score"]*-1).mean()
    desvio_padrao = (resultados["test_score"]*-1).std()
    lim_inferior = media - (2*desvio_padrao)
    lim_superior = media + (2*desvio_padrao)
    print(f"Intervalo de confiança {lim_inferior} - {lim_superior}")


# In[ ]:


from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import numpy as np

SEED= 1232
np.random.seed(SEED)



partes = KFold(n_splits = 10, shuffle=True)
modelo_arvore = DecisionTreeRegressor(max_depth=3)
resultados = cross_validate(modelo_arvore, x, y, cv= partes, scoring="neg_mean_squared_error")
calcula_mse(resultados)


# In[ ]:


def regressor_arvore(nivel):
    SEED= 1232
    np.random.seed(SEED)
    partes = KFold(n_splits = 10, shuffle=True)
    modelo_arvore = DecisionTreeRegressor(max_depth=nivel)
    resultados = cross_validate(modelo_arvore, x, y, cv= partes, scoring="neg_mean_squared_error", return_train_score=True)
    print(f"Treino = {(resultados['train_score']*-1).mean()}|Teste = {(resultados['test_score']*-1).mean()}")

regressor_arvore(4)


# In[ ]:


for i in range(1,21):
    regressor_arvore(i)


# Desafio 01: Pesquisar sobre intervalo de confiança (Quem quiser discutir no Discord, estaremos lá)
# 
# Desafio 02: Testar com outros parâmetros da árvore de decisão
# 
# Desafio 03: Procurar outras formas de realizar os ajustes de parâmetros com o Sklearn
# 
# Desafio 04: Pesquisar o que é o problema de underfit.
# 
# Desafio 05: Plotar um gráfico com test_score e train_test.
# 
# 

# In[ ]:




