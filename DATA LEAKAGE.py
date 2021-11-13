#!/usr/bin/env python
# coding: utf-8

# **[Intermediate Machine Learning Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# ---
# 

# Most people find target leakage very tricky until they've thought about it for a long time.
# 
# So, before trying to think about leakage in the housing price example, we'll go through a few examples in other applications. Things will feel more familiar once you come back to a question about house prices.
# 
# # Setup
# 
# The questions below will give you feedback on your answers. Run the following cell to set up the feedback system.

# In[1]:


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex7 import *
print("Setup Complete")


# # 1. The Data Science of Shoelaces
# 
# Nike has hired you as a data science consultant to help them save money on shoe materials. Your first assignment is to review a model one of their employees built to predict how many shoelaces they'll need each month. The features going into the machine learning model include:
# - The current month (January, February, etc)
# - Advertising expenditures in the previous month
# - Various macroeconomic features (like the unemployment rate) as of the beginning of the current month
# - The amount of leather they ended up using in the current month
# 
# The results show the model is almost perfectly accurate if you include the feature about how much leather they used. But it is only moderately accurate if you leave that feature out. You realize this is because the amount of leather they use is a perfect indicator of how many shoes they produce, which in turn tells you how many shoelaces they need.
# 
# Do you think the _leather used_ feature constitutes a source of data leakage? If your answer is "it depends," what does it depend on?
# 
# After you have thought about your answer, check it against the solution below.
# 
# A Nike contratou você como consultor de ciência de dados para ajudá-los a economizar dinheiro em materiais para calçados. Sua primeira tarefa é revisar um modelo que um de seus funcionários construiu para prever quantos cadarços eles vão precisar por mês. Os recursos que vão para o modelo de aprendizado de máquina incluem:
# - O mês atual (janeiro, fevereiro, etc)
# - Despesas com publicidade no mês anterior
# - Vários aspectos macroeconômicos (como a taxa de desemprego) desde o início do mês atual
# - A quantidade de couro que acabaram usando no mês atual
# 
# Os resultados mostram que o modelo é quase perfeitamente preciso se você incluir o recurso sobre a quantidade de couro usado. Mas é apenas moderadamente preciso se você deixar esse recurso de fora. Você percebe que isso ocorre porque a quantidade de couro que eles usam é um indicador perfeito de quantos sapatos eles produzem, o que, por sua vez, indica de quantos cadarços eles precisam.
# 
# Você acha que o recurso _couro usado_ constitui uma fonte de vazamento de dados? Se sua resposta for "depende", do que depende?
# 
# Depois de pensar em sua resposta, compare-a com a solução abaixo.
# 
# Solução: Isso é complicado e depende dos detalhes de como os dados são coletados (o que é comum quando se pensa em vazamentos). Você decidiria no início do mês quanto couro será usado naquele mês? Se sim, está tudo bem. Mas se isso for determinado durante o mês, você não terá acesso a ele quando fizer a previsão. Se você tiver um palpite no início do mês e ele for alterado posteriormente durante o mês, o valor real usado durante o mês não pode ser usado como um recurso (porque causa vazamento).
# 

# In[2]:


# Check your answer (Run this code cell to receive credit!)
q_1.solution()


# # 2. Return of the Shoelaces
# 
# You have a new idea. You could use the amount of leather Nike ordered (rather than the amount they actually used) leading up to a given month as a predictor in your shoelace model.
# 
# Does this change your answer about whether there is a leakage problem? If you answer "it depends," what does it depend on?
# 
# # 2. Retorno dos cadarços
# Você tem uma nova ideia. Você pode usar a quantidade de couro pedido pela Nike (em vez da quantidade que eles realmente usaram) até um determinado mês como um indicador em seu modelo de cadarço.
# 
# Isso muda sua resposta sobre se há um problema de vazamento? Se você responder "depende", de que depende?
# 
# Solução: Isso pode ser bom, mas depende se eles pedem o cadarço primeiro ou o couro primeiro. Se eles pedirem os cadarços primeiro, você não saberá quanto couro eles encomendaram quando prever as necessidades de cadarços. Se eles fizerem o pedido de couro primeiro, você terá esse número disponível quando fizer o pedido do cadarço e não terá problemas.

# In[3]:


# Check your answer (Run this code cell to receive credit!)
q_2.solution()


# # 3. Getting Rich With Cryptocurrencies?
# 
# You saved Nike so much money that they gave you a bonus. Congratulations.
# 
# Your friend, who is also a data scientist, says he has built a model that will let you turn your bonus into millions of dollars. Specifically, his model predicts the price of a new cryptocurrency (like Bitcoin, but a newer one) one day ahead of the moment of prediction. His plan is to purchase the cryptocurrency whenever the model says the price of the currency (in dollars) is about to go up.
# 
# The most important features in his model are:
# - Current price of the currency
# - Amount of the currency sold in the last 24 hours
# - Change in the currency price in the last 24 hours
# - Change in the currency price in the last 1 hour
# - Number of new tweets in the last 24 hours that mention the currency
# 
# The value of the cryptocurrency in dollars has fluctuated up and down by over \$100 in the last year, and yet his model's average error is less than \$1. He says this is proof his model is accurate, and you should invest with him, buying the currency whenever the model says it is about to go up.
# 
# Is he right? If there is a problem with his model, what is it?
# 
# 3. Ficando rico com criptomoedas?
# Você economizou tanto dinheiro para a Nike que ela lhe deu um bônus. Parabéns.
# 
# Seu amigo, que também é cientista de dados, diz que construiu um modelo que permitirá que você transforme seu bônus em milhões de dólares. Especificamente, seu modelo prevê o preço de uma nova criptomoeda (como Bitcoin, mas uma mais recente) um dia antes do momento da previsão. Seu plano é comprar a criptomoeda sempre que o modelo disser que o preço da moeda (em dólares) está prestes a subir.
# 
# Os recursos mais importantes em seu modelo são:
# 
# Preço atual da moeda
# Quantidade de moeda vendida nas últimas 24 horas
# Mudança no preço da moeda nas últimas 24 horas
# Mudança no preço da moeda na última 1 hora
# Número de novos tweets nas últimas 24 horas que mencionam a moeda
# O valor da criptomoeda em dólares oscilou para cima e para baixo em mais de 100𝑖𝑛𝑡ℎ𝑒𝑙𝑎𝑠𝑡𝑦𝑒𝑎𝑟, 𝑎𝑛𝑑𝑦𝑒𝑡ℎ𝑖𝑠𝑚𝑜𝑑𝑒𝑙′𝑠𝑎𝑣𝑒𝑟𝑎𝑔𝑒𝑒𝑟𝑟𝑜𝑟𝑖𝑠𝑙𝑒𝑠𝑠𝑡ℎ𝑎𝑛 1. Ele diz que isso prova que seu modelo é preciso, e você deve investir com ele, comprando a moeda sempre que o modelo disser que está prestes a sair acima.
# 
# Ele esta certo? Se houver algum problema com o modelo dele, qual é?
# 
# Solução: Não há fonte de vazamento aqui. Esses recursos devem estar disponíveis no momento em que você deseja fazer uma predição e é improvável que sejam alterados nos dados de treinamento após a determinação do alvo da predição. Mas, a maneira como ele descreve a precisão pode ser enganosa se você não for cuidadoso. Se o preço se mover gradualmente, o preço de hoje será um preditor preciso do preço de amanhã, mas pode não dizer se é uma boa hora para investir. Por exemplo, se for 100𝑡𝑜𝑑𝑎𝑦, 𝑎𝑚𝑜𝑑𝑒𝑙𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑛𝑔𝑎𝑝𝑟𝑖𝑐𝑒𝑜𝑓 100 amanhã pode parecer preciso, mesmo que não possa dizer se o preço está subindo ou descendo em relação ao preço atual. Uma meta de previsão melhor seria a mudança no preço no dia seguinte. Se você puder prever consistentemente se o preço está prestes a subir ou descer (e em quanto), pode ter uma oportunidade de investimento vencedora.

# In[4]:


# Check your answer (Run this code cell to receive credit!)
q_3.solution()


# # 4. Preventing Infections
# 
# An agency that provides healthcare wants to predict which patients from a rare surgery are at risk of infection, so it can alert the nurses to be especially careful when following up with those patients.
# 
# You want to build a model. Each row in the modeling dataset will be a single patient who received the surgery, and the prediction target will be whether they got an infection.
# 
# Some surgeons may do the procedure in a manner that raises or lowers the risk of infection. But how can you best incorporate the surgeon information into the model?
# 
# You have a clever idea. 
# 1. Take all surgeries by each surgeon and calculate the infection rate among those surgeons.
# 2. For each patient in the data, find out who the surgeon was and plug in that surgeon's average infection rate as a feature.
# 
# Does this pose any target leakage issues?
# Does it pose any train-test contamination issues?
# 
# 
# 4. Prevenção de infecções
# Uma agência que fornece serviços de saúde deseja prever quais pacientes de uma cirurgia rara estão em risco de infecção, para que possa alertar as enfermeiras para serem especialmente cuidadosas ao acompanhar esses pacientes.
# 
# Você quer construir um modelo. Cada linha no conjunto de dados de modelagem será um único paciente que recebeu a cirurgia, e o alvo de previsão será se eles contraíram uma infecção.
# 
# Alguns cirurgiões podem realizar o procedimento de maneira a aumentar ou diminuir o risco de infecção. Mas como você pode incorporar melhor as informações do cirurgião ao modelo?
# 
# Você tem uma ideia inteligente.
# 
# Pegue todas as cirurgias de cada cirurgião e calcule a taxa de infecção entre esses cirurgiões.
# Para cada paciente nos dados, descubra quem era o cirurgião e inclua a taxa média de infecção desse cirurgião como um recurso.
# Isso representa algum problema de vazamento de destino? Isso representa algum problema de contaminação de teste de trem?
# 
# Solução: Isso representa um risco de vazamento no alvo e contaminação do teste do trem (embora você possa evitar ambos se for cuidadoso).
# 
# Você tem o vazamento alvo se o resultado de um determinado paciente contribuir para a taxa de infecção de seu cirurgião, que é então conectado de volta ao modelo de previsão para verificar se o paciente foi infectado. Você pode evitar o vazamento alvo se calcular a taxa de infecção do cirurgião usando apenas as cirurgias anteriores ao paciente que estamos prevendo. Calcular isso para cada cirurgia em seus dados de treinamento pode ser um pouco complicado.
# 
# Você também terá um problema de contaminação de teste de trem se calcular isso usando todas as cirurgias que um cirurgião realizou, incluindo aquelas do conjunto de teste. O resultado seria que seu modelo poderia parecer muito preciso no conjunto de teste, mesmo que não generalizasse bem para novos pacientes após o modelo ser implantado. Isso aconteceria porque o recurso de risco do cirurgião considera os dados no conjunto de teste. Existem conjuntos de teste para estimar como o modelo se sairá ao ver novos dados. Portanto, essa contaminação anula o propósito do conjunto de teste.

# In[5]:


# Check your answer (Run this code cell to receive credit!)
q_4.solution()


# # 5. Housing Prices
# 
# You will build a model to predict housing prices.  The model will be deployed on an ongoing basis, to predict the price of a new house when a description is added to a website.  Here are four features that could be used as predictors.
# 1. Size of the house (in square meters)
# 2. Average sales price of homes in the same neighborhood
# 3. Latitude and longitude of the house
# 4. Whether the house has a basement
# 
# You have historic data to train and validate the model.
# 
# Which of the features is most likely to be a source of leakage?
# 
# 
# 5. Preços de habitação
# Você construirá um modelo para prever os preços dos imóveis. O modelo será implantado continuamente, para prever o preço de uma nova casa quando uma descrição for adicionada a um site. Aqui estão quatro recursos que podem ser usados ​​como preditores.
# 
# Tamanho da casa (em metros quadrados)
# Preço médio de venda de casas no mesmo bairro
# Latitude e longitude da casa
# Se a casa tem um porão
# Você tem dados históricos para treinar e validar o modelo.
# 
# Qual dos recursos tem mais probabilidade de ser uma fonte de vazamento?
# 
# 
# Solução: 2 é a fonte do vazamento alvo. Aqui está uma análise para cada recurso:
# 
# É improvável que o tamanho de uma casa seja alterado após a venda (embora tecnicamente seja possível). Mas, normalmente, isso estará disponível quando precisarmos fazer uma previsão e os dados não serão modificados após a venda da casa. Portanto, é muito seguro.
# 
# Não sabemos as regras para quando isso é atualizado. Se o campo for atualizado nos dados brutos depois que uma casa foi vendida e a venda da casa for usada para calcular a média, isso constituirá um caso de vazamento alvo. No extremo, se apenas uma casa for vendida no bairro, e for a casa que estamos tentando prever, a média será exatamente igual ao valor que estamos tentando prever. Em geral, para bairros com poucas vendas, o modelo terá um desempenho muito bom nos dados de treinamento. Mas quando você aplica o modelo, a casa que você está prevendo ainda não foi vendida, então esse recurso não funcionará da mesma forma que funcionou nos dados de treinamento.
# 
# Eles não mudam e estarão disponíveis no momento em que quisermos fazer uma previsão. Portanto, não há risco de vazamento do alvo aqui.
# 
# Isso também não muda e está disponível no momento em que queremos fazer uma previsão. Portanto, não há risco de vazamento do alvo aqui.

# In[2]:


# Fill in the line below with one of 1, 2, 3 or 4.
potential_leakage_feature = 2

# Check your answer
#q_5.check()


# In[6]:


#q_5.hint()
q_5.solution()


# # Conclusion
# Leakage is a hard and subtle issue. You should be proud if you picked up on the issues in these examples.
# 
# Now you have the tools to make highly accurate models, and pick up on the most difficult practical problems that arise with applying these models to solve real problems.
# 
# There is still a lot of room to build knowledge and experience. Try out a [Machine Learning Competition](https://www.kaggle.com/competitions) or look through our [Datasets](https://kaggle.com/datasets) to practice your new skills.
# 
# Again, Congratulations!

# ---
# **[Intermediate Machine Learning Home Page](https://www.kaggle.com/learn/intermediate-machine-learning)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161289) to chat with other Learners.*
