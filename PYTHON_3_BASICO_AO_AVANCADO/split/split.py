#https://blog.betrybe.com/python/python-split/


#Qual é a sintaxe?
nome = "João Paulo da Silva"
print(nome.split())


#Separator
nomes = "João Paulo/Maria Paula/Ana Beatriz/José Pedro"
print(nomes.split('/'))
#resultado: ['João Paulo', 'Maria Paula', 'Ana Beatriz', 'José Pedro']


#Maxsplit
nome = "João Paulo Menezes da Silva"
print(nome.split(None, 2))
#resultado: ['João', 'Paulo', 'Menezes da Silva']

print(nome.split(" ", 2))
#resultado: ['João', 'Paulo', 'Menezes da Silva']

print(nome.split(2))
'''
resultado: 
    print(nome.split(2))
TypeError: must be str or None, not int
'''


#Separar strings com ocorrências de um caractere específico

numeros = "123x124x125x126"
print(numeros.split("x"))
#resultado: ['123', '124', '125', '126']


#Separar strings que contenham o caractere de nova linha ou tabulação

nomes = "Maria Cecília\nCláudia Rodrigues\nJoão Paulo"
print(nomes)
'''
Resultado da exibição do conteúdo original:
Maria Cecília
Cláudia Rodrigues
João Paulo
'''

print(nomes.split("\n"))
#Resultado após o split(): ['Maria Cecília, 'Cláudia Rodrigues', 'João Paulo']


#Dividir a string em tamanhos definidos

linguagens = "Linguagens de programação;Python;C;JavaScript;Ruby"
print(linguagens.split(';',0))
#resultado: ['Linguagens de programação;Python;C;JavaScript;Ruby']

print(linguagens.split(';',1))
#resultado: ['Linguagens de programação', 'Python;C;JavaScript;Ruby']

print(linguagens.split(';',2))
#resultado: ['Linguagens de programação', 'Python', 'C;JavaScript;Ruby']

print(linguagens.split(';',3))
#resultado: ['Linguagens de programação', 'Python', 'C', 'JavaScript;Ruby']

print(linguagens.split(';',4))
#resultado: ['Linguagens de programação', 'Python', 'C', 'JavaScript', 'Ruby']

print(linguagens.split(';',9))
#resultado: ['Linguagens de programação', 'Python', 'C', 'JavaScript', 'Ruby']


#Atribuir os valores da lista a variáveis diferentes

dados = "Maria Cláudia;23;Desenvolvedora Web"
nome,idade,profissao = dados.split(";")
print(nome)
#resultado: Maria Cláudia

print(idade)
#resultado: 23

print(profissao)
#resultado: Desenvolvedora Web

print(type(nome))
#resultado: <class 'str'>
print(type(idade))
#resultado: <class 'str'>
print(type(profissao))
#resultado: <class 'str'>

dados = "Maria Cláudia;23;Desenvolvedora Web"
nome,idade = dados.split(";")
print(nome)
print(idade)
'''
resultado:
  nome,idade = dados.split(";")
ValueError: too many values to unpack (expected 2)
'''


#Separar dados de um arquivo externo

arquivo = open("texto.txt", "r")
conteudo = arquivo.read()
print(conteudo)
'''
Resultado:
Primeira linha.
Segunda linha.
Terceira linha.
'''

print(conteudo.split("\n"))
'''
Resultado:
['Primeira linha.', 'Segunda linha.', 'Terceira linha.']
'''
arquivo.close()

arquivo_csv = open("numeros.csv", "r")
dados = arquivo_csv.read()
print(dados)
'''
exibe o conteúdo original do arquivo numeros.csv
resultado: 
10;20;30;40;50;60;80;90;100
'''

print(dados.split(";"))
'''
resultado:
['10', '20', '30', '40', '50', '60', '80', '90', '100']
'''
arquivo_csv.close()