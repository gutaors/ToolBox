# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

alphabet = ['a','b','c']
print (alphabet)

# ### atualizando itens em um dicionário

# +
# update item in a dictionary
# dictionary of a sample portfolio
shares = {'APPL': 100, 'GOOG': 50}
# print
print("Shares in your portfolio:", shares)

# update the shares of 'GOOG' to 150
shares['GOOG'] = 150
# print
print("Shares in your portfolio:", shares)
# -

# ### atualizando itens em uma lista de dicionários

# +
my_dicts = [ 
    { 'key1' : 'value1',
      'key2' : 'value2' },

    { 'key3' : 'value3',  
      'key4' : 'value4' },

    { 'key5' : 'value5',  
      'key6' : 'value6' }]

update = {'key2':'value3'}
new_dicts = [{**d,**update} for d in my_dicts]
new_dicts
# -

# ### Outros updates de itens

# +
data = [
    {'name': 'sravan', 'subjects': ['java', 'python']},
    {'name': 'bobby', 'subjects': ['c/cpp', 'java']},
    {'name': 'ojsawi', 'subjects': ['iot', 'cloud']},
    {'name': 'rohith', 'subjects': ['php', 'os']},
    {'name': 'gnanesh', 'subjects': ['html', 'sql']}
]
  
# display first student
print(data[0])
  
# display all student
data


# +

# update first student python subject
# to html
data[0]['subjects'].append('html')
data[0]['subjects'].pop(1)
  
# update third student java subject
# to dbms
data[2]['subjects'].append('dbms')
data[2]['subjects'].pop(1)
  
# update forth student php subject
# to php-mysql
data[3]['subjects'].append('php-mysql')
data[3]['subjects'].pop(0)
  
# display updated list
data

# +

# update first student python subject
# to html
data[0]['subjects'].insert(0, 'html')
data[0]['subjects'].pop(1)
  
# update third student java subject
# to dbms
data[2]['subjects'].insert(0, 'dbms')
data[2]['subjects'].pop(1)
  
# update forth student php subject
# to php-mysql
data[3]['subjects'].insert(1, 'php-mysql')
data[3]['subjects'].pop(0)
  
# display updated list
data
# -

# ### Sorting: Bubble Sort

# +
primeiro_elemento = '1'
segundo_elemento = '1 2 3'

#    n = int(input().strip())
#    a = list(map(int, input().rstrip().split()))

n = int(primeiro_elemento.strip())
a = list(map(int, segundo_elemento.rstrip().split()))
def countSwaps(a):
    swaps=0
    for i in range(len(a)):
        for j in range(len(a)-1):
            if a[j]> a[j+1]:
                a[j],a[j+1]=a[j+1],a[j]
                swaps+=1
    print("Array is sorted in " + str(swaps) + " swaps.")
    print("First Element: " + str(a[0]))
    print("Last Element: " + str(a[len(a)-1]))
countSwaps(a)
# -

# ### ordenação

# +
primeiro_elemento = '1'
segundo_elemento = '1 2 3'

#    n = int(input().strip())
#    a = list(map(int, input().rstrip().split()))

n = int(primeiro_elemento.strip())
a = list(map(int, segundo_elemento.rstrip().split()))

# Write your code here
j = 1
i = 0
swap = []
# Write your code here
for i in range(n):
    currentSwaps = 0
    for j in range(n-1):
        if a[j]>a[j+1]:
            a[j],a[j+1]=a[j+1],a[j]
            swap.append(a[i])
            currentSwaps += 1
    if currentSwaps == 0:
        break

print("Array is sorted in",len(swap),"swaps.")
print("First Element:",a[0])
print("Last Element:",a[-1])
# -

# ### List Comprehensions Hackerrank

x=2
y=2
z=2
n=1
x, y, z, n = int(x), int(y), int(z), int(n)
print ([[a,b,c] for a in range(0,x+1) for b in range(0,y+1) for c in range(0,z+1) if a + b + c != n ])

seasons = ['Spring', 'Summer', 'Fall', 'Winter']
#list(enumerate(seasons))
list(enumerate(seasons, start=10))
x=1
eval('x+1')





# ### ORGANIZING CONTAINERS GUILHERME SILVEIRA

# +
import math
import os
import random
import re
import sys

#
# Complete the 'organizingContainers' function below.
#
# The function is expected to return a STRING.
# The function accepts 2D_INTEGER_ARRAY container as parameter.
#

def organizingContainers(containers):
    n = len(containers)
    capacidade_de_containers = []
    quantidade_de_bolas = [0] * n
    for container in containers:
        total_do_container = sum(container)
        capacidade_de_containers.append(total_do_container)
        for tipo,quantidade in enumerate(container):
            quantidade_de_bolas[tipo] += quantidade
    #sorted(capacidade_de_containers)
    #o sorted não devolve a variável reordenada, só na exibição
    #por isto usamos .sort()
    capacidade_de_containers.sort()
    quantidade_de_bolas.sort()
    capacidade_de_containers.sort()
    if capacidade_de_containers == quantidade_de_bolas:
        return "Possible"
    return "Impossible"



# +
print(organizingContainers([[1,3,1],[2,1,2],[3,3,3]]))

print(organizingContainers([[0,2,1],[1,1,1],[2,0,0]]))

print(organizingContainers([[2,0,0],[1,1,1],[0,2,1]]))


print(organizingContainers([[2,0,0],[1,1,1],[0,2,1]]))
# -

2
3
1 3 1
2 1 2
3 3 3
3
0 2 1
1 1 1
2 0 0


# +
def saveThePrisoner(cadeiras,  inicial, doces):
    # Write your code here
    res = (inicial + doces-1) % cadeiras
    return res if res != 0 else cadeiras
    #return res
    t = int(input().strip())
    for a0 in range(t):
        cadeiras, doces, inicial = input().strip().split(' ')
        cadeiras, doces, inicial = [int(cadeiras), int(doces), int(inicial)]
        result = saveThePrisoner(cadeiras, doces, inicial)
        print(result)

#x=(2 5 5)        

cadeiras=8
inicial=8 
doces=9

saveThePrisoner(cadeiras,inicial,doces)


# -

# # SAVE THE PRISIONER GUILHERME SILVEIRA

# SAVE THE PRISIONER GUILHERME SILVEIRA
# https://www.youtube.com/watch?v=0aRlTx9kh18
def saveThePrisoner(cadeiras, doces, inicial):
    # Write your code here
    sobraram = (doces % cadeiras)
    if sobraram == 0 and inicial == 1:
        return cadeiras
    pessoa = (inicial + sobraram - 1) % cadeiras
    if pessoa == 0:
        return cadeiras
    return pessoa


#teste livre
saveThePrisoner (3,7,3)

#caso normal, 4 cadeiras, 6 doces, inicio na cadeira 1
#caso que hackerrank deu de exemplo
saveThePrisoner (4,6,1)

#caso , 4 cadeiras, 4 doces, inicio na cadeira 1
#caso normal, mesmo número de doces e cadeiras e começa na primeira
saveThePrisoner (4,4,1)

#caso , 4 cadeiras, 4 doces, inicio na cadeira 2
#caso um pouco diferente, começa mais na frente pra ver quando roda a mesa toda e recomeça
saveThePrisoner (4,4,2)

#caso , 4 cadeiras, 4 doces, inicio na cadeira 4
#caso borda, mesmo número de cadeiras e doces e começa na última
saveThePrisoner (4,4,4)

# +
#caso , 4 cadeiras, 1 doce, inicio na cadeira 1
#caso borda, só dá um doce

saveThePrisoner (4,1,1)
# -

#caso , 4 cadeiras, 3 doces, inicio na cadeira 3
#caso borda, mais cadeiras que doces e começa no final
saveThePrisoner (4,3,3)

#caso borda, 3 cadeiras, 7 doces, inicio na cadeira 3
#caso borda, mais que o dobro de doces que cadeiras e começa no final (o teste hackerrank quebrou neste caso)
saveThePrisoner (3,7,3)

# +
# Pulando nuvens
'''
7
0 0 1 0 0 1 0
'''
# você pode pular em cima de qualquer cumulus (0) que esteja uma ou duas posições depois da atual 
# as thunder (1) devem ser evitadas - não pulamos em cima delas

def jumpingOnClouds(c):   #len(c) = 8
    current_position = 0
    number_of_jumps = 0
    last_cloud_postion = len(c)-1  # 7
    last_second_postion = len(c)-2 # 6
              #   0     <      6
    while current_position<last_second_postion:
        #Checking if the cloud next to the next cloud is thunderstorm
        if c[current_position+2] == 0:   #70010010     se a segunda posição pra frente for zero pulamos duas posições
            current_position += 2        #2,4,   7
        else:
            current_position += 1             #5        caso contrário, pulamos só uma posição
        number_of_jumps += 1             # adiciona um salto à sua lista (não é o número de posições que avançamos, simplesmente salto)
    #Checking if we are in the last cloud or the last second cloud
    #se não estamos na última posição ainda, soma mais um salto
    if current_position != last_cloud_postion:   # 7  != 7
        number_of_jumps += 1              # 5
    return number_of_jumps
    
entrada = '7 0 0 1 0 0 1 0'
# entrada='6 0 0 0 0 1 0'
c = list(map(int,entrada.split())) # o c entra sem espaços por conta do split
print(jumpingOnClouds(c))


# +
#DESAFIO HACKERRANK COUNTING VALLEYS

def countingValleys(steps, path):

    level = 0
    valleys = 0
    
    for step in path:
    
        if step == 'U':
            level+=1
                # se ele está subindo Up e atinge o nível do mar 0, então ele conta um vale
            if level == 0:
                valleys+=1
                
        else:
            level-=1
            
    return valleys

countingValleys(8,'UDDDUDUU')
# -



# +
from itertools import combinations
x = "a a c d"

N = 4
L = x.split()
K = 2

C = list(combinations(L, K))
F = filter(lambda c: 'a' in c, C)
print("{0:.3}".format(len(list(F))/len(C)))
# -

#print(C)
print(list(F))

# +
import itertools
  
a_list = [("Animal", "cat"), 
          ("Animal", "dog"), 
          ("Bird", "peacock"), 
          ("Bird", "pigeon"),
          ("Bird", "passaralho")]
  

an_iterator = itertools.groupby(a_list, lambda x : x[0])
  
for key, group in an_iterator:
    key_and_group = {key : list(group)}
    print(key_and_group)

# +
import itertools
  
  
L = [("a", 1), ("a", 2), ("b", 3), ("b", 4)]
  
# Key function
key_func = lambda x: x[0]
  
for key, group in itertools.groupby(L, key_func):
    print(key + " :", list(group))
# -

#Compress the String!
from itertools import groupby
x='1222311'
for k, c in groupby(x):
    print("(%d, %d)" % (len(list(c)), int(k)), end=' ')

#itertools.combinations_with_replacement()
#solucao mais simples 
from itertools import combinations_with_replacement
z = "hack 2"
x=z.split()
s,p=x[0],int(x[1])
y=combinations_with_replacement(sorted(s),p)
for i in (y):
    print(*i,sep="")

#itertools.combinations_with_replacement()
# solucao com list comprehensions
# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import combinations_with_replacement
#s, n = input().split()
x = "hack 2"
s, n = x.split()
print(*[''.join(i) for i in combinations_with_replacement(sorted(s), int(n))], sep="\n")
# aqui é aquele caso que não tem repeticao, por ex, se tem ac não tem ca, se tem ah não tem ha



# +
# itertools.combinations() in Python - Hacker Rank Solution
from itertools import combinations
x = "hack 2"
#s , n  = input().split()
s,n = x.split()

for i in range(1, int(n)+1):
    for j in combinations(sorted(s), i):
        print(''.join(j)) # o join fala o caracter que fica entre os dois quando tiver mais de dois

# +
# itertools.permutations() in Python - Hacker Rank Solution
# Python 3
# itertools.permutations() in Python - Hacker Rank Solution START
from itertools import permutations
x = "hack 2"
#no input digite hack 2
#s,k = input().split()
s,k = x.split()

words = list(permutations(s,int(k)))
words = sorted(words, reverse=False)
for word in words:
    print(*word,sep='')
    
# itertools.permutations() in Python - Hacker Rank Solution END


# +
# Strip split o strip tease da banana split é um jeito de pegar coluna a coluna de uma matriz

my_string = "blah, lots  ,  of ,  spaces, here "
result = [x.strip() for x in my_string.split(',')]
# result is ["blah", "lots", "of", "spaces", "here"]
print(result)

# +
#strip tira os espaços - Remove spaces at the beginning and at the end of the string:
txt = "     banana     "

x = txt.strip()

print("of all fruits", x, "is my favorite")

#Split a string into a list where each word is a list item:

txt = "welcome to the jungle"

x = txt.split()

print(x)

# +
A = [1, 2]
B = [3, 4]

# itertools.product() in Python - Hacker Rank Solution
# Enter your code here. Read input from STDIN. Print output to STDOUT
# itertools.product() in Python - Hacker Rank Solution START
from itertools import product
#A = input().split()
A = list(map(int,A))
#B = input().split()
B = list(map(int, B))
output = list(product(A,B))
for i in output:
    print(i, end = " ");
# itertools.product() in Python - Hacker Rank Solution END
print(output)

# +
# #!/bin/python3
# o legal aqui é que o script pega as colunas da matriz e vai percorrendo e trocando símbolos esquisitos por espaço
import math
import os
import random
import re
import sys


first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)

# start   
matrix = list(zip(*matrix))

sample = str()

for words in matrix:
    for char in words:
        sample += char
       
print(re.sub(r'(?<=\w)([^\w\d]+)(?=\w)', ' ', sample))

#no input aqui embaixo vá colocando linha a linha
'''
7 3
Tsi
h%x
i #
sM 
$a 
#t%
ir!
'''

# +

# validando cep, tem que estar entre 100000 e 999999 e não pode repetir números separados por um outro,
# por exemplo 101 929

#regex_integer_in_range = r"_________"	# Do not delete 'r'.
#regex_alternating_repetitive_digit_pair = r"_________"	# Do not delete 'r'.



regex_integer_in_range = r"^[1-9][\d]{5}$"    # Do not delete 'r'. 
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"    # Do not delete 'r'.
# Validating Postal Codes in Python - Hacker Rank Solution END



import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)
# -

# Validating Credit Card Numbers in Python Hacker Rank Solution
# Python 3
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Validating Credit Card Numbers in Python Hacker Rank Solution START
import re
for i in range(int(input())):
    S = input().strip()
    pre_match = re.search(r'^[456]\d{3}(-?)\d{4}\1\d{4}\1\d{4}$',S)
    if pre_match:
        processed_string = "".join(pre_match.group(0).split('-'))
        final_match = re.search(r'(\d)\1{3,}',processed_string)
        if final_match:
            print('Invalid')
        else :
            print('Valid')
    else:
        print('Invalid')
# Validating Credit Card Numbers in Python Hacker Rank Solution END 

# +
#outra solucao cartao de credito
import re

def check(card):
    if not re.search("^[456]\d{3}(-?\d{4}){3}$",card) or re.search(r"(\d)\1{3}", re.sub("-", "",card)):
        return False
    return True

for i in range(int(input())):
    print("Valid" if check(input()) else "Invalid")

# -

#imprimir a sequencia de números de 1 até o número digitado, tem que ser colados
print(*range(1, int(input())+1), sep='')
#parece que o asterisco é para remover espaços


#imprimir a sequencia de números de 1 até o número digitado, tem que ser colados
#OUTRA SOLUCAO
n = int(input())
for i in range(1, n+1):
    print(i, end="")

# +

STDIN = input()
import re
for _ in range(int(STDIN())):
    s=STDIN().strip()
    print('Valid' if re.search(r'[A-Z].*[A-Z]',s) and re.search(r'[0-9].*[0-9].*[0-9]',s) and re.search(r'^[0-9a-zA-Z]{10}$',s) and not re.search(r'(.).*\1',s) else 'Invalid')
# -

for i in xrange(10):
    print i,

# +
#hackerrank opares e impares

# #!/bin/python3

import sys

#If  is odd(impar), print Weird
#If  is even(par) and in the inclusive range of 2 to 5 , print Not Weird
#If  is even and in the inclusive range of 6 to 20 , print Weird
#If  is even and greater than 20, print Not Weird

N = int(input().strip())
if N % 2 == 1 or N >= 6 and N <= 20:
    print("Weird")
elif N >= 2 and N <= 5 or N >= 20:
    print("Not Weird")
# -

#Encontrar o elemento de valor 10 na lista [1, 2, 10, 5, 20]
#e retornar a posição em que ele foi encontrado
lista = [1, 2, 10, 5, 20]
valor = 10
pos = -1
for i in range(len(lista)-1,-1,-1):
     if lista[i] == valor:
         pos = i
print(pos)

lista = [1, 2, 3, 4]
#remove a segunda posicao - lembre que comeca na 0
lista.pop(2)
print(lista)
print (lista[2])

lista = [1, 4, 5, 6, 4, 7]
lista.remove(4)
print(lista)


#crie uma funçao chamada quadrado para calcular o quadrado
def quadrado(x):
    quadrado = x*x
    return (quadrado)
quadrado(9)


def cubo(x):
    return x ** 3
cubo(3)

type (cubo)

#faz função lambda para calcular quadrado
lambda x: x ** 2

quadrado (5)

#Quadrado de 5 usando funcao lambda
#faz função lambda para calcular quadrado
(lambda x: x ** 2)(5)

#chamando a funçäo lambda de minha_funcao
minha_funcao = lambda x: x ** 2
minha_funcao(6)


lista = [1, 2, 10, 5, 20]
pos = lista.index(10)
print(pos)

#Quando o elemento procurado não está na lista, o
#método index lança uma exceção
lista = [1, 2, 3, 4]
lista.index(7)

lista = [1, 2, 3, 4]
resultado = 7 in lista
print(resultado)


resultado = 3 in lista
print(resultado)


lista = [10, 9, 8, 7, 5, 3, 4, 3, 1, 2, 11]
lista.sort()
print(lista)

lista = [1, 3, 2, 4]
lista.reverse()
print(lista)

lista = ['a', 'b', 'c', 'd', 'e']
lista[2:]

lista [:3]

lista[:0] 

lista[1:3] 
lista[1:-1] 

lista = [1,2,3,4,5]
lista[1:3] = ['a', 'b'] 
lista











import pandas as pd

import os
#note que as barras aqui embaixo são para a direita
os.chdir("C:/Users/gustavo/Downloads/taxigov-users-geral-2021-09")
#df  =  pd.read_csv('tipo1.csv', parse_dates=['lanData'])
#df  =  pd.read_csv('tipo1.csv', parse_dates= ['lanData'],encoding='utf-8-sig', usecols= ['lanData', 'lanCod'],)
#df  =  pd.read_csv('taxigov_users-geral.csv', parse_dates= ['lanData'],encoding='utf-8-sig')
df  =  pd.read_csv('taxigov_users-geral.csv',encoding='utf-8-sig')
df.head()


