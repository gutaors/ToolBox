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

# +
#itertools.combinations_with_replacement()

# Enter your code here. Read input from STDIN. Print output to STDOUT
from itertools import combinations_with_replacement
#s, n = input().split()
x = "hack 2"
s, n = x.split()
print(*[''.join(i) for i in combinations_with_replacement(sorted(s), int(n))], sep="\n")
# aqui é aquele caso que não tem repeticao, por ex, se tem ac não tem ca, se tem ah não tem ha
# -



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


