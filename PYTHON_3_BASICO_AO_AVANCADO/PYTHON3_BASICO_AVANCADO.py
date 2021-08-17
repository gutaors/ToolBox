# prints
print("abc")
print("clica shift enter")
print("entre o codigo e comentario tem dois espaços")  # Dois espaços antes da cerquilha
print('aspas simples também funcionam')

"""
COMENTARIO
MULTILINHAS
não é chamado de comentário mas documentação
"""

print('depois do final do arquivo é bom deixar uma linha, basta dar enter')
# o que a vírgula faz em um print?
print('virgula coloca','um espaço')
# como eu uso a vígula para substituir o espaço por outro separador? (ao inves de espaço colocar - p ex)
print('virgula também pode colocar','outro separador',sep='-')
#como separo do print da linha de cima?
print('ele já coloca enter para separar do print da linha de cima')
#como mudo o que separa da linha de cima, usando outro símbolo?
print('também pode mudar o que separa da linha de cima','outro separador',sep='-',end='')
#muda para o separador ser - e a linha de cima não ter enter pra de baixo
print('repetindo','linha1',sep='-',end='')
print('linha2',end='')
#print(('924','176','070',sep='.'),'18',sep='-')
# como eu printo o tipo de um valor ou variável?
print(type('luiz'))
print(type(10==10))
print(type(253.22))

# como fazer um dicionário de pares de chave/valore?
a_dict = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}

# como vou interando e imprimindo somente as chaves de um dicionário?
for key in a_dict:
     print(key)
# como eu intero e imprimo os chaves -> valores de um dicionário exatamente neste formato ch -> val ?
for key in a_dict:
     print(key, '->', a_dict[key])

d_items = a_dict.items()
d_items  # Here d_items is a view of items


for item in a_dict.items():
     print(item)


# github do instrutor
# https://gist.github.com/luizomf

#[ ]:

#!pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install --user
# [16]:

idade = 49
#[18]:

nome = input("qual o seu nome?")
​
#qual o seu nome? gu
# [19]:

sobre = input("e o sobrenome?")
​
#e o sobrenome? stavo

print(nome,sobre)

ano_nascimento = 2021-int(idade)

ano_nascimento

print()
print(f'{nome} tem {idade} anos ' 
      f'{nome} nasceu em {ano_nascimento}')

#gu tem 49 anos gu nasceu em 1972


numero_1 = int(input('digite um número'))
#digite um número 2


numero_2 = int(input('digite outro número'))
numero_2 = int(numero_2)
#digite outro número a
#---------------------------------------------------------------------------
#ValueError                                Traceback (most recent call last)
#/tmp/ipykernel_56/2776762699.py in <module>
#----> 1 numero_2 = int(input('digite outro número'))
#      2 numero_2 = int(numero_2)

#ValueError: invalid literal for int() with base 10: 'a'

print(numero_1+numero_2)

if False:
    print("foi falso")
elif True:
    print("agora é verdadeiro")
    print("posso colocar um monte de código aqui")
elif False:
    print("outra falsa")
else:
    print("nem um nem outro")
    
    
# operadores relacionais ==, >, <, >=, <=, !=
#operadores lógicos and, or, not, in, not in
#agora é verdadeiro
#posso colocar um monte de código aqui

nome = input('qual o seu nome?')
idade = int(input('qual sua idade?'))
qual o seu nome? gustavo
qual sua idade? 19

idade_limite = 18
if idade >= idade_limite:
    print(f'{nome} pode dirigir')
else:
    print(f'{nome} não pode dirigir')

usuario = input('digite user')
qtd = len(usuario)
print(usuario, qtd, type(qtd))
#digite user alds
#alds 4 <class 'int'>

#Lists Comprehension
#vamos criar lista l1 e a l2 que vai interando na l1 item por item e montando,
# então a l2 é igual à l1
l1 = [1,2,3,4,5,6,7,8,9]
l2 = [variavel for variavel in l1]


 
l1 = [1,2,3,4,5,6,7,8,9]
ex1 = [variavel for variavel in l1]
print(ex1)
ex2 = [v*2 for v in l1]
print(ex2)
ex3 = [(v, v2) for v in l1 for v2 in range (3)]
print(ex3)
l2 = ['Luiz','Mauro','Maria']
ex4 = [v.replace('a','@').upper() for v in l2]
print(ex4)
tupla = (
	('chave', 'valor'), 
	('chave2', 'valor2'),
)
 
 
ex5 = [(x,y) for  x,y in tupla]
ex5 = dict(ex5)
print(ex5['valor2'])
 
l3=list(range(100))
#todos os números de 0 a 99 que são divisíveis por 3 e por 8
ex6 = [v for v in l3 if v % 3 == 0 if v % 8 == 0]
print(ex6)
 
ex7 = [v if v % 3 == 0 else 'não é' for v in l3]
print(ex7)
 
 
ex7 = [v if v % 3 == 0 and v % 8 == 0 else 0 for v in l3]
print(v7)
 
print(l3)
 
print (ex3)

####################################################################################################
#também usei o list comprehension no airflow, olha só

files = [os.path.join(
        _get_and_make_local_tmp_dir(**context),
        file_name) for file_name in files_changed]  #files_changed é um parametro recebido pela def

####################################################################################################    
#o files aqui embaixo vem da list comprehension acima 
 
    send_email(to=EMAIL_NOTIFY_LIST,
            cc=EMAIL_CC_LIST,
            subject=subject,
            files=files,
            html_content=replace_to_html_encode(content)
            )
####################################################################################################
#tirei estas funções daqui
# tem uma função chamada
def _zip_files(files_list: list, **context):
    #lá no miolo dela tem onde ele monta os nomes dos excel que são a aprtir do os.path.join(... logo abaixo
    for file_name in files_list:
        file_path = os.path.join(
            _get_and_make_local_tmp_dir(**context),
            file_name
        )
         # Apagando arquivo local
        #os.remove(file_path)


# Dictionary comprehension (dicionários compreensivos)
l1 = [1,2,3,4,5,6]
# como multiplico todos valores por 2?
l2 = [v*2 for v in l1]
print(l2)
lista = [
    ('chave','valor'),
    ('chave2','valor2'),
]
#vamos fazer uma compreensao de dicionario com a lista acima

d1 = {x:y for x, y in lista}
#linha acima é criei uma chave (x) e(:) valor (y) 
# para (for) chave, valor (x, y) que estão na lista

d1 = {x: y*2 for x, y in lista}
print(d1)


#outro exemplo
lista = [
    ('chave',2),
    ('chave2','valor2'),
]
d1 = {x: y*2 for x, y in lista}
print(d1)


# outro
lista = [
    ('chave','valor'),
    ('chave2','valor2'),
]

d1 = {x.upper(): y.upper() for x,y in lista}
print(d1)

d1 = {x.upper(): y.upper() for x,y in enumerate(range(5))}
print(d1)

d1 = {x.upper(): y.upper() for x,y in enumerate(range(5))}
print(d1)

d1 = {x for x in range(5)}
print(d1, type(d1)) # set comprehensions ou compreensoes de conjuntos

d1 = {f'chave_{x}':x**2 for x in range(5)}
print(d1, type(d1))
